import os
from datetime import datetime
from html import escape
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="WattWise", layout="wide")

DATA_FILE = os.path.join(os.path.dirname(__file__), "sample_raw_energy.csv")
BAD_KWH = 50000
SOURCE_COLORS = {"Grid": "#4C78A8", "Solar": "#F2B134", "Diesel Generator": "#B23A48"}
M_TREND = "Linear trend"
M_AVG = "Recent average (last 8 readings)"


@st.cache_data
def clean(raw):
    df = raw.copy()
    df.columns = [column.strip().lower() for column in df.columns]
    for column, default in {"peak_kw": np.nan, "voltage": np.nan,
                            "source": "Grid", "occupancy": np.nan}.items():
        if column not in df.columns:
            df[column] = default

    quality_log = [("Rows in file", len(df))]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for column in ["units_kwh", "peak_kw", "voltage", "occupancy"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["building"] = df["building"].astype("string").str.strip()
    quality_log.append(("Invalid timestamps excluded", int(df["timestamp"].isna().sum())))
    df = df.dropna(subset=["timestamp"])
    missing_building = df["building"].isna() | df["building"].eq("")
    quality_log.append(("Rows with missing building labels excluded", int(missing_building.sum())))
    df = df[~missing_building]

    original_count = len(df)
    df = df.drop_duplicates(subset=["timestamp", "building"], keep="first")
    quality_log.append(("Duplicate rows removed", original_count - len(df)))

    valid_kwh = (df["units_kwh"] > 0) & (df["units_kwh"] <= BAD_KWH)
    df["kwh_actual"] = df["units_kwh"].where(valid_kwh)
    df["peak_kw_actual"] = df["peak_kw"].where(df["peak_kw"] > 0)
    invalid_peak = df["peak_kw"].notna() & (df["peak_kw"] <= 0)
    quality_log.append(("Non-positive peak_kw values excluded", int(invalid_peak.sum())))
    df.loc[invalid_peak, "peak_kw"] = np.nan

    source = df["source"].astype("string").str.strip()
    source_before = source.nunique()
    df["source"] = source.fillna("Unknown").replace("", "Unknown").str.title()
    quality_log.append(("Source labels unified", f"{source_before} to {df['source'].nunique()}"))
    df = df.sort_values(["building", "timestamp"]).reset_index(drop=True)

    invalid_kwh = df["units_kwh"].notna() & ~valid_kwh
    quality_log.append(("Impossible kWh values excluded", int(invalid_kwh.sum())))
    df.loc[invalid_kwh, "units_kwh"] = np.nan
    missing_kwh = df["units_kwh"].isna()
    quality_log.append(("Missing or invalid kWh values estimated for legacy analysis", int(missing_kwh.sum())))
    df["kwh"] = df.groupby("building")["units_kwh"].transform(
        lambda values: values.interpolate(limit_direction="both"))
    df["kwh"] = df["kwh"].fillna(df.groupby("building")["units_kwh"].transform("median"))
    df["kwh_fixed"] = missing_kwh

    quality_log.append(("Missing peak_kw values estimated for legacy analysis", int(df["peak_kw"].isna().sum())))
    df["peak_kw"] = df["peak_kw"].fillna(df.groupby("building")["peak_kw"].transform("median"))
    quality_log.append(("Negative voltage readings corrected", int((df["voltage"] < 0).sum())))
    df["voltage"] = df["voltage"].abs()
    df = df.dropna(subset=["kwh"])
    quality_log.append(("Clean rows used", len(df)))
    df["date"] = df["timestamp"].dt.date
    df["month"] = df["timestamp"].dt.strftime("%Y-%m")
    quality = pd.DataFrame([(name, str(value)) for name, value in quality_log],
                           columns=["Check", "Count"])
    return df, quality


def actual_summary(frame):
    measured = frame[frame["kwh_actual"].notna()].copy()
    building_totals = measured.groupby("building", as_index=False).agg(
        observations=("kwh_actual", "count"), total_kwh=("kwh_actual", "sum"),
        average_kwh=("kwh_actual", "mean"), average_peak_kw=("peak_kw_actual", "mean"),
        max_peak_kw=("peak_kw_actual", "max"))
    monthly = measured.groupby("month", as_index=False).agg(
        total_kwh=("kwh_actual", "sum"), average_kwh=("kwh_actual", "mean"),
        observations=("kwh_actual", "count")).sort_values("month")
    sources = measured.groupby("source", as_index=False).agg(
        kwh=("kwh_actual", "sum"), observations=("kwh_actual", "count"))
    return measured, building_totals, monthly, sources


def answer_energy_question(question, frame):
    measured, building_totals, monthly, sources = actual_summary(frame)
    if measured.empty:
        return "No valid recorded energy readings are available in the current filters."
    query = question.casefold()
    building = next((name for name in building_totals["building"]
                     if name.casefold() in query), None)
    if building:
        readings = measured[measured["building"] == building]
        stats = building_totals[building_totals["building"] == building].iloc[0]
        answer = (f"{building} has {stats['total_kwh']:,.1f} kWh across "
                  f"{int(stats['observations'])} valid recorded readings, averaging "
                  f"{stats['average_kwh']:,.1f} kWh per reading.")
        peaks = readings.dropna(subset=["peak_kw_actual"])
        if len(peaks):
            peak = peaks.loc[peaks["peak_kw_actual"].idxmax()]
            answer += f" Highest recorded peak: {peak['peak_kw_actual']:,.1f} kW on {peak['timestamp']:%Y-%m-%d}."
        return answer
    if any(word in query for word in ("peak", "demand", "maximum kw")):
        peaks = measured.dropna(subset=["peak_kw_actual"])
        if peaks.empty:
            return "Valid peak-demand readings are unavailable in this selection."
        peak = peaks.loc[peaks["peak_kw_actual"].idxmax()]
        return f"Highest recorded peak: {peak['peak_kw_actual']:,.1f} kW at {peak['building']} on {peak['timestamp']:%Y-%m-%d}."
    if any(word in query for word in ("source", "solar", "grid", "diesel", "mix")):
        source = sources.sort_values("kwh", ascending=False).iloc[0]
        shares = ", ".join(f"{row.source}: {row.kwh / measured['kwh_actual'].sum():.0%}"
                            for row in sources.itertuples())
        return f"Recorded energy totals {measured['kwh_actual'].sum():,.1f} kWh. Largest source: {source['source']} ({source['kwh']:,.1f} kWh). Shares: {shares}."
    if any(word in query for word in ("trend", "month", "increased", "decreased")):
        if len(monthly) < 2:
            return "There are not enough months of valid readings in this selection to describe a trend."
        first, last = monthly.iloc[0], monthly.iloc[-1]
        change = (last["average_kwh"] / first["average_kwh"] - 1) * 100
        direction = "increased" if change > 0 else "decreased" if change < 0 else "was unchanged"
        return f"Average recorded energy per reading {direction} by {abs(change):.1f}% from {first['month']} to {last['month']}."
    highest = building_totals.sort_values("total_kwh", ascending=False).iloc[0]
    return f"{highest['building']} has the highest recorded total: {highest['total_kwh']:,.1f} kWh from {int(highest['observations'])} valid readings."


def report_table(rows, widths):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#c2410c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d6d3d1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fff7ed")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def build_report_pdf(frame, quality_frame, tariffs, emission_factors):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    measured, building_totals, monthly, sources = actual_summary(frame)
    output = BytesIO()
    document = SimpleDocTemplate(output, pagesize=letter, rightMargin=38, leftMargin=38,
                                 topMargin=42, bottomMargin=40, title="WattWise Campus Energy Report")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="WattTitle", parent=styles["Title"],
                              textColor=colors.HexColor("#9a3412"), alignment=0, spaceAfter=8))
    styles.add(ParagraphStyle(name="WattSection", parent=styles["Heading2"],
                              textColor=colors.HexColor("#c2410c"), spaceBefore=12, spaceAfter=6))
    story = [Paragraph("WattWise Campus Energy Report", styles["WattTitle"]),
             Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M}", styles["Normal"]), Spacer(1, 12)]
    total = measured["kwh_actual"].sum()
    peak_rows = measured.dropna(subset=["peak_kw_actual"])
    known_cost = measured["kwh_actual"] * measured["source"].map(tariffs)
    known_co2 = measured["kwh_actual"] * measured["source"].map(emission_factors)
    peak_summary = (f"{peak_rows['peak_kw_actual'].max():,.1f} kW maximum from {len(peak_rows)} readings"
                    if len(peak_rows) else "Unavailable")
    story.extend([
        Paragraph("Executive Summary", styles["WattSection"]),
        Paragraph(f"{total:,.1f} kWh from {len(measured)} valid recorded readings across "
                  f"{measured['building'].nunique()} buildings. Missing and invalid energy readings are excluded.",
                  styles["BodyText"]),
        Paragraph("Campus Statistics", styles["WattSection"]),
        report_table([["Metric", "Result"], ["Recorded energy", f"{total:,.1f} kWh"],
                      ["Valid readings", f"{len(measured):,}"], ["Peak demand", peak_summary],
                      ["Estimated cost", f"Rs {known_cost.sum():,.0f}"],
                      ["Estimated emissions", f"{known_co2.sum():,.1f} kg CO2"]],
                     [2.0 * inch, 4.6 * inch]),
        Paragraph("Building Summary", styles["WattSection"]),
    ])
    building_rows = [["Building", "Readings", "Total kWh", "Avg kWh", "Avg peak kW"]]
    for row in building_totals.sort_values("total_kwh", ascending=False).itertuples():
        building_rows.append([row.building, f"{row.observations:,}", f"{row.total_kwh:,.1f}",
                              f"{row.average_kwh:,.1f}",
                              f"{row.average_peak_kw:,.1f}" if pd.notna(row.average_peak_kw) else "Unavailable"])
    story.append(report_table(building_rows, [1.6 * inch, 0.8 * inch, 1.3 * inch, 1.3 * inch, 1.6 * inch]))
    story.extend([Paragraph("Monthly Recorded Energy", styles["WattSection"]),
                  report_table([["Month", "Total kWh", "Average kWh", "Observations"]] +
                               [[row.month, f"{row.total_kwh:,.1f}", f"{row.average_kwh:,.1f}",
                                 f"{row.observations:,}"] for row in monthly.itertuples()],
                               [1.4 * inch, 1.7 * inch, 1.8 * inch, 1.7 * inch]),
                  Paragraph("Findings and Sources", styles["WattSection"])])
    if len(building_totals):
        top = building_totals.sort_values("total_kwh", ascending=False).iloc[0]
        story.append(Paragraph(f"Highest building total: {escape(top['building'])} ({top['total_kwh']:,.1f} kWh).",
                               styles["BodyText"]))
    story.append(report_table([["Source", "Recorded kWh", "Readings"]] +
                              [[row.source, f"{row.kwh:,.1f}", f"{row.observations:,}"]
                               for row in sources.itertuples()], [2.2 * inch, 2.2 * inch, 2.2 * inch]))
    story.extend([Paragraph("Data Quality and Limitations", styles["WattSection"]),
                  report_table([["Check", "Result"]] +
                               [[escape(str(row.Check)), escape(str(row.Count))]
                                for row in quality_frame.itertuples(index=False)], [4.3 * inch, 2.3 * inch]),
                  Spacer(1, 8),
                  Paragraph("Energy totals use valid recorded readings only. Cost and emissions are estimates "
                            "using configured rates; unknown-source readings are excluded from these estimates. "
                            "They are not utility bills or direct emissions measurements. Legacy forecasts use "
                            "estimated values where indicated in the data-quality log.", styles["BodyText"])])
    document.build(story)
    return output.getvalue()


def predict(method, train, horizon):
    if method == M_TREND:
        x_values = np.arange(len(train))
        slope, intercept = np.polyfit(x_values, train, 1)
        return slope * np.arange(len(train), len(train) + horizon) + intercept
    return np.repeat(train[-8:].mean(), horizon)


def forecast_building(frame, horizon, test_count=4):
    ordered = frame.sort_values("timestamp")
    values = ordered["kwh"].to_numpy(float)
    if len(values) < 12:
        return None
    timestamps = ordered["timestamp"]
    step = timestamps.diff().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        step = pd.Timedelta(days=7)
    future = pd.DatetimeIndex([timestamps.iloc[-1] + step * (index + 1)
                               for index in range(horizon)])
    errors = {}
    for method in (M_TREND, M_AVG):
        prediction = predict(method, values[:-test_count], test_count)
        errors[method] = float(np.mean(np.abs((values[-test_count:] - prediction) /
                                               values[-test_count:])) * 100)
    selected_method = min(errors, key=errors.get)
    return timestamps, values, future, predict(selected_method, values, horizon), selected_method, errors[selected_method]


st.sidebar.header("Data")
upload = st.sidebar.file_uploader("Upload a CSV (optional)", type="csv")
if upload is not None:
    raw = pd.read_csv(upload)
else:
    raw = pd.read_csv(DATA_FILE)
    st.sidebar.success("Using sample_raw_energy.csv")
required_columns = {"timestamp", "building", "units_kwh"}
if not required_columns.issubset({column.strip().lower() for column in raw.columns}):
    st.error("The CSV needs timestamp, building, and units_kwh columns.")
    st.stop()

df_all, quality = clean(raw)

st.sidebar.header("Rates")
tariff = {
    "Grid": st.sidebar.number_input("Grid tariff (Rs per kWh)", 0.0, 100.0, 8.0, 0.5),
    "Diesel Generator": st.sidebar.number_input("Diesel generator cost (Rs per kWh)", 0.0, 200.0, 25.0, 1.0),
    "Solar": st.sidebar.number_input("Solar running cost (Rs per kWh)", 0.0, 100.0, 0.0, 0.5),
}
with st.sidebar.expander("CO2 factors (kg per kWh)"):
    emission_factors = {
        "Grid": st.number_input("Grid CO2 factor", 0.0, 3.0, 0.71, 0.01),
        "Diesel Generator": st.number_input("Diesel CO2 factor", 0.0, 3.0, 0.90, 0.01),
        "Solar": st.number_input("Solar CO2 factor", 0.0, 3.0, 0.0, 0.01),
    }


def robust_z(values):
    median = values.median()
    mad = (values - median).abs().median()
    scale = 1.4826 * mad if mad > 0 else values.std()
    return (values - median) / scale if scale and scale > 0 else values * 0


st.sidebar.header("Filters")
threshold = st.sidebar.slider("Anomaly sensitivity (lower = more alerts)", 2.0, 5.0, 2.5, 0.5)
df_all["typical"] = df_all.groupby("building")["kwh"].transform("median")
df_all["z"] = df_all.groupby("building")["kwh"].transform(robust_z)
df_all["anomaly"] = (df_all["z"] > threshold) & (~df_all["kwh_fixed"])
df_all["cost"] = df_all["kwh"] * df_all["source"].map(tariff).fillna(tariff["Grid"])
df_all["co2"] = df_all["kwh"] * df_all["source"].map(emission_factors).fillna(emission_factors["Grid"])
buildings = sorted(df_all["building"].unique())
selected_buildings = st.sidebar.multiselect("Buildings", buildings, default=buildings)
first_date, last_date = df_all["date"].min(), df_all["date"].max()
selected_dates = st.sidebar.date_input("Date range", (first_date, last_date),
                                        min_value=first_date, max_value=last_date)
base = df_all[df_all["building"].isin(selected_buildings)]
d = base
if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    d = base[(base["date"] >= selected_dates[0]) & (base["date"] <= selected_dates[1])]
if d.empty:
    st.warning("No data for this selection. Choose at least one building and a valid date range.")
    st.stop()

measured, building_summary, monthly_summary, source_summary = actual_summary(d)
if measured.empty:
    st.warning("No valid recorded kWh readings are available in this selection.")
    st.stop()
actual_energy = measured["kwh_actual"].sum()
known_cost = (measured["kwh_actual"] * measured["source"].map(tariff)).sum()
known_emissions = (measured["kwh_actual"] * measured["source"].map(emission_factors)).sum()
solar_share = measured.loc[measured["source"] == "Solar", "kwh_actual"].sum() / actual_energy
recorded_peaks = measured.dropna(subset=["peak_kw_actual"])

st.title("⚡ WattWise")
st.caption(f"{measured['building'].nunique()} buildings, {len(measured)} valid recorded readings, "
           f"{measured['date'].min()} to {measured['date'].max()}. Estimates are excluded from the dashboard.")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Recorded energy", f"{actual_energy:,.0f} kWh")
k2.metric("Estimated cost", f"Rs {known_cost:,.0f}")
k3.metric("Estimated CO2", f"{known_emissions / 1000:,.2f} t")
k4.metric("Solar share", f"{solar_share:.0%}")
k5.metric("Highest recorded peak", f"{recorded_peaks['peak_kw_actual'].max():,.1f} kW" if len(recorded_peaks) else "Unavailable")
k6.metric("Recorded readings", f"{len(measured):,}")

tabs = st.tabs(["Dashboard", "Campus Energy Map", "Anomalies", "Forecast",
                "Energy Mix & Carbon", "Recommendations", "Data Quality", "Campus Energy Report"])

with tabs[0]:
    trend_fig = px.line(measured, x="timestamp", y="kwh_actual", color="building", markers=True,
                        title="Recorded energy per reading (kWh)")
    trend_fig.update_yaxes(title="kWh")
    st.plotly_chart(trend_fig, width="stretch")
    left, right = st.columns(2)
    left.plotly_chart(px.bar(building_summary.sort_values("total_kwh"), x="total_kwh", y="building",
                             orientation="h", color="building", title="Recorded energy by building (kWh)"),
                      width="stretch")
    right.plotly_chart(px.pie(source_summary, values="kwh", names="source", hole=0.45,
                              color="source", color_discrete_map=SOURCE_COLORS,
                              title="Recorded energy by source"), width="stretch")
    peak_chart = px.bar(building_summary.sort_values("average_peak_kw"), x="average_peak_kw",
                        y="building", orientation="h", color="building",
                        title="Average recorded peak demand by building (kW)")
    st.plotly_chart(peak_chart, width="stretch")
    occupancy = measured.groupby("building").agg(kwh=("kwh_actual", "sum"),
                                                  occupants=("occupancy", "sum"))
    occupancy = occupancy[occupancy["occupants"] > 0]
    if len(occupancy):
        occupancy["kwh_per_person"] = occupancy["kwh"] / occupancy["occupants"]
        st.plotly_chart(px.bar(occupancy.reset_index().sort_values("kwh_per_person"),
                               x="building", y="kwh_per_person", color="building",
                               title="Energy per occupant (kWh per person)"), width="stretch")

with tabs[1]:
    st.subheader("Campus Energy Map")
    st.caption("Schematic building layout; positions do not represent geographic locations.")
    map_buildings = building_summary["building"].tolist()
    positions = pd.DataFrame({"building": map_buildings,
                              "x": [index % 3 for index in range(len(map_buildings))],
                              "y": [-(index // 3) for index in range(len(map_buildings))]})
    map_data = positions.merge(building_summary, on="building", how="left")
    map_figure = go.Figure(go.Scatter(
        x=map_data["x"], y=map_data["y"], mode="markers+text",
        text=map_data["building"], textposition="bottom center",
        customdata=map_data[["building", "total_kwh", "observations", "average_peak_kw"]],
        marker=dict(size=42, color=map_data["total_kwh"], colorscale="Oranges", showscale=True,
                    colorbar=dict(title="kWh"), line=dict(color="#9a3412", width=1)),
        hovertemplate="<b>%{customdata[0]}</b><br>Recorded energy: %{customdata[1]:,.1f} kWh"
                      "<br>Observations: %{customdata[2]}<br>Average recorded peak: %{customdata[3]:,.1f} kW<extra></extra>"))
    map_figure.update_layout(height=390, title="Building energy schematic", showlegend=False,
                             xaxis=dict(visible=False), yaxis=dict(visible=False),
                             margin=dict(l=10, r=10, t=50, b=45))
    st.plotly_chart(map_figure, width="stretch")
    selected_building = st.selectbox("Building details", map_buildings, key="map_building")
    selected_data = measured[measured["building"] == selected_building]
    selected_stats = building_summary[building_summary["building"] == selected_building].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recorded energy", f"{selected_stats['total_kwh']:,.1f} kWh")
    c2.metric("Average per reading", f"{selected_stats['average_kwh']:,.1f} kWh")
    c3.metric("Average recorded peak", f"{selected_stats['average_peak_kw']:,.1f} kW" if pd.notna(selected_stats["average_peak_kw"]) else "Unavailable")
    c4.metric("Valid observations", int(selected_stats["observations"]))
    st.plotly_chart(px.line(selected_data, x="timestamp", y="kwh_actual", markers=True,
                            title=f"{selected_building}: recorded consumption"), width="stretch")

    monthly_means = measured.pivot_table(index="building", columns="month", values="kwh_actual", aggfunc="mean")
    monthly_counts = measured.pivot_table(index="building", columns="month", values="kwh_actual", aggfunc="count")
    monthly_counts = monthly_counts.reindex(index=monthly_means.index, columns=monthly_means.columns)
    heatmap = go.Figure(go.Heatmap(
        z=monthly_means.to_numpy(), x=monthly_means.columns.tolist(), y=monthly_means.index.tolist(),
        customdata=monthly_counts.to_numpy(),
        colorscale=[[0, "#ffcc80"], [0.5, "#ff6d00"], [1, "#b71c1c"]], colorbar=dict(title="Avg kWh"),
        hovertemplate="Building: %{y}<br>Month: %{x}<br>Average energy: %{z:.1f} kWh"
                      "<br>Observations: %{customdata}<extra></extra>"))
    heatmap.update_layout(title="Monthly recorded energy heatmap", xaxis_title="Month",
                          yaxis_title="Building", height=max(360, 48 * len(monthly_means)))
    st.plotly_chart(heatmap, width="stretch")

with tabs[2]:
    anomaly_building = st.selectbox("Building", selected_buildings, key="anomaly_building")
    group = d[d["building"] == anomaly_building].sort_values("timestamp")
    anomaly_figure = go.Figure()
    anomaly_figure.add_scatter(x=group["timestamp"], y=group["kwh"], mode="lines+markers", name="Usage")
    anomaly_figure.add_hline(y=group["typical"].iloc[0], line_dash="dot", annotation_text="typical median")
    flagged = group[group["anomaly"]]
    anomaly_figure.add_scatter(x=flagged["timestamp"], y=flagged["kwh"], mode="markers", name="Anomaly",
                               marker=dict(color="red", size=13, symbol="x"))
    anomaly_figure.update_layout(title=f"{anomaly_building}: usage with anomalies", yaxis_title="kWh")
    st.plotly_chart(anomaly_figure, width="stretch")
    flagged_rows = d[d["anomaly"]].copy()
    if flagged_rows.empty:
        st.success("No anomalies at this sensitivity. Lower the sidebar slider to see more.")
    else:
        flagged_rows["extra_kwh"] = (flagged_rows["kwh"] - flagged_rows["typical"]).round(1)
        flagged_rows["why_flagged"] = flagged_rows.apply(
            lambda row: f"{row.kwh:.0f} kWh vs typical {row.typical:.0f} kWh", axis=1)
        st.dataframe(flagged_rows.sort_values("extra_kwh", ascending=False)[
            ["timestamp", "building", "source", "why_flagged", "extra_kwh"]], hide_index=True)
    st.caption("Robust z-score within each building; imputed readings are never flagged.")

with tabs[3]:
    horizon = st.slider("Readings to forecast ahead (weeks for weekly data)", 4, 12, 8)
    forecast_building_name = st.selectbox("Building", selected_buildings, key="forecast_building")
    forecast_result = forecast_building(base[base["building"] == forecast_building_name], horizon)
    if forecast_result is None:
        st.info("At least 12 readings are needed to forecast this building.")
    else:
        timestamps, values, future, predictions, method, error = forecast_result
        forecast_figure = go.Figure()
        forecast_figure.add_scatter(x=timestamps, y=values, mode="lines+markers", name="Actual / legacy cleaned")
        forecast_figure.add_scatter(x=future, y=predictions, mode="lines+markers", name="Forecast",
                                    line=dict(dash="dash"))
        forecast_figure.update_layout(title=f"{forecast_building_name}: demand forecast", yaxis_title="kWh per reading")
        st.plotly_chart(forecast_figure, width="stretch")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Forecast, next {horizon}", f"{predictions.sum():,.0f} kWh")
        m2.metric("Backtest error (MAPE)", f"{error:.1f}%")
        m3.metric("Method chosen", method)
        st.caption("Forecast uses the legacy cleaned series, which can estimate missing readings as described in Data Quality.")

with tabs[4]:
    mix_left, mix_right = st.columns(2)
    source_mix = measured.groupby(["building", "source"], as_index=False)["kwh_actual"].sum()
    mix_left.plotly_chart(px.bar(source_mix, x="building", y="kwh_actual", color="source",
                                 color_discrete_map=SOURCE_COLORS,
                                 title="Recorded energy by source and building"), width="stretch")
    carbon = measured.assign(estimated_co2=measured["kwh_actual"] * measured["source"].map(emission_factors))
    carbon = carbon.groupby("building", as_index=False)["estimated_co2"].sum()
    mix_right.plotly_chart(px.bar(carbon.sort_values("estimated_co2"), x="estimated_co2", y="building",
                                  orientation="h", title="Estimated CO2 by building (kg)"), width="stretch")
    source_stats = measured.assign(estimated_cost=measured["kwh_actual"] * measured["source"].map(tariff),
                                   estimated_co2=measured["kwh_actual"] * measured["source"].map(emission_factors))
    source_stats = source_stats.groupby("source", as_index=False).agg(
        recorded_kwh=("kwh_actual", "sum"), readings=("kwh_actual", "count"),
        estimated_cost=("estimated_cost", "sum"), estimated_co2=("estimated_co2", "sum"))
    st.dataframe(source_stats.round(1), hide_index=True)
    st.caption("Cost and CO2 are estimates; unknown-source measurements are omitted from those estimates.")

with tabs[5]:
    recommendations = []
    for building, group in d.groupby("building"):
        diesel = group[group["source"] == "Diesel Generator"]
        if len(diesel):
            energy = diesel["kwh"].sum()
            recommendations.append((0, f"{building}: {len(diesel)} diesel readings account for {energy:,.0f} kWh. Review backup-generator use."))
        anomalies = group[group["anomaly"]]
        if len(anomalies):
            recommendations.append((1, f"{building}: {len(anomalies)} unusually high reading(s); review operations on {', '.join(map(str, anomalies['date']))}."))
    peak_by_building = d.groupby("building")["peak_kw"].mean().sort_values(ascending=False)
    if len(peak_by_building):
        recommendations.append((2, f"{peak_by_building.index[0]} has the highest average peak demand ({peak_by_building.iloc[0]:.1f} kW). Consider staggering large-load startup."))
    if not recommendations:
        st.info("No findings can be derived from the selected data.")
    for level, message in recommendations:
        {0: st.error, 1: st.warning, 2: st.info}[level](message)
    st.caption("Findings are based on the selected readings; they are not guaranteed savings estimates.")

with tabs[6]:
    st.subheader("Data quality and provenance")
    st.dataframe(quality, hide_index=True)
    estimated_rows = df_all[df_all["kwh_fixed"]][["timestamp", "building", "kwh", "source"]]
    if len(estimated_rows):
        st.caption("The legacy forecast and anomaly views estimate missing or invalid kWh values. The dashboard, map, assistant, and PDF exclude those estimates.")
        st.dataframe(estimated_rows.rename(columns={"kwh": "estimated_kwh"}), hide_index=True)

with tabs[7]:
    st.subheader("Campus Energy Report")
    st.write("Generate a PDF using the current building and date filters. Recorded readings only.")
    report_pdf = build_report_pdf(d, quality, tariff, emission_factors)
    st.download_button("Generate Campus Energy Report", data=report_pdf,
                       file_name=f"wattwise-energy-report-{datetime.now():%Y-%m-%d}.pdf",
                       mime="application/pdf", type="primary", icon=":material/download:")
    st.caption("Configured tariffs and emission factors produce estimates; unknown sources are excluded from those estimates.")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { overflow-x: hidden; }
[data-testid="stMainBlockContainer"] { padding-left: clamp(1rem, 3vw, 2.5rem); padding-right: clamp(1rem, 3vw, 2.5rem); }
[data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.75rem; }
[data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width: min(100%, 190px); }
[data-testid="stDataFrame"] { max-width: 100%; overflow-x: auto; }
[data-testid="stPopover"] { position: fixed; right: 1.25rem; bottom: 1.25rem; z-index: 1000; }
[data-testid="stPopover"] [data-testid="stPopoverBody"] { width: min(380px, calc(100vw - 2rem)); max-height: min(70vh, 620px); overflow-y: auto; }
@media (max-width: 768px) {
  [data-testid="stMainBlockContainer"] { padding: 1rem 0.75rem 5rem; }
  [data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width: calc(50% - 0.5rem); flex: 1 1 calc(50% - 0.5rem); }
  [data-testid="stPlotlyChart"] { min-width: 0 !important; }
  [data-testid="stPopover"] { right: 0.75rem; bottom: 0.75rem; }
  [data-testid="stPopover"] [data-testid="stPopoverBody"] { width: calc(100vw - 1.5rem); max-height: 72vh; }
}
@media (max-width: 480px) {
  [data-testid="stHorizontalBlock"] > [data-testid="column"] { min-width: 100%; flex-basis: 100%; }
  [data-testid="stTabs"] button { padding: 0.4rem 0.55rem; font-size: 0.82rem; }
}
</style>
""", unsafe_allow_html=True)

with st.popover("🍃⚡", help="Open the Energy Assistant", key="energy_assistant"):
    st.subheader("Energy Assistant")
    st.caption("Answers use valid recorded readings in the current filters.")
    chat_history = st.session_state.setdefault("energy_chat_history", [])
    for chat_item in chat_history:
        with st.chat_message(chat_item["role"]):
            st.write(chat_item["content"])
    with st.form("energy_assistant_form", clear_on_submit=True):
        question = st.text_input("Ask about campus energy", placeholder="Which building has the highest use?")
        sent = st.form_submit_button("Send", type="primary", width="stretch")
    if sent and question.strip():
        chat_history.append({"role": "user", "content": question.strip()})
        chat_history.append({"role": "assistant", "content": answer_energy_question(question.strip(), d)})
        st.rerun()
    st.caption("Suggested questions")
    suggested_questions = ["What is the campus energy trend?", "Which building uses the most energy?",
                           "What is the highest recorded peak?", "What is the energy source mix?"]
    suggestion_columns = st.columns(2)
    for index, suggested_question in enumerate(suggested_questions):
        if suggestion_columns[index % 2].button(suggested_question, key=f"energy_suggestion_{index}", width="stretch"):
            chat_history.append({"role": "user", "content": suggested_question})
            chat_history.append({"role": "assistant", "content": answer_energy_question(suggested_question, d)})
            st.rerun()
