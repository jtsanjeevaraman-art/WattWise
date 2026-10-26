import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Campus Energy Insights", layout="wide")

DATA_FILE = "sample_raw_energy.csv"   # keep this file next to app.py
BAD_KWH = 50000                       # readings above this are sensor errors (e.g. 99999)
SOURCE_COLORS = {"Grid": "#4C78A8", "Solar": "#F2B134", "Diesel Generator": "#B23A48"}
M_TREND = "Linear trend"
M_AVG = "Recent average (last 8 readings)"


# ---------- 1. LOAD + CLEAN ----------
@st.cache_data
def clean(raw):
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for c, default in {"peak_kw": np.nan, "voltage": np.nan,
                       "source": "Grid", "occupancy": np.nan}.items():
        if c not in df.columns:
            df[c] = default

    log = [("Rows in file", len(df))]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["units_kwh", "peak_kw", "voltage", "occupancy"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["building"] = df["building"].astype(str).str.strip()
    df = df.dropna(subset=["timestamp"])

    n = len(df)
    df = df.drop_duplicates(subset=["timestamp", "building"], keep="first")
    log.append(("Duplicate rows removed", n - len(df)))

    before = df["source"].nunique()
    df["source"] = df["source"].astype(str).str.strip().str.title()
    log.append(("Source labels unified (before to after)", f"{before} to {df['source'].nunique()}"))

    df = df.sort_values(["building", "timestamp"]).reset_index(drop=True)

    bad = (df["units_kwh"] > BAD_KWH) | (df["units_kwh"] <= 0)
    log.append(("Impossible kWh values removed (e.g. 99999)", int(bad.sum())))
    df.loc[bad, "units_kwh"] = np.nan
    missing = df["units_kwh"].isna()
    log.append(("Missing kWh values filled (total, incl. the above)", int(missing.sum())))
    df["kwh"] = df.groupby("building")["units_kwh"].transform(
        lambda s: s.interpolate(limit_direction="both"))
    df["kwh"] = df["kwh"].fillna(df.groupby("building")["units_kwh"].transform("median"))
    df["kwh_fixed"] = missing

    log.append(("Missing peak_kw filled with building median", int(df["peak_kw"].isna().sum())))
    df["peak_kw"] = df["peak_kw"].fillna(df.groupby("building")["peak_kw"].transform("median"))

    log.append(("Negative voltage readings corrected", int((df["voltage"] < 0).sum())))
    df["voltage"] = df["voltage"].abs()

    df = df.dropna(subset=["kwh"])
    log.append(("Clean rows used", len(df)))
    df["date"] = df["timestamp"].dt.date
    df["month"] = df["timestamp"].dt.strftime("%Y-%m")
    return df, pd.DataFrame([(a, str(b)) for a, b in log], columns=["Check", "Count"])


st.sidebar.header("Data")
up = st.sidebar.file_uploader("Upload a CSV (optional)", type="csv")
if up is not None:
    raw = pd.read_csv(up)
elif os.path.exists(DATA_FILE):
    raw = pd.read_csv(DATA_FILE)
    st.sidebar.success(f"Using {DATA_FILE}")
else:
    st.error(f"Put {DATA_FILE} in the same folder as app.py, or upload a CSV in the sidebar.")
    st.stop()

if not {"timestamp", "building", "units_kwh"}.issubset({c.strip().lower() for c in raw.columns}):
    st.error("The CSV needs at least these columns: timestamp, building, units_kwh")
    st.stop()

df_all, quality = clean(raw)

# ---------- 2. RATES (editable) ----------
st.sidebar.header("Rates")
tariff = {
    "Grid": st.sidebar.number_input("Grid tariff (Rs per kWh)", 0.0, 100.0, 8.0, 0.5),
    "Diesel Generator": st.sidebar.number_input("Diesel generator cost (Rs per kWh)", 0.0, 200.0, 25.0, 1.0),
    "Solar": st.sidebar.number_input("Solar running cost (Rs per kWh)", 0.0, 100.0, 0.0, 0.5),
}
with st.sidebar.expander("CO2 factors (kg per kWh)"):
    emis = {
        "Grid": st.number_input("Grid", 0.0, 3.0, 0.71, 0.01),
        "Diesel Generator": st.number_input("Diesel generator", 0.0, 3.0, 0.90, 0.01),
        "Solar": st.number_input("Solar", 0.0, 3.0, 0.0, 0.01),
    }

# ---------- 3. ANOMALY SCORE (robust z-score within each building) ----------
def robust_z(s):
    med = s.median()
    mad = (s - med).abs().median()
    scale = 1.4826 * mad if mad > 0 else s.std()
    return (s - med) / scale if scale and scale > 0 else s * 0

st.sidebar.header("Filters")
thr = st.sidebar.slider("Anomaly sensitivity (lower = more alerts)", 2.0, 5.0, 2.5, 0.5)

df_all["typical"] = df_all.groupby("building")["kwh"].transform("median")
df_all["z"] = df_all.groupby("building")["kwh"].transform(robust_z)
df_all["anomaly"] = (df_all["z"] > thr) & (~df_all["kwh_fixed"])
df_all["cost"] = df_all["kwh"] * df_all["source"].map(tariff).fillna(tariff["Grid"])
df_all["co2"] = df_all["kwh"] * df_all["source"].map(emis).fillna(emis["Grid"])

buildings = sorted(df_all["building"].unique())
sel = st.sidebar.multiselect("Buildings", buildings, default=buildings)
d0, d1 = df_all["date"].min(), df_all["date"].max()
picked = st.sidebar.date_input("Date range", (d0, d1), min_value=d0, max_value=d1)

base = df_all[df_all["building"].isin(sel)]
d = base
if isinstance(picked, (tuple, list)) and len(picked) == 2:
    d = base[(base["date"] >= picked[0]) & (base["date"] <= picked[1])]
if d.empty:
    st.warning("No data for this selection. Pick at least one building and a valid date range.")
    st.stop()

# ---------- 4. HEADER + KPIs ----------
st.title("⚡ Campus Energy Insights")
st.caption(f"{d['building'].nunique()} buildings, {len(d)} readings, "
           f"{d['date'].min()} to {d['date'].max()}. Figures cover the readings in the data file.")
total = d["kwh"].sum()
solar_share = d.loc[d.source == "Solar", "kwh"].sum() / total
diesel_share = d.loc[d.source == "Diesel Generator", "kwh"].sum() / total
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total energy", f"{total:,.0f} kWh")
k2.metric("Estimated cost", f"₹{d['cost'].sum():,.0f}")
k3.metric("CO₂ emitted", f"{d['co2'].sum() / 1000:,.2f} t")
k4.metric("Solar share", f"{solar_share:.0%}")
k5.metric("Anomalies", int(d["anomaly"].sum()))

tabs = st.tabs(["Overview", "Anomalies", "Forecast", "Energy mix & carbon",
                "Recommendations", "Data quality"])

# ---------- OVERVIEW ----------
with tabs[0]:
    fig = px.line(d, x="timestamp", y="kwh", color="building", markers=True,
                  title="Energy per reading over time (kWh)")
    st.plotly_chart(fig)
    left, right = st.columns(2)
    by_b = d.groupby("building", as_index=False).agg(kwh=("kwh", "sum"), cost=("cost", "sum"))
    left.plotly_chart(px.bar(by_b.sort_values("kwh"), x="kwh", y="building", orientation="h",
                             color="building", title="Total energy by building (kWh)"))
    heat = d.pivot_table(index="building", columns="month", values="kwh", aggfunc="mean")
    right.plotly_chart(px.imshow(heat, aspect="auto", text_auto=".0f",
                                 color_continuous_scale="YlOrRd",
                                 labels=dict(x="Month", y="Building", color="Avg kWh"),
                                 title="Average kWh by building and month"))
    eff = d.groupby("building").agg(kwh=("kwh", "sum"), occ=("occupancy", "sum"))
    eff = eff[eff["occ"] > 0]
    if len(eff):
        eff["kwh_per_person"] = eff["kwh"] / eff["occ"]
        st.plotly_chart(px.bar(eff.reset_index().sort_values("kwh_per_person"),
                               x="building", y="kwh_per_person", color="building",
                               title="Energy per occupant (kWh per person): higher means less efficient"))

# ---------- ANOMALIES ----------
with tabs[1]:
    b = st.selectbox("Building", sel, key="anom_b")
    g = d[d["building"] == b].sort_values("timestamp")
    fig = go.Figure()
    fig.add_scatter(x=g["timestamp"], y=g["kwh"], mode="lines+markers", name="Usage")
    fig.add_hline(y=g["typical"].iloc[0], line_dash="dot",
                  annotation_text="typical (median)")
    bad = g[g["anomaly"]]
    fig.add_scatter(x=bad["timestamp"], y=bad["kwh"], mode="markers", name="Anomaly",
                    marker=dict(color="red", size=13, symbol="x"))
    fig.update_layout(title=f"{b}: usage with anomalies marked", yaxis_title="kWh")
    st.plotly_chart(fig)

    out = d[d["anomaly"]].copy()
    if out.empty:
        st.success("No anomalies at this sensitivity. Lower the slider in the sidebar to see more.")
    else:
        out["extra_kwh"] = (out["kwh"] - out["typical"]).round(1)
        out["extra_cost_rs"] = (out["extra_kwh"] * out["cost"] / out["kwh"]).round(0)
        out["why flagged"] = out.apply(
            lambda r: f"{r.kwh:.0f} kWh vs typical {r.typical:.0f} kWh "
                      f"(+{(r.kwh / r.typical - 1) * 100:.0f}%)", axis=1)
        st.subheader("Flagged readings")
        st.dataframe(out.sort_values("extra_kwh", ascending=False)
                        [["timestamp", "building", "source", "why flagged",
                          "extra_kwh", "extra_cost_rs"]], hide_index=True)
    st.caption("Method: robust z-score. Each reading is compared with its own building's median, "
               "so a big building is not flagged just for being big. Imputed values are never flagged.")

# ---------- FORECAST ----------
def predict(method, train, h):
    if method == M_TREND:
        x = np.arange(len(train))
        slope, icpt = np.polyfit(x, train, 1)
        return slope * np.arange(len(train), len(train) + h) + icpt
    return np.repeat(train[-8:].mean(), h)

def forecast_building(g, h, test_n=4):
    g = g.sort_values("timestamp")
    y = g["kwh"].to_numpy(float)
    if len(y) < 12:
        return None
    ts = g["timestamp"]
    step = ts.diff().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        step = pd.Timedelta(days=7)
    future = pd.DatetimeIndex([ts.iloc[-1] + step * (i + 1) for i in range(h)])
    scores = {}
    for m in (M_TREND, M_AVG):
        p = predict(m, y[:-test_n], test_n)
        scores[m] = float(np.mean(np.abs((y[-test_n:] - p) / y[-test_n:])) * 100)
    best = min(scores, key=scores.get)
    return ts, y, future, predict(best, y, h), best, scores[best]

with tabs[2]:
    h = st.slider("Readings to forecast ahead (weeks for weekly data)", 4, 12, 8)
    fb = st.selectbox("Building", sel, key="fore_b")
    res = forecast_building(base[base["building"] == fb], h)
    if res is None:
        st.info("Need at least 12 readings in this building to forecast.")
    else:
        ts, y, future, pred, method, mape = res
        fig = go.Figure()
        fig.add_scatter(x=ts, y=y, mode="lines+markers", name="Actual")
        fig.add_scatter(x=future, y=pred, mode="lines+markers", name="Forecast",
                        line=dict(dash="dash"))
        fig.update_layout(title=f"{fb}: demand forecast", yaxis_title="kWh per reading")
        st.plotly_chart(fig)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Forecast, next {h}", f"{pred.sum():,.0f} kWh")
        c2.metric("Backtest error (MAPE)", f"{mape:.1f}%")
        c3.metric("Method chosen", method)
        st.caption("Two simple methods are tested on the last 4 known readings. "
                   "The one with the lower error is used for the forecast.")

    rows = []
    for b2 in sel:
        g2 = base[base["building"] == b2]
        r2 = forecast_building(g2, h)
        if r2:
            rate = g2["cost"].sum() / g2["kwh"].sum()
            rows.append({"building": b2, f"forecast next {h} (kWh)": round(r2[3].sum()),
                         "forecast cost (Rs)": round(r2[3].sum() * rate),
                         "method": r2[4], "error %": round(r2[5], 1)})
    if rows:
        st.subheader("Campus forecast")
        tbl = pd.DataFrame(rows)
        st.dataframe(tbl, hide_index=True)
        st.metric(f"Whole campus, next {h}", f"{tbl.iloc[:, 1].sum():,.0f} kWh")

# ---------- ENERGY MIX & CARBON ----------
with tabs[3]:
    left, right = st.columns(2)
    mix = d.groupby(["building", "source"], as_index=False)["kwh"].sum()
    left.plotly_chart(px.bar(mix, x="building", y="kwh", color="source",
                             color_discrete_map=SOURCE_COLORS,
                             title="Energy source by building (kWh)"))
    co = d.groupby("building", as_index=False)["co2"].sum()
    right.plotly_chart(px.bar(co.sort_values("co2"), x="co2", y="building", orientation="h",
                              title="CO₂ by building (kg)"))
    src = d.groupby("source", as_index=False).agg(kwh=("kwh", "sum"), cost=("cost", "sum"),
                                                  co2=("co2", "sum"))
    src["Rs per kWh"] = (src["cost"] / src["kwh"]).round(1)
    st.dataframe(src.round(0), hide_index=True)

# ---------- RECOMMENDATIONS ----------
with tabs[4]:
    recs = []
    weak = []
    for b, g in d.groupby("building"):
        diesel = g[g["source"] == "Diesel Generator"]
        if len(diesel):
            k = diesel["kwh"].sum()
            recs.append(dict(level=0, cost=k * (tariff["Diesel Generator"] - tariff["Grid"]),
                             co2=k * (emis["Diesel Generator"] - emis["Grid"]), kwh=0,
                             text=f"**{b}**: {len(diesel)} of {len(g)} readings ran on diesel "
                                  f"({k:,.0f} kWh, {k / g['kwh'].sum():.0%} of its energy). "
                                  f"Move this load to grid or solar and keep diesel for outages only."))
        a = g[g["anomaly"]]
        if len(a):
            extra = (a["kwh"] - a["typical"])
            recs.append(dict(level=1, kwh=extra.sum(),
                             cost=(extra * a["cost"] / a["kwh"]).sum(),
                             co2=(extra * a["co2"] / a["kwh"]).sum(),
                             text=f"**{b}**: {len(a)} abnormal spike(s), about {extra.sum():,.0f} kWh above "
                                  f"normal. Check what was running on {', '.join(map(str, a['date']))}."))
        gg = g.dropna(subset=["occupancy"])
        if len(gg) >= 8 and gg["occupancy"].std() > 0:
            r = gg["kwh"].corr(gg["occupancy"])
            if r < 0.3:
                weak.append(f"{b} (r = {r:.2f})")
    if weak:
        recs.append(dict(level=2, kwh=0, cost=0, co2=0,
                         text=f"Energy use does not follow occupancy in {len(weak)} of "
                              f"{d['building'].nunique()} buildings: {', '.join(weak)}. "
                              f"Part of the load probably runs whether or not people are present. "
                              f"Add occupancy-based lighting and AC control and audit after-hours loads."))
    pk = d.groupby("building")["peak_kw"].mean().sort_values(ascending=False)
    recs.append(dict(level=2, kwh=0, cost=0, co2=0,
                     text=f"**{pk.index[0]}** has the highest average peak demand ({pk.iloc[0]:.0f} kW). "
                          f"Stagger start-up of heavy loads (AC, pumps) and consider solar plus battery "
                          f"for peak shaving."))

    s1, s2, s3 = st.columns(3)
    s1.metric("Possible saving", f"₹{sum(r['cost'] for r in recs):,.0f}")
    s2.metric("Wasted energy found", f"{sum(r['kwh'] for r in recs):,.0f} kWh")
    s3.metric("CO₂ avoidable", f"{sum(r['co2'] for r in recs):,.0f} kg")
    st.caption("Savings cover the readings in the selected date range. "
               "Diesel saving assumes the load moves to grid power.")
    for r in sorted(recs, key=lambda r: r["level"]):
        msg = r["text"]
        if r["cost"] > 0:
            msg += f"  \n**Saving: ₹{r['cost']:,.0f}, {r['co2']:,.0f} kg CO₂**"
        {0: st.error, 1: st.warning, 2: st.info}[r["level"]](msg)

# ---------- DATA QUALITY ----------
with tabs[5]:
    st.subheader("What was fixed before analysis")
    st.dataframe(quality, hide_index=True)
    fixed = df_all[df_all["kwh_fixed"]][["timestamp", "building", "kwh", "source"]]
    if len(fixed):
        st.write("Readings where kWh was missing or invalid and was estimated from neighbouring weeks:")
        st.dataframe(fixed.rename(columns={"kwh": "estimated_kwh"}), hide_index=True)
    st.caption("Cleaning rules: duplicates dropped, source names unified, kWh above 50,000 or "
               "zero treated as sensor errors, missing values interpolated per building, "
               "negative voltage made positive.")
               
               