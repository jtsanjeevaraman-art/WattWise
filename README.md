# WattWise Campus Energy

A Streamlit dashboard for exploring campus energy readings from `sample_raw_energy.csv`.

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run wattwise_app.py
```

## Deploy

Use `wattwise_app.py` as the Streamlit Community Cloud app file. The app reads the included sample CSV by default and also supports CSV uploads.

## Data

The sample CSV includes timestamps, buildings, energy readings in kWh, peak demand in kW, energy source, voltage, and occupancy. Dashboard and report totals use valid recorded energy readings. Estimated values are identified separately; unknown-source readings are excluded from cost and emissions estimates.
