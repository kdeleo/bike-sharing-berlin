"""
Streamlit dashboard for Berlin bike-sharing demand forecasting.

Run from the project root:
    streamlit run streamlit_app.py
"""

import lightgbm as lgb
import numpy as np
import json
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Berlin Bike Demand",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
FEATURES_PATH = ROOT / "data" / "features" / "features.parquet"
MODEL_PATH = ROOT / "models" / "best_model.txt"

SPLIT_DATE = pd.Timestamp("2026-01-01")
LOW_DEMAND_DISTRICTS = ["Marzahn-Hellersdorf", "Spandau", "Reinickendorf"]
TARGET = "relative_demand_tomorrow"
FEATURE_COLS = [
    "district",
    "dow", "month", "is_weekend", "is_holiday",
    "daylight_hours",
    "lag_1d", "lag_2d", "lag_7d", "lag_14d",
    "roll_3d_mean", "roll_3d_std",
    "roll_7d_mean", "roll_7d_std",
    "roll_14d_mean", "roll_14d_std",
    "active_stations",
    "temperature_2m", "apparent_temperature", "precipitation",
    "rain", "snowfall", "wind_speed_10m", "cloud_cover", "relative_humidity_2m",
    "temp_change_1d", "apparent_temperature_tomorrow", "precipitation_tomorrow",
    "apparent_temp_x_weekend"
]

PALETTE = {
    "actual":   "#4C78A8",
    "predicted":"#E45756",
    "baseline": "#F58518",
}

# ── Data & model (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=FEATURE_COLS + [TARGET]).copy()
    df = df[~df["district"].isin(LOW_DEMAND_DISTRICTS)].copy()
    df["district"] = df["district"].cat.remove_unused_categories()
    return df

@st.cache_data
def load_geojson():
    with open(ROOT / "configs" / "berlin_bezirke.geojson") as f:
        return json.load(f)

@st.cache_data
def load_stations():
    return pd.read_parquet(ROOT / "data" / "stations.parquet")

geojson  = load_geojson()
stations = load_stations()

@st.cache_resource
def load_model() -> lgb.Booster:
    return lgb.Booster(model_file=str(MODEL_PATH))


df = load_data()
model = load_model()

df["pred_rel"] = model.predict(df[FEATURE_COLS])
df["pred_abs"] = df["pred_rel"] * df["active_stations"]

test_df = df[df["date"] >= SPLIT_DATE]
train_df = df[df["date"] < SPLIT_DATE]

districts = sorted(df["district"].unique())


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Berlin Bike-Sharing")
st.sidebar.markdown("Next-day demand forecast by district")
st.sidebar.divider()

selected_district = st.sidebar.selectbox("Select district", districts)
show_train = st.sidebar.checkbox("Include training period", value=False)

st.sidebar.divider()
st.sidebar.caption(
    "Model: LightGBM + Optuna  \n"
    "Target: relative demand (rentals / active stations)  \n"
    "Train: Jan–Dec 2025  |  Test: Jan–Apr 2026"
)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Berlin Bike-Sharing — Demand Forecast")

# ── Top metrics ───────────────────────────────────────────────────────────────
rmse_abs = np.sqrt(mean_squared_error(test_df["rentals_tomorrow"], test_df["pred_abs"]))
mae_abs  = mean_absolute_error(test_df["rentals_tomorrow"], test_df["pred_abs"])
r2       = r2_score(test_df[TARGET], test_df["pred_rel"])
baseline_rmse = np.sqrt(
    mean_squared_error(
        test_df["rentals_tomorrow"],
        test_df["lag_7d"] * test_df["active_stations"],
    )
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test RMSE", f"{rmse_abs:.0f} rentals",
          delta=f"{rmse_abs - baseline_rmse:.0f} vs lag-7d baseline",
          delta_color="inverse")
c2.metric("Test MAE", f"{mae_abs:.0f} rentals")
c3.metric("R² (relative demand)", f"{r2:.3f}")
c4.metric("Districts modelled", f"{df['district'].nunique()} / 12")

st.divider()

# ── Time series ───────────────────────────────────────────────────────────────
st.subheader(f"{selected_district} — actual vs predicted rentals")

plot_df = df[df["district"] == selected_district].sort_values("date")
if not show_train:
    plot_df = plot_df[plot_df["date"] >= SPLIT_DATE]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=plot_df["date"], y=plot_df["rentals_tomorrow"],
    name="actual", line=dict(color=PALETTE["actual"], width=1.5), opacity=0.8,
))
fig.add_trace(go.Scatter(
    x=plot_df["date"], y=plot_df["pred_abs"],
    name="predicted", line=dict(color=PALETTE["predicted"], width=2),
))
if show_train:
    fig.add_shape(
        type="line",
        x0=SPLIT_DATE, x1=SPLIT_DATE,
        y0=0, y1=1, yref="paper",
        line=dict(dash="dash", color="gray", width=1.5),
    )
    fig.add_annotation(
        x=SPLIT_DATE, y=1, yref="paper",
        text="train / test split",
        showarrow=False,
        xanchor="left", yanchor="top",
        font=dict(size=11, color="gray"),
    )

d_rmse = np.sqrt(mean_squared_error(plot_df["rentals_tomorrow"], plot_df["pred_abs"]))
d_mae  = mean_absolute_error(plot_df["rentals_tomorrow"], plot_df["pred_abs"])
fig.update_layout(
    template="simple_white", height=380,
    yaxis_title="Rentals",
    xaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=20),
    title=dict(text=f"RMSE {d_rmse:.0f}  |  MAE {d_mae:.0f}", font=dict(size=13), x=1, xanchor="right"),
)
st.plotly_chart(fig, width='stretch')

st.divider()

# ── Per-district comparison ───────────────────────────────────────────────────
st.subheader("Per-district performance — test set")

rows = []
for district, grp in test_df.groupby("district", observed=True):
    p = grp["pred_abs"]
    a = grp["rentals_tomorrow"]
    b = grp["lag_7d"] * grp["active_stations"]
    rows.append({
        "District": str(district),
        "RMSE": np.sqrt(mean_squared_error(a, p)),
        "MAE": mean_absolute_error(a, p),
        "R²": r2_score(grp[TARGET], grp["pred_rel"]),
        "Baseline RMSE": np.sqrt(mean_squared_error(a, b)),
        "vs baseline": np.sqrt(mean_squared_error(a, p)) - np.sqrt(mean_squared_error(a, b)),
    })

metrics_df = pd.DataFrame(rows).sort_values("RMSE", ascending=False)

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=metrics_df["District"],
    y=metrics_df["RMSE"],
    name="Model RMSE",
    marker_color=PALETTE["actual"],
))
fig2.add_trace(go.Scatter(
    x=metrics_df["District"],
    y=metrics_df["Baseline RMSE"],
    name="Baseline RMSE (lag 7d)",
    mode="markers",
    marker=dict(symbol="line-ew", size=14, color=PALETTE["predicted"],
                line=dict(width=2.5, color=PALETTE["predicted"])),
))
fig2.update_layout(
    template="simple_white", height=320,
    yaxis_title="RMSE (absolute rentals)",
    xaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=20),
)
st.plotly_chart(fig2, width='stretch')

st.dataframe(
    metrics_df.style
    .format({"RMSE": "{:.1f}", "MAE": "{:.1f}", "R²": "{:.3f}",
             "Baseline RMSE": "{:.1f}", "vs baseline": "{:+.1f}"})
    .background_gradient(subset=["R²"], cmap="RdBu", vmin=0, vmax=1),
    width='stretch',
    hide_index=True,
)

st.divider()

# ── Berlin districts map ───────────────────────────────────────────────────
st.subheader("Bike stations & district performance")

col_map, col_ctrl = st.columns([3, 1])

with col_ctrl:
    color_col = st.radio(
        "Colour districts by",
        ["R²", "RMSE", "MAE", "vs baseline"],
)
    show_stations = st.checkbox("Show bike stations", value=True)

with col_map:
    choropleth = go.Choroplethmapbox(
        geojson=geojson,
        locations=metrics_df["District"],
        featureidkey="properties.name",
        z=metrics_df[color_col],
        colorscale="RdBu_r",
        reversescale=color_col != "R²",
        zmin=metrics_df[color_col].min(),
        zmax=metrics_df[color_col].max(),
        colorbar_title=color_col,
        marker_line_color="white",
        marker_line_width=1,
        hovertemplate="<b>%{location}</b><br>" + f"{color_col}: " + "%{z:.2f}<extra></extra>",
    )

    traces = [choropleth]

    if show_stations:
        traces.append(go.Scattermapbox(
            lat=stations["latitude"],
            lon=stations["longitude"],
            mode="markers",
            marker=dict(size=4, color="black", opacity=0.4),
            text=stations["name"],
            hovertemplate="<b>%{text}</b><extra></extra>",
            name="Stations",
        ))

    fig3 = go.Figure(traces)
    fig3.update_layout(
        height=520,
        margin=dict(t=0, b=0, l=0, r=0),
        mapbox_style="open-street-map",
        mapbox_zoom=9,
        mapbox_center={"lat": 52.52, "lon": 13.41},
    )
    st.plotly_chart(fig3, width='stretch')
