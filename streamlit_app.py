"""
Streamlit dashboard for Berlin bike-sharing demand forecasting.

Run from the project root:
    streamlit run streamlit_app.py
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Berlin Bike Demand",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
FEATURES_PATH = ROOT / "data" / "features" / "features.parquet"
MODEL_PATH    = ROOT / "models" / "best_model.txt"
PREDICTIONS_PATH = ROOT / "data" / "predictions" / "predictions_latest.parquet"

SPLIT_DATE           = pd.Timestamp("2026-01-01")
LOW_DEMAND_DISTRICTS = ["Marzahn-Hellersdorf", "Spandau", "Reinickendorf"]
TARGET = "relative_demand_tomorrow"
FEATURE_COLS = [
    "district",
    "dow", "month", "is_weekend", "is_holiday", "is_pre_holiday", "is_post_holiday",
    "daylight_hours",
    "lag_1d", "lag_2d", "lag_7d", "lag_14d",
    "roll_3d_mean", "roll_3d_std",
    "roll_7d_mean", "roll_7d_std",
    "roll_14d_mean", "roll_14d_std",
    "active_stations",
    "temperature_2m", "apparent_temperature", "precipitation",
    "rain", "snowfall", "wind_speed_10m", "cloud_cover", "relative_humidity_2m",
    "temp_change_1d", "apparent_temperature_tomorrow", "precipitation_tomorrow",
    "apparent_temp_x_weekend",
]

PALETTE = {
    "actual":    "#4C78A8",
    "predicted": "#E45756",
    "baseline":  "#F58518",
    "forecast":  "#54A24B",
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
    path = ROOT / "data" / "stations.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)

@st.cache_data(ttl=300)
def load_predictions():
    if not PREDICTIONS_PATH.exists():
        return None
    df = pd.read_parquet(PREDICTIONS_PATH)
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])
    df["features_date"]   = pd.to_datetime(df["features_date"])
    return df

@st.cache_resource
def load_model() -> lgb.Booster:
    return lgb.Booster(model_file=str(MODEL_PATH))


geojson     = load_geojson()
stations    = load_stations()
df          = load_data()
model       = load_model()
predictions = load_predictions()

df["pred_rel"] = model.predict(df[FEATURE_COLS])
df["pred_abs"] = df["pred_rel"] * df["active_stations"]

test_df  = df[df["date"] >= SPLIT_DATE]
train_df = df[df["date"] < SPLIT_DATE]

districts = sorted(df["district"].unique())

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Berlin Bike-Sharing")
st.sidebar.markdown("Next-day demand forecast by district")
st.sidebar.divider()

selected_district = st.sidebar.selectbox("Select district", districts)
show_train        = st.sidebar.checkbox("Include training period", value=False)

st.sidebar.divider()
st.sidebar.caption(
    "Model: LightGBM + Optuna  \n"
    "Target: relative demand (rentals / active stations)  \n"
    "Train: Jan–Dec 2025  |  Test: Jan–Apr 2026"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Berlin Bike-Sharing — Demand Forecast")


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Live Forecast
# ══════════════════════════════════════════════════════════════════════════════
if predictions is not None:
    pred_date    = predictions["prediction_date"].iloc[0].date()
    features_date = predictions["features_date"].iloc[0].date()
    app_temp     = predictions["apparent_temperature_tomorrow"].iloc[0]
    precip       = predictions["precipitation_tomorrow"].iloc[0]

    st.subheader(f"Tomorrow's forecast — {pred_date}")

    # ── Weather context ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Apparent temperature", f"{app_temp:.1f} °C")
    c2.metric("Precipitation", f"{precip:.1f} mm")
    c3.metric("Feature data as of", str(features_date))

    # ── Bar chart + map ───────────────────────────────────────────────────────
    col_bar, col_map = st.columns([1, 1.4])

    with col_bar:
        pred_sorted = predictions.sort_values("pred_rentals", ascending=True)
        fig_bar = go.Figure(go.Bar(
            x=pred_sorted["pred_rentals"],
            y=pred_sorted["district"],
            orientation="h",
            marker_color=PALETTE["forecast"],
            text=pred_sorted["pred_rentals"].round(0).astype(int),
            textposition="outside",
        ))
        fig_bar.update_layout(
            template="simple_white", height=340,
            xaxis_title="Predicted rentals",
            yaxis_title="",
            margin=dict(t=10, b=20, l=10, r=60),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_map:
        fig_forecast_map = go.Figure(go.Choroplethmapbox(
            geojson=geojson,
            locations=predictions["district"],
            featureidkey="properties.name",
            z=predictions["pred_rentals"],
            colorscale="Greens",
            zmin=0,
            zmax=predictions["pred_rentals"].max() * 1.1,
            colorbar_title="Pred. rentals",
            marker_line_color="white",
            marker_line_width=1,
            hovertemplate=(
                "<b>%{location}</b><br>"
                "Predicted rentals: %{z:.0f}<extra></extra>"
            ),
        ))
        fig_forecast_map.update_layout(
            height=340,
            margin=dict(t=0, b=0, l=0, r=0),
            mapbox_style="open-street-map",
            mapbox_zoom=8.5,
            mapbox_center={"lat": 52.52, "lon": 13.41},
        )
        st.plotly_chart(fig_forecast_map, use_container_width=True)

    st.dataframe(
        predictions[["district", "pred_rentals", "pred_relative_demand", "active_stations"]]
        .rename(columns={
            "district":            "District",
            "pred_rentals":        "Predicted rentals",
            "pred_relative_demand":"Relative demand",
            "active_stations":     "Active stations",
        })
        .set_index("District")
        .style.format({"Predicted rentals": "{:.0f}", "Relative demand": "{:.2f}", "Active stations": "{:.0f}"}),
        use_container_width=True,
    )
else:
    st.info(
        "No live forecast found. Run `python3 -m src.data.collection.fetch_live` "
        "then `python3 -m src.models.predict` to generate tomorrow's predictions.",
        icon="ℹ️",
    )

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Model performance — test set (Jan–Apr 2026)")

rmse_abs      = np.sqrt(mean_squared_error(test_df["rentals_tomorrow"], test_df["pred_abs"]))
mae_abs       = mean_absolute_error(test_df["rentals_tomorrow"], test_df["pred_abs"])
r2            = r2_score(test_df[TARGET], test_df["pred_rel"])
baseline_rmse = np.sqrt(mean_squared_error(
    test_df["rentals_tomorrow"],
    test_df["lag_7d"] * test_df["active_stations"],
))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test RMSE", f"{rmse_abs:.0f} rentals",
          delta=f"{rmse_abs - baseline_rmse:.0f} vs lag-7d baseline",
          delta_color="inverse")
c2.metric("Test MAE",  f"{mae_abs:.0f} rentals")
c3.metric("R² (relative demand)", f"{r2:.3f}")
c4.metric("Districts modelled", f"{df['district'].nunique()} / 12")

# ── Per-district comparison ───────────────────────────────────────────────────
rows = []
for district, grp in test_df.groupby("district", observed=True):
    p = grp["pred_abs"]
    a = grp["rentals_tomorrow"]
    b = grp["lag_7d"] * grp["active_stations"]
    rows.append({
        "District":      str(district),
        "RMSE":          np.sqrt(mean_squared_error(a, p)),
        "MAE":           mean_absolute_error(a, p),
        "R²":            r2_score(grp[TARGET], grp["pred_rel"]),
        "Baseline RMSE": np.sqrt(mean_squared_error(a, b)),
        "vs baseline":   np.sqrt(mean_squared_error(a, p)) - np.sqrt(mean_squared_error(a, b)),
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
    template="simple_white", height=300,
    yaxis_title="RMSE (absolute rentals)",
    xaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=20),
)
st.plotly_chart(fig2, use_container_width=True)

st.dataframe(
    metrics_df.style
    .format({"RMSE": "{:.1f}", "MAE": "{:.1f}", "R²": "{:.3f}",
             "Baseline RMSE": "{:.1f}", "vs baseline": "{:+.1f}"})
    .background_gradient(subset=["R²"], cmap="RdBu", vmin=0, vmax=1),
    use_container_width=True,
    hide_index=True,
)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — District Detail
# ══════════════════════════════════════════════════════════════════════════════
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
    template="simple_white", height=360,
    yaxis_title="Rentals",
    xaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=20),
    title=dict(text=f"RMSE {d_rmse:.0f}  |  MAE {d_mae:.0f}", font=dict(size=13), x=1, xanchor="right"),
)
st.plotly_chart(fig, use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — District Map
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("District map — model performance")

col_map, col_ctrl = st.columns([3, 1])

with col_ctrl:
    color_col = st.radio(
        "Colour districts by",
        ["R²", "RMSE", "MAE", "vs baseline"],
    )
    show_stations_cb = st.checkbox("Show bike stations", value=stations is not None)

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

    if show_stations_cb and stations is not None:
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
        height=500,
        margin=dict(t=0, b=0, l=0, r=0),
        mapbox_style="open-street-map",
        mapbox_zoom=9,
        mapbox_center={"lat": 52.52, "lon": 13.41},
    )
    st.plotly_chart(fig3, use_container_width=True)
