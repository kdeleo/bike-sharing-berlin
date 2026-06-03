"""
Streamlit dashboard for Berlin bike-sharing demand forecasting.

Run from the project root:
    streamlit run streamlit_app.py
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
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
    "lag_1d", "lag_2d", "lag_7d", "lag_14d",
    "roll_3d_mean", "roll_3d_std",
    "roll_7d_mean", "roll_7d_std",
    "roll_14d_mean", "roll_14d_std",
    "active_stations",
    "temperature_2m", "apparent_temperature", "precipitation",
    "rain", "snowfall", "wind_speed_10m", "cloud_cover", "relative_humidity_2m",
]


# ── Data & model (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=FEATURE_COLS + [TARGET]).copy()
    df = df[~df["district"].isin(LOW_DEMAND_DISTRICTS)].copy()
    df["district"] = df["district"].cat.remove_unused_categories()
    return df


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
    name="actual", line=dict(color="#4C78A8", width=1.5), opacity=0.8,
))
fig.add_trace(go.Scatter(
    x=plot_df["date"], y=plot_df["pred_abs"],
    name="predicted", line=dict(color="#E45756", width=2),
))
if show_train:
    fig.add_vline(
        x=SPLIT_DATE.timestamp() * 1000,
        line_dash="dash", line_color="gray",
        annotation_text="train / test", annotation_position="top right",
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
    marker_color="#4C78A8",
))
fig2.add_trace(go.Scatter(
    x=metrics_df["District"],
    y=metrics_df["Baseline RMSE"],
    name="Baseline RMSE (lag 7d)",
    mode="markers",
    marker=dict(symbol="line-ew", size=14, color="#E45756",
                line=dict(width=2.5, color="#E45756")),
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
    .background_gradient(subset=["R²"], cmap="RdYlGn", vmin=0, vmax=1),
    width='stretch',
    hide_index=True,
)
