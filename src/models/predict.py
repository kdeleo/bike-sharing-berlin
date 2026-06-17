"""
Generate next-day district-level demand predictions using today's live data.

Strategy: take the last known feature row per district from features.parquet,
then patch temporal, weather, and station features with today's actual values.
Lag/rolling features stay as the last known values (stale if ETL hasn't run
recently — accuracy degrades proportionally).

Requires (run beforehand):
    python3 -m src.data.collection.fetch_live

Outputs:
    data/predictions/predictions_latest.parquet
    data/predictions/predictions_YYYY-MM-DD.parquet  (tomorrow's date)

Usage:
    python3 -m src.models.predict
"""

import logging
from pathlib import Path

import geopandas as gpd
import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT             = Path(__file__).resolve().parents[2]
FEATURES_PATH    = ROOT / "data" / "features" / "features.parquet"
MODEL_PATH       = ROOT / "models" / "best_model.txt"
WEATHER_PATH     = ROOT / "data" / "processed" / "weather_daily.parquet"
BEZIRKE_PATH     = ROOT / "configs" / "berlin_bezirke.geojson"
SNAPSHOT_DIR     = ROOT / "bike_data_berlin"
PREDICTIONS_DIR  = ROOT / "data" / "predictions"

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
LOW_DEMAND_DISTRICTS = ["Marzahn-Hellersdorf", "Spandau", "Reinickendorf"]
TIMEZONE = "Europe/Berlin"


def _patch_temporal(base: pd.DataFrame, today: pd.Timestamp) -> None:
    berlin_holidays = set(
        holidays.Germany(state="BE", years=[today.year, (today + pd.Timedelta(days=1)).year]).keys()
    )
    base["date"]       = today
    base["dow"]        = today.dayofweek
    base["month"]      = today.month
    base["is_weekend"] = int(today.dayofweek >= 5)
    base["is_holiday"]     = int(today.date() in berlin_holidays)
    base["is_pre_holiday"]  = int((today + pd.Timedelta(days=1)).date() in berlin_holidays)
    base["is_post_holiday"] = int((today - pd.Timedelta(days=1)).date() in berlin_holidays)

    doy   = today.dayofyear
    B     = 2 * np.pi * (doy - 1) / 365
    decl  = (0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B)
             - 0.006758 * np.cos(2 * B) + 0.000907 * np.sin(2 * B)
             - 0.002697 * np.cos(3 * B) + 0.001480 * np.sin(3 * B))
    cos_ha = np.clip(-np.tan(np.deg2rad(52.52)) * np.tan(decl), -1, 1)
    base["daylight_hours"] = 2 * np.degrees(np.arccos(cos_ha)) / 15

    log.info(
        "Temporal: dow=%d  month=%d  is_weekend=%d  is_holiday=%d  "
        "is_pre_holiday=%d  is_post_holiday=%d  daylight_hours=%.2fh",
        base["dow"].iloc[0], base["month"].iloc[0],
        base["is_weekend"].iloc[0], base["is_holiday"].iloc[0],
        base["is_pre_holiday"].iloc[0], base["is_post_holiday"].iloc[0],
        base["daylight_hours"].iloc[0],
    )


def _patch_weather(
    base: pd.DataFrame,
    weather: pd.DataFrame,
    today: pd.Timestamp,
    tomorrow: pd.Timestamp,
) -> None:
    today_weather    = weather[weather["date"] == today]
    tomorrow_weather = weather[weather["date"] == tomorrow]

    if today_weather.empty:
        log.warning("No weather row for today — using last available row as fallback")
        today_weather = weather.sort_values("date").tail(1)
    if tomorrow_weather.empty:
        log.warning("No forecast row for tomorrow — run fetch_live first")

    tw = today_weather.iloc[0]
    for col in ["temperature_2m", "apparent_temperature", "precipitation",
                "rain", "snowfall", "wind_speed_10m", "cloud_cover", "relative_humidity_2m"]:
        base[col] = tw[col]

    yesterday = weather[weather["date"] < today].sort_values("date").tail(1)
    if not yesterday.empty:
        base["temp_change_1d"] = float(tw["temperature_2m"]) - float(yesterday["temperature_2m"].iloc[0])

    if not tomorrow_weather.empty:
        base["apparent_temperature_tomorrow"] = float(tomorrow_weather["apparent_temperature"].iloc[0])
        base["precipitation_tomorrow"]        = float(tomorrow_weather["precipitation"].iloc[0])

    base["apparent_temp_x_weekend"] = base["apparent_temperature"] * base["is_weekend"]

    log.info(
        "Weather: temp=%.1f°C  apparent=%.1f°C  temp_change_1d=%+.1f°C  "
        "apparent_tomorrow=%.1f°C  precip_tomorrow=%.1fmm",
        base["temperature_2m"].iloc[0], base["apparent_temperature"].iloc[0],
        base["temp_change_1d"].iloc[0],
        base["apparent_temperature_tomorrow"].iloc[0],
        base["precipitation_tomorrow"].iloc[0],
    )


def _patch_active_stations(base: pd.DataFrame, today: pd.Timestamp) -> None:
    snapshot_path = SNAPSHOT_DIR / f"live_{today.date()}.parquet"
    if not snapshot_path.exists():
        log.warning("%s not found — active_stations will stay stale", snapshot_path.name)
        return

    snapshot = pd.read_parquet(snapshot_path)
    bezirke  = gpd.read_file(BEZIRKE_PATH).to_crs("EPSG:4326")

    stations_gdf = gpd.GeoDataFrame(
        snapshot,
        geometry=gpd.points_from_xy(snapshot["longitude"], snapshot["latitude"]),
        crs="EPSG:4326",
    )
    bezirke_join = bezirke[["name", "geometry"]].rename(columns={"name": "district"})
    joined = gpd.sjoin(stations_gdf, bezirke_join, predicate="within", how="left")

    live_active = (
        joined.dropna(subset=["district"])
        .groupby("district")
        .size()
        .reset_index(name="active_stations_live")
    )

    base_merged = base.merge(live_active, on="district", how="left")
    base["active_stations"] = (
        base_merged["active_stations_live"].fillna(base["active_stations"]).values
    )
    log.info(
        "Active stations (live): %s",
        dict(zip(base_merged["district"], base_merged["active_stations_live"].fillna(base["active_stations"]))),
    )


def predict() -> pd.DataFrame:
    today    = pd.Timestamp.now(tz=TIMEZONE).normalize().tz_localize(None)
    tomorrow = today + pd.Timedelta(days=1)
    log.info("Today: %s  |  Predicting for: %s", today.date(), tomorrow.date())

    # ── Load ──────────────────────────────────────────────────────────────────
    features = pd.read_parquet(FEATURES_PATH)
    features["date"] = pd.to_datetime(features["date"])
    model    = lgb.Booster(model_file=str(MODEL_PATH))

    base = (
        features
        .dropna(subset=FEATURE_COLS)
        .sort_values("date")
        .groupby("district", observed=True)
        .last()
        .reset_index()
    )
    base = base[~base["district"].isin(LOW_DEMAND_DISTRICTS)].copy()
    staleness = (today - base["date"].max()).days
    log.info(
        "Base row date: %s  (lag staleness: %d days)  Districts: %d",
        base["date"].max().date(), staleness, len(base),
    )

    # ── Patch features ────────────────────────────────────────────────────────
    _patch_temporal(base, today)

    weather = pd.read_parquet(WEATHER_PATH)
    weather["date"] = pd.to_datetime(weather["date"])
    _patch_weather(base, weather, today, tomorrow)

    _patch_active_stations(base, today)

    # ── Predict ───────────────────────────────────────────────────────────────
    train_districts = (
        features[~features["district"].isin(LOW_DEMAND_DISTRICTS)]["district"]
        .cat.remove_unused_categories()
        .cat.categories
    )
    base["district"] = pd.Categorical(base["district"], categories=train_districts)

    base["pred_relative_demand"] = model.predict(base[FEATURE_COLS])
    base["pred_rentals"]         = base["pred_relative_demand"] * base["active_stations"]

    # ── Save ──────────────────────────────────────────────────────────────────
    predictions = base[[
        "district", "active_stations",
        "apparent_temperature_tomorrow", "precipitation_tomorrow",
        "pred_relative_demand", "pred_rentals",
    ]].copy()
    predictions.insert(0, "prediction_date", tomorrow)
    predictions.insert(1, "features_date",   today)
    predictions = predictions.sort_values("pred_rentals", ascending=False).reset_index(drop=True)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = PREDICTIONS_DIR / "predictions_latest.parquet"
    dated_path  = PREDICTIONS_DIR / f"predictions_{tomorrow.date()}.parquet"
    predictions.to_parquet(latest_path, index=False)
    predictions.to_parquet(dated_path,  index=False)

    log.info("Saved → %s", latest_path.name)
    log.info("Saved → %s", dated_path.name)
    log.info(
        "Predictions for %s:\n%s",
        tomorrow.date(),
        predictions[["district", "pred_relative_demand", "pred_rentals", "active_stations"]]
        .to_string(index=False),
    )

    return predictions


def main() -> None:
    predict()


if __name__ == "__main__":
    main()
