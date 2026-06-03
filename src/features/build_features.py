"""
Feature engineering: district_daily_demand + weather → feature matrix for modelling.

Inputs (from data/processed/):
  district_daily_demand.parquet   district × day demand + active_stations
  weather_daily.parquet           daily Berlin weather

Output (to data/features/):
  features.parquet   one row per district × day, all features + target

Usage:
    python3 -m src.features.build_features
"""

import logging
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_DIR = ROOT / "data" / "features"

LAG_DAYS = [1, 2, 7, 14]
ROLL_WINDOWS = [3, 7, 14]
WEATHER_COLS = [
    "temperature_2m", "apparent_temperature", "precipitation",
    "rain", "snowfall", "wind_speed_10m", "cloud_cover", "relative_humidity_2m",
]
ANOMALY_PERCENTILE = 0.02  # city-wide days below this percentile are treated as outages

FEATURE_COLS = [
    "district",
    "dow", "month", "is_weekend", "is_holiday",
    *[f"lag_{l}d" for l in LAG_DAYS],
    *[f"roll_{w}d_{s}" for w in ROLL_WINDOWS for s in ["mean", "std"]],
    "active_stations",
    *WEATHER_COLS,
]


# ---------------------------------------------------------------------------
# Step 1 – Anomaly filtering
# ---------------------------------------------------------------------------
def filter_anomalous_days(demand: pd.DataFrame) -> pd.DataFrame:
    """Set rentals to NaN for dates where city-wide total is implausibly low.

    Lags and rolling features referencing these days will also become NaN
    and are dropped during the final dropna at save time.
    """
    city_daily = demand.groupby("date")["rentals"].sum()
    threshold = city_daily.quantile(ANOMALY_PERCENTILE)
    anomalous_dates = city_daily[city_daily < threshold].index

    log.info(
        "Anomaly filter: threshold %.0f city-wide rentals (p%.0f) — %d dates flagged",
        threshold, ANOMALY_PERCENTILE * 100, len(anomalous_dates),
    )

    demand = demand.copy()
    demand.loc[demand["date"].isin(anomalous_dates), "rentals"] = np.nan
    return demand


# ---------------------------------------------------------------------------
# Step 2 – Temporal features
# ---------------------------------------------------------------------------
def _add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    years = df["date"].dt.year.unique().tolist()
    berlin_holidays = set(holidays.Germany(state="BE", years=years).keys())

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    df["is_holiday"] = df["date"].dt.date.isin(berlin_holidays).astype(int)
    df["season"] = df["date"].dt.month.map({
        12: "winter", 1: "winter",  2: "winter",
         3: "spring", 4: "spring",  5: "spring",
         6: "summer", 7: "summer",  8: "summer",
         9: "autumn", 10: "autumn", 11: "autumn",
    })
    return df


# ---------------------------------------------------------------------------
# Step 3 – Lag and rolling features (per district)
# ---------------------------------------------------------------------------
def _add_lag_rolling(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("district")["rentals"]

    for lag in LAG_DAYS:
        df[f"lag_{lag}d"] = grp.shift(lag)

    for window in ROLL_WINDOWS:
        df[f"roll_{window}d_mean"] = grp.transform(
            lambda s, w=window: s.shift(1).rolling(w).mean()
        )
        df[f"roll_{window}d_std"] = grp.transform(
            lambda s, w=window: s.shift(1).rolling(w).std()
        )

    return df


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_features(demand: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    demand = demand.copy()
    demand["date"] = pd.to_datetime(demand["date"])
    weather = weather[["date"] + WEATHER_COLS].copy()
    weather["date"] = pd.to_datetime(weather["date"])

    demand = filter_anomalous_days(demand)

    df = demand.sort_values(["district", "date"]).copy()
    df["district"] = pd.Categorical(df["district"])

    df = _add_temporal(df)
    df = _add_lag_rolling(df)

    # Today's weather predicts tomorrow's demand
    df = df.merge(weather, on="date", how="left")

    # Target
    df = df.sort_values(["district", "date"])
    df["rentals_tomorrow"] = df.groupby("district")["rentals"].shift(-1)

    n_total = len(df)
    n_valid = df.dropna(subset=FEATURE_COLS + ["rentals_tomorrow"]).shape[0]
    log.info(
        "Feature matrix: %s rows × %s columns | valid modelling rows: %s",
        n_total, len(df.columns), n_valid,
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    demand = pd.read_parquet(PROCESSED_DIR / "district_daily_demand.parquet")
    weather = pd.read_parquet(PROCESSED_DIR / "weather_daily.parquet")
    log.info("Loaded demand: %s rows | weather: %s rows", len(demand), len(weather))

    features = build_features(demand, weather)

    out_cols = ["date", *FEATURE_COLS, "season", "rentals_tomorrow"]
    out_cols = list(dict.fromkeys(out_cols))  # deduplicate (district appears in both)
    out = features[out_cols].reset_index(drop=True)

    out_path = FEATURES_DIR / "features.parquet"
    out.to_parquet(out_path, index=False)
    log.info("Saved %s records to %s", len(out), out_path)


if __name__ == "__main__":
    main()
