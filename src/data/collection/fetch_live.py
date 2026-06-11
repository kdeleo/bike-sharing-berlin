"""
Fetch today's station snapshot from the CityBikes API and tomorrow's weather
forecast from Open-Meteo, so the daily prediction pipeline has up-to-date inputs.

Outputs:
  bike_data_berlin/live_YYYY-MM-DD.parquet   today's station snapshot (one per day)
  data/processed/weather_daily.parquet       updated with tomorrow's forecast row

The snapshot file is named so the existing ETL pipeline picks it up automatically
on the next run (it globs all *.parquet files in bike_data_berlin/).

Usage:
    python3 -m src.data.collection.fetch_live
"""

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import openmeteo_requests
import pandas as pd
import requests
import requests_cache
from retry_requests import retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parents[3]
SNAPSHOT_DIR = ROOT / "bike_data_berlin"
WEATHER_PATH = ROOT / "data" / "processed" / "weather_daily.parquet"

TIMEZONE      = "Europe/Berlin"
CITYBIKES_URL = "https://api.citybik.es/v2/networks/{}"
FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"

# Maps CityBikes network ID → tag value used in existing parquet schema
SYSTEMS = {
    "nextbike-berlin" : "nextbike-berlin",
    "callabike-berlin": "callabike-berlin",
}
HOURLY_VARS = [
    "temperature_2m", "precipitation", "apparent_temperature",
    "rain", "snowfall", "weather_code", "wind_speed_10m",
    "relative_humidity_2m", "cloud_cover",
]


# ---------------------------------------------------------------------------
# Station snapshot
# ---------------------------------------------------------------------------
def fetch_snapshot(network_id: str, tag: str) -> pd.DataFrame:
    """Fetch current station states for one CityBikes network."""
    url = CITYBIKES_URL.format(network_id)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    stations = resp.json()["network"]["stations"]

    # Stored as naive UTC to match existing parquet schema;
    # the ETL pipeline does tz_localize("UTC") on ingest.
    now = pd.Timestamp.utcnow().replace(tzinfo=None)
    df = pd.DataFrame([
        {
            "tag"      : tag,
            "id"       : s["id"],
            "nuid"     : s["id"],
            "name"     : s["name"],
            "latitude" : s["latitude"],
            "longitude": s["longitude"],
            "bikes"    : np.int32(s["free_bikes"] or 0),
            "free"     : float(s["empty_slots"]) if s["empty_slots"] is not None else np.nan,
            "extra"    : s.get("extra", {}),
            "timestamp": now,
        }
        for s in stations
    ])
    log.info("%s: fetched %d stations", tag, len(df))
    return df


# ---------------------------------------------------------------------------
# Weather forecast
# ---------------------------------------------------------------------------
def _fetch_hourly_forecast() -> pd.DataFrame:
    """Fetch 48-hour hourly forecast from Open-Meteo (today + tomorrow)."""
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude"     : 52.52,
        "longitude"    : 13.41,
        "hourly"       : HOURLY_VARS,
        "forecast_days": 2,
    }
    response = client.weather_api(FORECAST_URL, params=params)[0]
    hourly   = response.Hourly()

    data = {
        "timestamp": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    for i, var in enumerate(HOURLY_VARS):
        data[var] = hourly.Variables(i).ValuesAsNumpy()

    hourly_df = pd.DataFrame(data)
    hourly_df["timestamp"] = pd.to_datetime(hourly_df["timestamp"], utc=True).dt.tz_convert(TIMEZONE)
    hourly_df["date"] = hourly_df["timestamp"].dt.date
    return hourly_df


def aggregate_to_daily(hourly_df: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """Aggregate hourly forecast to a single daily row for target_date."""
    day = hourly_df[hourly_df["date"] == target_date]
    if day.empty:
        raise ValueError(f"No forecast data for {target_date}")

    daily = day.agg({
        "temperature_2m"      : "mean",
        "apparent_temperature": "mean",
        "precipitation"       : "sum",
        "rain"                : "sum",
        "snowfall"            : "sum",
        "wind_speed_10m"      : "mean",
        "cloud_cover"         : "mean",
        "relative_humidity_2m": "mean",
        "weather_code"        : lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    }).to_frame().T
    daily.insert(0, "date", pd.Timestamp(target_date))
    return daily.reset_index(drop=True)


def update_weather_cache(forecast_df: pd.DataFrame) -> None:
    """Upsert forecast rows into weather_daily.parquet (replace if date exists)."""
    if WEATHER_PATH.exists():
        existing = pd.read_parquet(WEATHER_PATH)
        existing["date"] = pd.to_datetime(existing["date"])
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])
        mask    = ~existing["date"].isin(forecast_df["date"])
        updated = pd.concat([existing[mask], forecast_df], ignore_index=True).sort_values("date")
    else:
        updated = forecast_df

    updated.to_parquet(WEATHER_PATH, index=False)
    log.info("Weather cache updated: %d rows  (last date: %s)",
             len(updated), updated["date"].max().date())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    today    = date.today()
    tomorrow = today + timedelta(days=1)

    # ── Station snapshot ──────────────────────────────────────────────────────
    out_path = SNAPSHOT_DIR / f"live_{today.isoformat()}.parquet"
    if out_path.exists():
        log.info("Snapshot for %s already exists, skipping fetch", today)
    else:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        frames = []
        for network_id, tag in SYSTEMS.items():
            try:
                frames.append(fetch_snapshot(network_id, tag))
            except Exception as exc:
                log.warning("Failed to fetch %s: %s", tag, exc)

        if not frames:
            raise RuntimeError("No station data fetched — all networks failed")

        snapshot = pd.concat(frames, ignore_index=True)
        snapshot.to_parquet(out_path, index=False)
        log.info("Saved snapshot → %s  (%d rows)", out_path.name, len(snapshot))

    # ── Tomorrow's weather forecast ───────────────────────────────────────────
    try:
        hourly_df   = _fetch_hourly_forecast()
        forecast_df = aggregate_to_daily(hourly_df, tomorrow)
        log.info("Tomorrow (%s): %.1f°C apparent, %.1f mm precip",
                 tomorrow,
                 forecast_df["apparent_temperature"].iloc[0],
                 forecast_df["precipitation"].iloc[0])
        update_weather_cache(forecast_df)
    except Exception as exc:
        log.warning("Weather forecast fetch failed: %s", exc)


if __name__ == "__main__":
    main()
