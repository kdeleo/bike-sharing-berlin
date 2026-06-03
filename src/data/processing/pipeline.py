"""
Data processing pipeline: raw bike snapshots + weather → district-level daily demand.

Outputs saved to data/processed/:
  - district_daily_demand.parquet   district × day demand + active stations
  - weather_daily.parquet           daily weather for Berlin

Plots saved to reports/.

Usage:
    python3 -m src.data.processing.pipeline
    python3 -m src.data.processing.pipeline --data-dir bike_data_berlin --no-plots
"""

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import holidays
import matplotlib.pyplot as plt
import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "bike_data_berlin"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
BEZIRKE_PATH = ROOT / "configs" / "berlin_bezirke.geojson"

TIMEZONE = "Europe/Berlin"
SYSTEMS = ["nextbike-berlin", "callabike-berlin"]
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m", "precipitation", "apparent_temperature",
    "rain", "snowfall", "weather_code", "wind_speed_10m",
    "relative_humidity_2m", "cloud_cover",
]


# ---------------------------------------------------------------------------
# Step 1 – Load raw data
# ---------------------------------------------------------------------------
def load_raw_data(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.parquet"))
    log.info("Found %d parquet files in %s", len(files), data_dir)

    dfs = [pd.read_parquet(f) for f in files]
    raw = pd.concat(dfs, ignore_index=True)
    raw = raw[raw["tag"].isin(SYSTEMS)].copy()

    log.info(
        "Loaded %s rows | %s unique stations | %s → %s",
        f"{len(raw):,}",
        f"{raw['nuid'].nunique():,}",
        raw["timestamp"].min().date(),
        raw["timestamp"].max().date(),
    )
    return raw


# ---------------------------------------------------------------------------
# Step 2 – Compute daily demand per station
# ---------------------------------------------------------------------------
def compute_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["nuid", "timestamp"]).copy()
    df["date"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(TIMEZONE).dt.date

    grp = df.groupby(["nuid", "date"])
    df["prev_bikes"] = grp["bikes"].shift(1)
    df["rentals"] = (df["prev_bikes"] - df["bikes"]).clip(lower=0).fillna(0)

    daily = (
        df.groupby(["nuid", "name", "latitude", "longitude", "date"])
        .agg(rentals=("rentals", "sum"), snapshots=("bikes", "count"))
        .reset_index()
    )
    return daily


# ---------------------------------------------------------------------------
# Step 3 – Assign stations to Berlin districts (Bezirke)
# ---------------------------------------------------------------------------
def assign_districts(demand_df: pd.DataFrame, bezirke_path: Path) -> pd.DataFrame:
    bezirke = (
        gpd.read_file(bezirke_path)[["name", "geometry"]]
        .rename(columns={"name": "district"})
        .to_crs("EPSG:4326")
    )

    stations = demand_df[["nuid", "latitude", "longitude"]].drop_duplicates("nuid").dropna()
    stations_gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["longitude"], stations["latitude"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(stations_gdf, bezirke, how="left", predicate="within")
    joined = joined[["nuid", "district"]].drop_duplicates("nuid")

    n_matched = joined["district"].notna().sum()
    log.info(
        "District assignment: %s / %s stations matched (%.1f%%)",
        f"{n_matched:,}", f"{len(joined):,}", n_matched / len(joined) * 100,
    )
    return joined


# ---------------------------------------------------------------------------
# Step 4 – Fetch or load weather data
# ---------------------------------------------------------------------------
def fetch_hourly_weather(start_date: str, end_date: str) -> pd.DataFrame:
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": 52.52,
        "longitude": 13.41,
        "hourly": HOURLY_VARS,
        "start_date": start_date,
        "end_date": end_date,
    }
    response = client.weather_api(ARCHIVE_URL, params=params)[0]
    hourly = response.Hourly()

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

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TIMEZONE)
    return df


def aggregate_weather_to_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
    hourly_df["date"] = hourly_df["timestamp"].dt.date
    daily = hourly_df.groupby("date").agg(
        temperature_2m=("temperature_2m", "mean"),
        apparent_temperature=("apparent_temperature", "mean"),
        precipitation=("precipitation", "sum"),
        rain=("rain", "sum"),
        snowfall=("snowfall", "sum"),
        wind_speed_10m=("wind_speed_10m", "mean"),
        cloud_cover=("cloud_cover", "mean"),
        relative_humidity_2m=("relative_humidity_2m", "mean"),
        weather_code=("weather_code", lambda x: x.mode()[0] if not x.mode().empty else np.nan),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def get_weather(raw: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        log.info("Loading weather from cache: %s", cache_path)
        return pd.read_parquet(cache_path)

    start = raw["timestamp"].min().date().isoformat()
    end = raw["timestamp"].max().date().isoformat()
    log.info("Fetching weather from Open-Meteo: %s → %s", start, end)

    hourly = fetch_hourly_weather(start, end)
    daily = aggregate_weather_to_daily(hourly)
    daily.to_parquet(cache_path, index=False)
    log.info("Weather saved to %s", cache_path)
    return daily


# ---------------------------------------------------------------------------
# Step 5 – Build district × day demand table
# ---------------------------------------------------------------------------
def build_district_daily(
    demand_nb: pd.DataFrame,
    demand_cb: pd.DataFrame,
    stations_joined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    demand_all = pd.concat([
        demand_nb.assign(system="nextbike"),
        demand_cb.assign(system="callabike"),
    ])
    demand_all = demand_all.merge(stations_joined, on="nuid", how="left")
    demand_all["date"] = pd.to_datetime(demand_all["date"])

    district_daily = (
        demand_all.dropna(subset=["district"])
        .groupby(["date", "district"])["rentals"]
        .sum()
        .reset_index()
    )

    active_stations = (
        demand_all.dropna(subset=["district"])
        .groupby(["date", "district"])["nuid"]
        .nunique()
        .reset_index()
        .rename(columns={"nuid": "active_stations"})
    )

    out = district_daily.merge(active_stations, on=["date", "district"])
    out = out.sort_values(["district", "date"]).reset_index(drop=True)
    return out, demand_all


# ---------------------------------------------------------------------------
# Step 6 – Save plots
# ---------------------------------------------------------------------------
def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path.name)


def save_plots(
    raw: pd.DataFrame,
    district_daily: pd.DataFrame,
    demand_all: pd.DataFrame,
    weather_daily: pd.DataFrame,
    reports_dir: Path,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bikes distribution by system
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, (tag, grp) in zip(axes, raw.groupby("tag")):
        grp["bikes"].clip(0, 30).value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"{tag} — bikes per snapshot")
        ax.set_xlabel("Bikes available")
        ax.set_ylabel("Count")
    plt.tight_layout()
    _savefig(fig, reports_dir / "bikes_distribution.png")

    # 2. Station count over time
    raw["month"] = raw["timestamp"].dt.to_period("M")
    station_counts = raw.groupby(["month", "tag"])["nuid"].nunique().unstack("tag")
    fig, ax = plt.subplots(figsize=(12, 4))
    station_counts.plot(ax=ax, marker="o", markersize=4)
    ax.set_title("Active stations per month by system")
    ax.set_ylabel("Unique station IDs")
    ax.set_xlabel("")
    plt.tight_layout()
    _savefig(fig, reports_dir / "station_counts_over_time.png")

    # 3. City-wide daily demand (by system)
    nb_city = demand_all[demand_all["system"] == "nextbike"].groupby("date")["rentals"].sum()
    cb_city = demand_all[demand_all["system"] == "callabike"].groupby("date")["rentals"].sum()
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(nb_city.index, nb_city.values, label="nextbike", color="steelblue")
    ax.plot(cb_city.index, cb_city.values, label="callabike", color="tomato")
    ax.set_title("Estimated city-wide daily rentals")
    ax.set_ylabel("Rentals")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, reports_dir / "city_daily_demand.png")

    # 4. District demand time series
    fig, ax = plt.subplots(figsize=(13, 5))
    for district, grp in district_daily.groupby("district"):
        grp_sorted = grp.sort_values("date")
        ax.plot(grp_sorted["date"], grp_sorted["rentals"], label=district)
    ax.set_title("Daily rentals by Berlin district")
    ax.set_ylabel("Estimated rentals")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    _savefig(fig, reports_dir / "district_demand_timeseries.png")

    # 5. Average rentals by day-of-week per district
    district_daily["dow"] = district_daily["date"].dt.dayofweek
    district_daily["dow_name"] = district_daily["date"].dt.day_name()
    dow_avg = (
        district_daily.groupby(["district", "dow", "dow_name"])["rentals"]
        .mean()
        .reset_index()
        .sort_values(["district", "dow"])
    )
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    districts = dow_avg["district"].unique()
    x = np.arange(7)
    width = 0.8 / len(districts)
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, district in enumerate(sorted(districts)):
        vals = dow_avg[dow_avg["district"] == district].set_index("dow_name").reindex(dow_order)["rentals"]
        ax.bar(x + i * width, vals, width=width, label=district)
    ax.set_xticks(x + width * len(districts) / 2)
    ax.set_xticklabels(dow_order)
    ax.set_title("Average daily rentals by day of week and district")
    ax.set_ylabel("Avg rentals")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    _savefig(fig, reports_dir / "district_dow_avg.png")

    # 6. Monthly demand by district (seasonality)
    district_daily["month"] = district_daily["date"].dt.month
    district_daily["month_name"] = district_daily["date"].dt.strftime("%b")
    month_avg = (
        district_daily.groupby(["district", "month", "month_name"])["rentals"]
        .mean()
        .reset_index()
        .sort_values(["district", "month"])
    )
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(13, 5))
    for district, grp in month_avg.groupby("district"):
        grp_sorted = grp.sort_values("month")
        ax.plot(grp_sorted["month_name"], grp_sorted["rentals"], marker="o", label=district)
    ax.set_title("Average daily rentals by month (seasonality)")
    ax.set_ylabel("Avg rentals/day")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    _savefig(fig, reports_dir / "district_seasonality.png")

    # 7. District × month heatmap
    pivot = month_avg.pivot(index="district", columns="month_name", values="rentals")
    pivot = pivot[[m for m in month_order if m in pivot.columns]]
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.0f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="Avg daily rentals")
    ax.set_title("Avg daily rentals — district × month")
    plt.tight_layout()
    _savefig(fig, reports_dir / "district_month_heatmap.png")

    # 8. Weather overview
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(weather_daily["date"], weather_daily["temperature_2m"], color="tomato", label="Mean temp")
    axes[0].plot(weather_daily["date"], weather_daily["apparent_temperature"], color="salmon", linestyle="--", label="Apparent temp")
    axes[0].set_ylabel("°C")
    axes[0].set_title("Daily temperature")
    axes[0].legend()
    axes[1].bar(weather_daily["date"], weather_daily["precipitation"], color="steelblue", label="Precipitation")
    axes[1].bar(weather_daily["date"], weather_daily["snowfall"], color="lightblue", label="Snowfall")
    axes[1].set_ylabel("mm")
    axes[1].set_title("Daily precipitation & snowfall")
    axes[1].legend()
    axes[2].plot(weather_daily["date"], weather_daily["wind_speed_10m"], color="grey", label="Wind speed")
    axes[2].plot(weather_daily["date"], weather_daily["cloud_cover"], color="darkgrey", linestyle="--", label="Cloud cover %")
    axes[2].set_ylabel("m/s  /  %")
    axes[2].set_title("Wind speed & cloud cover")
    axes[2].legend()
    plt.tight_layout()
    _savefig(fig, reports_dir / "weather_overview.png")

    # 9. Weather–demand correlation
    city_total = district_daily.groupby("date")["rentals"].sum().reset_index().rename(columns={"rentals": "total_rentals"})
    weather_demand = city_total.merge(weather_daily, on="date", how="inner")
    weather_cols = ["temperature_2m", "apparent_temperature", "precipitation",
                    "rain", "snowfall", "wind_speed_10m", "cloud_cover", "relative_humidity_2m"]
    corr = weather_demand[weather_cols + ["total_rentals"]].corr()["total_rentals"].drop("total_rentals").sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    corr.plot(kind="barh", ax=ax, color=["tomato" if v < 0 else "steelblue" for v in corr])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Correlation of weather features with city-wide daily rentals")
    ax.set_xlabel("Pearson r")
    plt.tight_layout()
    _savefig(fig, reports_dir / "weather_demand_correlation.png")

    # 10. Active stations per district over time
    active = (
        demand_all.dropna(subset=["district"])
        .groupby(["date", "district"])["nuid"]
        .nunique()
        .reset_index()
        .rename(columns={"nuid": "active_stations"})
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    for district, grp in active.groupby("district"):
        grp_sorted = grp.sort_values("date")
        ax.plot(grp_sorted["date"], grp_sorted["active_stations"], label=district)
    ax.set_title("Active stations per district over time")
    ax.set_ylabel("Unique active stations")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    _savefig(fig, reports_dir / "active_stations_over_time.png")

    # 11. Feature correlation + lag autocorrelation for the busiest district
    target_district = district_daily["district"].value_counts().index[0]
    target_df = (
        district_daily[district_daily["district"] == target_district]
        .sort_values("date")
        .set_index("date")[["rentals"]]
        .copy()
    )
    de_holidays = holidays.Germany(state="BE", years=range(2025, 2027))
    target_df["dow"] = target_df.index.dayofweek
    target_df["month"] = target_df.index.month
    target_df["is_weekend"] = (target_df.index.dayofweek >= 5).astype(int)
    target_df["is_holiday"] = target_df.index.map(lambda d: int(d in de_holidays))
    for lag in [1, 2, 7, 14]:
        target_df[f"lag_{lag}d"] = target_df["rentals"].shift(lag)
    for window in [3, 7, 14]:
        target_df[f"roll_{window}d_mean"] = target_df["rentals"].shift(1).rolling(window).mean()
        target_df[f"roll_{window}d_std"] = target_df["rentals"].shift(1).rolling(window).std()
    weather_cols_feat = ["temperature_2m", "apparent_temperature", "precipitation",
                         "rain", "snowfall", "wind_speed_10m", "cloud_cover", "relative_humidity_2m"]
    target_df = target_df.merge(
        weather_daily[["date"] + weather_cols_feat].set_index("date"),
        left_index=True, right_index=True, how="left",
    )
    target_df["rentals_tomorrow"] = target_df["rentals"].shift(-1)

    feat_cols = [c for c in target_df.columns if c not in ("rentals", "rentals_tomorrow")]
    corr_feat = target_df[feat_cols + ["rentals_tomorrow"]].dropna().corr()["rentals_tomorrow"].drop("rentals_tomorrow").sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    corr_feat.plot(kind="barh", ax=ax, color=["tomato" if v < 0 else "steelblue" for v in corr_feat])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Feature correlation with rentals_tomorrow ({target_district})")
    ax.set_xlabel("Pearson r")
    plt.tight_layout()
    _savefig(fig, reports_dir / "feature_correlation.png")

    # 12. Lag autocorrelation
    lag_corrs = {}
    for lag in range(1, 22):
        lagged = target_df["rentals"].shift(lag)
        valid = target_df["rentals"].notna() & lagged.notna()
        lag_corrs[lag] = target_df["rentals"][valid].corr(lagged[valid])
    fig, ax = plt.subplots(figsize=(10, 4))
    pd.Series(lag_corrs).plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(f"Autocorrelation of daily rentals — {target_district}")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Pearson r")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    _savefig(fig, reports_dir / "lag_autocorrelation.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(data_dir: Path, save_plots_flag: bool) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load
    raw = load_raw_data(data_dir)

    # 2. Demand
    log.info("Computing daily demand for nextbike-berlin...")
    demand_nb = compute_daily_demand(raw[raw["tag"] == "nextbike-berlin"])
    log.info("  %s station-day records", f"{len(demand_nb):,}")

    log.info("Computing daily demand for callabike-berlin...")
    demand_cb = compute_daily_demand(raw[raw["tag"] == "callabike-berlin"])
    log.info("  %s station-day records", f"{len(demand_cb):,}")

    # 3. District assignment
    all_demand = pd.concat([demand_nb, demand_cb])
    stations_joined = assign_districts(all_demand, BEZIRKE_PATH)

    # 4. Weather
    weather_daily = get_weather(raw, PROCESSED_DIR / "weather_daily.parquet")

    # 5. District daily table
    district_daily, demand_all = build_district_daily(demand_nb, demand_cb, stations_joined)
    out_path = PROCESSED_DIR / "district_daily_demand.parquet"
    district_daily.to_parquet(out_path, index=False)
    log.info("Saved %s records to %s", f"{len(district_daily):,}", out_path)

    # 6. Plots
    if save_plots_flag:
        save_plots(raw, district_daily.copy(), demand_all, weather_daily, REPORTS_DIR)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bike-sharing data processing pipeline")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--no-plots", action="store_true", help="Skip saving plots")
    args = parser.parse_args()

    main(data_dir=args.data_dir, save_plots_flag=not args.no_plots)
