# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

End-to-end ML pipeline that predicts next-day bike-sharing demand in Berlin to guide operational bike redistribution — ensuring bikes are available where needed. Raw station snapshots (nextbike-berlin + callabike-berlin) are processed into district-level daily demand, combined with weather data, and fed to a regression model. Results are served via API and visualised in Streamlit; the pipeline is scheduled with Airflow and packaged with Docker for cloud deployment.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full ETL pipeline (processes raw data, fetches weather, saves processed files, generates plots)
python3 -m src.data.processing.pipeline

# Run pipeline without generating plots
python3 -m src.data.processing.pipeline --no-plots

# Run pipeline with a custom data directory
python3 -m src.data.processing.pipeline --data-dir /path/to/data

# Start JupyterLab
jupyter lab

# Run tests
pytest
pytest tests/path/to/test_file.py::test_name  # single test
pytest --cov=src                               # with coverage
```

## Architecture

### Data flow

```
bike_data_berlin/*.parquet   configs/berlin_bezirke.geojson   Open-Meteo API
        |                              |                            |
        v                              v                            v
src/data/processing/pipeline.py  (ETL — implemented)
        |
        v
data/processed/
  district_daily_demand.parquet   # date × district × rentals × active_stations
  weather_daily.parquet           # daily weather for Berlin (cached after first fetch)
        |
        v
src/features/    (feature engineering — planned)
        |
        v
src/models/      (LightGBM + Optuna + MLflow — planned)
        |
        ├── src/api/          (FastAPI — planned)
        ├── src/monitoring/   (Evidently drift — planned)
        └── Streamlit app     (visualisation — planned)

Orchestration: Airflow   |   Packaging: Docker
```

### Key design decisions

**Demand estimation** — The raw data is station-level snapshots (bikes available at a point in time), not transactions. Demand is estimated as `Σ max(0, bikes_prev - bikes_curr)` within each station × calendar day. Only within-day transitions are used. No gap threshold is applied because snapshot intervals are too irregular (nextbike median ~57 min, callabike median ~6 h). Rebalancing noise partially cancels at district level.

**District assignment** — Stations are spatially joined to Berlin's 12 Bezirke using GeoPandas (`sjoin` with `predicate="within"`, CRS EPSG:4326). 99.8% of stations match. The boundaries file is `configs/berlin_bezirke.geojson`.

**Weather** — Fetched from Open-Meteo archive API (free, no key). Hourly data aggregated to daily. Cached to `data/processed/weather_daily.parquet` on first run; subsequent runs skip the API call.

**Timezone** — All timestamps are converted from UTC to `Europe/Berlin` before any date-based aggregation.

### Feature engineering (planned, not yet implemented)

Target: `rentals_tomorrow` per district. Five feature groups:
- **Temporal**: dow, month, is_weekend, is_holiday (Berlin state holidays via `holidays.Germany(state="BE")`)
- **Lag**: rentals 1/2/7/14 days ago
- **Rolling**: 3/7/14-day rolling mean and std of lagged demand
- **Network**: active station count per district
- **Weather**: temperature, apparent temperature, precipitation, rain, snowfall, wind, cloud cover, humidity

### Package layout

| Path | Status | Purpose |
|---|---|---|
| `src/data/processing/pipeline.py` | Implemented | Full ETL pipeline |
| `src/features/` | Stub | Feature engineering |
| `src/models/` | Stub | LightGBM training + Optuna HPO + MLflow tracking |
| `src/api/` | Stub | FastAPI serving endpoint |
| `src/monitoring/` | Stub | Evidently drift monitoring |
| Streamlit app | Not started | Interactive demand visualisation |
| Airflow DAGs | Not started | Scheduling daily pipeline runs |
| Docker | Not started | Containerisation for deployment |

### Data

- `bike_data_berlin/` — monthly parquet snapshots, Jan 2025–Apr 2026; columns: `tag`, `nuid`, `name`, `latitude`, `longitude`, `bikes`, `free`, `timestamp`
- Only `tag` values `nextbike-berlin` and `callabike-berlin` are used (other tags such as `nextbike-campus-berlin-buch` are filtered out)
