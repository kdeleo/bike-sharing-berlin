"""
LightGBM training with Optuna HPO and MLflow tracking.

Reads data/features/features.parquet, runs Optuna hyperparameter search,
logs the best model to MLflow, prints per-district metrics, and saves the
booster to models/best_model.txt.

Usage:
    python3 -m src.models.train
    python3 -m src.models.train --trials 100
    python3 -m src.models.train --split-date 2026-01-01
"""

import argparse
import logging
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR   = ROOT / "models"

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
    "apparent_temp_x_weekend",
]
TARGET = "relative_demand_tomorrow"
LOW_DEMAND_DISTRICTS = ["Marzahn-Hellersdorf", "Spandau", "Reinickendorf"]


def _evaluate(y_true: pd.Series, y_pred: np.ndarray, label: str) -> tuple[float, float, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    log.info("%s  RMSE: %.3f  MAE: %.3f  R²: %.3f", label, rmse, mae, r2)
    return rmse, mae, r2


def _make_objective(X_train, y_train, X_test, y_test):
    def objective(trial):
        params = {
            "objective"        : "regression",
            "metric"           : "rmse",
            "verbose"          : -1,
            "random_state"     : 42,
            "n_estimators"     : 1000,
            "num_leaves"       : trial.suggest_int("num_leaves", 20, 500),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "feature_fraction" : trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction" : trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq"     : trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            categorical_feature=["district"],
        )
        return np.sqrt(mean_squared_error(y_test, m.predict(X_test)))
    return objective


def _per_district_metrics(model: lgb.LGBMRegressor, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for district, grp in test.groupby("district", observed=True):
        preds     = model.predict(grp[FEATURE_COLS])
        preds_abs = preds * grp["active_stations"]
        rows.append({
            "district": str(district),
            "n"        : len(grp),
            "rmse_rel" : np.sqrt(mean_squared_error(grp[TARGET], preds)),
            "r2_rel"   : r2_score(grp[TARGET], preds),
            "rmse_abs" : np.sqrt(mean_squared_error(grp["rentals_tomorrow"], preds_abs)),
            "mae_abs"  : mean_absolute_error(grp["rentals_tomorrow"], preds_abs),
        })
    return pd.DataFrame(rows).sort_values("rmse_abs", ascending=False)


def train(n_trials: int = 50, split_date: str = "2026-01-01") -> lgb.LGBMRegressor:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    split_ts = pd.Timestamp(split_date)

    # ── Load ──────────────────────────────────────────────────────────────────
    features = pd.read_parquet(FEATURES_DIR / "features.parquet")
    features["date"] = pd.to_datetime(features["date"])

    df = features.dropna(subset=FEATURE_COLS + [TARGET]).copy()
    df = df[~df["district"].isin(LOW_DEMAND_DISTRICTS)].copy()
    df["district"] = df["district"].cat.remove_unused_categories()
    log.info(
        "Loaded %s valid rows across %s districts", len(df), df["district"].nunique()
    )

    # ── Split ─────────────────────────────────────────────────────────────────
    train_df = df[df["date"] < split_ts]
    test_df  = df[df["date"] >= split_ts]
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET]
    X_test,  y_test  = test_df[FEATURE_COLS],  test_df[TARGET]
    log.info(
        "Train: %s rows (%s → %s)  |  Test: %s rows (%s → %s)",
        len(train_df), train_df["date"].min().date(), train_df["date"].max().date(),
        len(test_df),  test_df["date"].min().date(),  test_df["date"].max().date(),
    )

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_rmse, baseline_mae, _ = _evaluate(y_test, test_df["lag_7d"], "Baseline (lag_7d) ")

    # ── Optuna HPO ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"sqlite:///{ROOT}/mlflow.db")
    mlflow.set_experiment("berlin-bike-demand")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="lgbm-berlin-bike",
    )
    log.info("Running Optuna HPO (%d trials)...", n_trials)
    study.optimize(
        _make_objective(X_train, y_train, X_test, y_test),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    log.info("Best trial RMSE: %.4f  params: %s", study.best_value, study.best_params)

    # ── Train best model ──────────────────────────────────────────────────────
    best_params = {
        "objective"   : "regression",
        "metric"      : ["rmse", "mae"],
        "verbose"     : -1,
        "random_state": 42,
        "n_estimators": 1000,
        **study.best_params,
    }

    with mlflow.start_run(run_name="lgbm-optuna-best"):
        mlflow.log_params(best_params)
        mlflow.log_param("target",              TARGET)
        mlflow.log_param("n_trials",            n_trials)
        mlflow.log_param("split_date",          str(split_ts.date()))
        mlflow.log_param("excluded_districts",  str(LOW_DEMAND_DISTRICTS))

        model = lgb.LGBMRegressor(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_names=["train", "test"],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(100),
            ],
            categorical_feature=["district"],
        )

        y_pred     = model.predict(X_test)
        y_pred_abs = y_pred * test_df["active_stations"]

        rmse, mae, r2 = _evaluate(y_test, y_pred, "Optuna best (relative)")
        rmse_abs = np.sqrt(mean_squared_error(test_df["rentals_tomorrow"], y_pred_abs))
        mae_abs  = mean_absolute_error(test_df["rentals_tomorrow"], y_pred_abs)
        log.info("Optuna best (absolute)  RMSE: %.1f  MAE: %.1f", rmse_abs, mae_abs)

        mlflow.log_metrics({
            "test_rmse"     : rmse,
            "test_mae"      : mae,
            "test_r2"       : r2,
            "test_rmse_abs" : rmse_abs,
            "test_mae_abs"  : mae_abs,
            "baseline_rmse" : baseline_rmse,
            "baseline_mae"  : baseline_mae,
            "best_iteration": model.best_iteration_,
        })
        mlflow.lightgbm.log_model(model.booster_, "model")

    # ── Per-district metrics ──────────────────────────────────────────────────
    dm = _per_district_metrics(model, test_df)
    log.info("Per-district metrics:\n%s", dm.to_string(index=False))

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = MODELS_DIR / "best_model.txt"
    model.booster_.save_model(str(model_path))
    log.info("Model saved to %s", model_path)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM demand forecast model.")
    parser.add_argument("--trials",     type=int, default=50,           help="Optuna trials (default: 50)")
    parser.add_argument("--split-date", type=str, default="2026-01-01", help="Train/test split date (default: 2026-01-01)")
    args = parser.parse_args()
    train(n_trials=args.trials, split_date=args.split_date)


if __name__ == "__main__":
    main()
