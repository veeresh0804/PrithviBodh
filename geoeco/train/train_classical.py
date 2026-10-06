"""Train classical models (M1/M2/M3/M5) with spatial GroupKFold CV + MLflow.

Point table (Parquet): one row per labelled point with feature columns
(see geoeco.models.classical), ``label`` (int 0..5), ``block_id`` (str/int).
Outer CV: 5-fold GroupKFold over blocks; inner: GridSearchCV over
max_features/min_samples_leaf (RF tuning note §6.4). Final model refits on
all blocks with best params and registers metrics + artefact in MLflow.

No secrets: MLflow tracking URI comes from ``MLFLOW_TRACKING_URI`` env var
(default local ``mlruns``).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV

from geoeco.evaluation.metrics import classification_scores
from geoeco.models.classical import (
    RF_PARAM_GRID,
    ModelID,
    build_rf,
    feature_columns,
)

N_SPLITS = 5


def run_spatial_cv(
    df: pd.DataFrame,
    model_id: ModelID,
    experiment: str = "geoeco-classical",
    seed: int = 42,
) -> dict:
    """5-fold spatial CV; logs params/metrics/model to MLflow. Returns summary."""
    cols = feature_columns(model_id)
    missing = [c for c in cols + ["label", "block_id"] if c not in df.columns]
    if missing:
        raise KeyError(f"Point table missing columns: {missing[:8]}...")
    X = df[cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=int)
    groups = df["block_id"].to_numpy()

    mlflow.set_experiment(experiment)
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_f1: list[float] = []

    with mlflow.start_run(run_name=f"{model_id}-spatialCV"):
        mlflow.log_param("model_id", model_id)
        mlflow.log_param("n_splits", N_SPLITS)
        mlflow.log_param("n_samples", len(df))
        for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
            grid = GridSearchCV(
                build_rf(random_state=seed),
                param_grid=RF_PARAM_GRID,
                cv=GroupKFold(n_splits=3),
                scoring="f1_macro",
                n_jobs=-1,
            )
            grid.fit(X[tr], y[tr], groups=groups[tr])
            pred = grid.predict(X[te])
            scores = classification_scores(y[te], pred)
            fold_f1.append(scores["macro_f1"])
            mlflow.log_metric(f"fold{fold}_macro_f1", scores["macro_f1"])
            mlflow.log_params({f"fold{fold}_{k}": v for k, v in grid.best_params_.items()})
        # Refit best-config model on all data (mode of best params: use grid winner).
        best = GridSearchCV(build_rf(random_state=seed), RF_PARAM_GRID,
                            cv=GroupKFold(n_splits=3), scoring="f1_macro", n_jobs=-1)
        best.fit(X, y, groups=groups)
        mlflow.log_params({f"final_{k}": v for k, v in best.best_params_.items()})
        mlflow.log_metric("macro_f1_mean", float(np.mean(fold_f1)))
        mlflow.log_metric("macro_f1_std", float(np.std(fold_f1)))
        mlflow.sklearn.log_model(best.best_estimator_, artifact_path=f"{model_id}-rf")
        return {"model_id": model_id, "fold_macro_f1": fold_f1,
                "mean": float(np.mean(fold_f1)), "std": float(np.std(fold_f1)),
                "best_params": best.best_params_}


def main() -> None:
    ap = argparse.ArgumentParser(description="Spatial-CV training for M1/M2/M3/M5")
    ap.add_argument("--points", required=True, help="Parquet point table")
    ap.add_argument("--model", required=True, choices=["M1", "M2", "M3", "M3-lgbm", "M5"])
    ap.add_argument("--experiment", default="geoeco-classical")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    df = pd.read_parquet(Path(args.points))
    summary = run_spatial_cv(df, args.model, args.experiment, args.seed)
    print(f"{args.model}: mean macro-F1={summary['mean']:.4f} ± {summary['std']:.4f}")


if __name__ == "__main__":
    main()
