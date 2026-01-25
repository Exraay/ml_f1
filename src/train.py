from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data import build_base_dataset
from src.encoding import apply_target_encoding
from src.eval import evaluate_models
from src.feature_enhancements import add_enhanced_features
from src.models import build_search, make_model_registry
from src.plots import (
    plot_actual_vs_pred,
    plot_error_by_category,
    plot_error_distribution,
    plot_model_comparison,
    save_plot,
)
from src.split import SplitConfig, split_by_season_round


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DEFAULT_DATASET = DATA_DIR / "laps_base.parquet"
DEFAULT_METADATA = DATA_DIR / "laps_base.metadata.json"
TARGET_COL = "LapTimeSeconds"


def _save_metadata(metadata: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_metadata(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_and_save_dataset(
    years: List[int],
    *,
    output_path: Path = DEFAULT_DATASET,
    metadata_path: Path = DEFAULT_METADATA,
    include_physics: bool = True,
    exclude_lap1: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    df, numeric_features, categorical_features = build_base_dataset(
        years,
        include_physics=include_physics,
        exclude_lap1=exclude_lap1,
        verbose=verbose,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    metadata = {
        "created_at": datetime.utcnow().isoformat(),
        "dataset": str(output_path.as_posix()),
        "seasons": years,
        "include_physics": include_physics,
        "exclude_lap1": exclude_lap1,
        "target": TARGET_COL,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    _save_metadata(metadata, metadata_path)
    return df, metadata


def _prepare_features(
    df: pd.DataFrame,
    metadata: Dict,
    *,
    split_config: SplitConfig,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    train_df, val_df, test_df = split_by_season_round(df, config=split_config)

    # Train-only enhancements for tuning phase
    train_enh, val_enh, added_cols = add_enhanced_features(train_df, val_df)

    # Train-only enhancements for final test (fit on train+val)
    trainval = pd.concat([train_df, val_df], ignore_index=True)
    trainval_enh, test_enh, _ = add_enhanced_features(trainval, test_df)

    categorical = metadata["categorical_features"]

    # Target encoding (fit on train only)
    train_enc, val_enc = apply_target_encoding(
        train_enh, [val_enh], cols=categorical, target_col=TARGET_COL
    )

    # Target encoding for final test (fit on train+val)
    trainval_enc, test_enc = apply_target_encoding(
        trainval_enh, [test_enh], cols=categorical, target_col=TARGET_COL
    )

    base_numeric = metadata["numeric_features"]
    encoded_cols = [f"{c}__te" for c in categorical]
    numeric_features = base_numeric + added_cols + encoded_cols

    # Ensure columns exist
    numeric_features = [c for c in numeric_features if c in train_enc.columns]
    for df_out in (val_enc, trainval_enc, test_enc):
        for col in numeric_features:
            if col not in df_out.columns:
                df_out[col] = np.nan

    if verbose:
        print(f"Using {len(numeric_features)} numeric features for modeling.")

    return train_enc, val_enc, trainval_enc, test_enc, numeric_features


def _coerce_mlflow_params(params: Dict[str, object]) -> Dict[str, str]:
    safe_params: Dict[str, str] = {}
    for key, value in params.items():
        if isinstance(value, (int, float, str, bool)):
            safe_params[key] = str(value)
        elif value is None:
            safe_params[key] = "None"
        else:
            safe_params[key] = str(value)
    return safe_params


def _log_mlflow_run(
    *,
    output_dir: Path,
    tune_mode: str,
    split_config: SplitConfig,
    metadata: Dict,
    val_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    fitted: Dict[str, object],
    tracking_uri: str | None,
    experiment: str,
    run_name: str | None,
) -> None:
    try:
        import mlflow
    except ImportError:
        print("MLflow is not installed; skipping MLflow logging.")
        return

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        default_uri = (Path(__file__).resolve().parent.parent / "mlruns").as_uri()
        mlflow.set_tracking_uri(default_uri)
    mlflow.set_experiment(experiment)

    run_name = run_name or f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("tune_mode", tune_mode)
        mlflow.log_param("test_rounds", split_config.test_rounds)
        mlflow.log_param("dataset", metadata.get("dataset", ""))
        mlflow.log_param("seasons", metadata.get("seasons", []))
        mlflow.log_param("include_physics", metadata.get("include_physics", ""))
        mlflow.log_param("exclude_lap1", metadata.get("exclude_lap1", ""))

        # Metrics + params per model (nested runs)
        for model_name, estimator in fitted.items():
            with mlflow.start_run(run_name=model_name, nested=True):
                # Log best params when available, else log model params only
                if hasattr(estimator, "best_params_"):
                    params = estimator.best_params_  # type: ignore[assignment]
                elif hasattr(estimator, "get_params"):
                    params = {
                        k: v for k, v in estimator.get_params().items()
                        if k.startswith("model__")
                    }
                else:
                    params = {}
                mlflow.log_params(_coerce_mlflow_params(params))

                row_val = val_metrics[val_metrics["model"] == model_name].iloc[0]
                row_test = test_metrics[test_metrics["model"] == model_name].iloc[0]
                for metric in ("mae", "rmse", "r2"):
                    mlflow.log_metric(f"val_{metric}", float(row_val[metric]))
                    mlflow.log_metric(f"test_{metric}", float(row_test[metric]))

        # Artifacts
        for path in output_dir.glob("*.csv"):
            mlflow.log_artifact(str(path), artifact_path="reports")
        for path in output_dir.glob("*.png"):
            mlflow.log_artifact(str(path), artifact_path="plots")
        for path in output_dir.glob("*.html"):
            mlflow.log_artifact(str(path), artifact_path="plots")


def run_training(
    dataset_path: Path,
    metadata_path: Path,
    *,
    tune_mode: str,
    seed: int,
    split_config: SplitConfig,
    output_dir: Path,
    mlflow_enabled: bool,
    mlflow_uri: str | None,
    mlflow_experiment: str,
    mlflow_run_name: str | None,
) -> None:
    df = pd.read_parquet(dataset_path)
    metadata = _load_metadata(metadata_path)

    train_df, val_df, trainval_df, test_df, numeric_features = _prepare_features(
        df, metadata, split_config=split_config
    )

    # Hyperparameter search on train -> evaluate on val
    base_models = make_model_registry(numeric_features, random_state=seed)
    tuned_models = {
        name: build_search(name, model, random_state=seed, mode=tune_mode)
        for name, model in base_models.items()
    }

    X_train = train_df[numeric_features]
    y_train = train_df[TARGET_COL].to_numpy()
    X_val = val_df[numeric_features]
    y_val = val_df[TARGET_COL].to_numpy()

    val_metrics, val_preds, fitted = evaluate_models(
        tuned_models, X_train, y_train, X_val, y_val, verbose=True
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    val_metrics.to_csv(output_dir / "metrics_val.csv", index=False)

    # Refit best estimators on train+val, evaluate on test
    final_models: Dict[str, object] = {}
    for name, estimator in fitted.items():
        if hasattr(estimator, "best_estimator_"):
            final_models[name] = estimator.best_estimator_
        else:
            final_models[name] = estimator

    X_trainval = trainval_df[numeric_features]
    y_trainval = trainval_df[TARGET_COL].to_numpy()
    X_test = test_df[numeric_features]
    y_test = test_df[TARGET_COL].to_numpy()

    test_metrics, test_preds, _ = evaluate_models(
        final_models, X_trainval, y_trainval, X_test, y_test, verbose=True
    )
    test_metrics.to_csv(output_dir / "metrics_test.csv", index=False)

    # Plots for best model on test set
    best_model_name = test_metrics.sort_values("mae").iloc[0]["model"]
    best_pred = test_preds[best_model_name]

    fig = plot_model_comparison(test_metrics)
    save_plot(fig, output_dir / "model_comparison")

    fig = plot_actual_vs_pred(y_test, best_pred, title=f"{best_model_name}: Predicted vs Actual")
    save_plot(fig, output_dir / "pred_vs_actual")

    fig = plot_error_distribution(y_test, best_pred, title=f"{best_model_name}: Residuals")
    save_plot(fig, output_dir / "error_distribution")

    if "Circuit" in test_df.columns:
        fig = plot_error_by_category(
            test_df, y_test, best_pred, category_col="Circuit", title="MAE by Circuit"
        )
        save_plot(fig, output_dir / "error_by_circuit")

    if mlflow_enabled:
        _log_mlflow_run(
            output_dir=output_dir,
            tune_mode=tune_mode,
            split_config=split_config,
            metadata=metadata,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            fitted=fitted,
            tracking_uri=mlflow_uri,
            experiment=mlflow_experiment,
            run_name=mlflow_run_name,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="F1 lap time pipeline (build + train).")
    parser.add_argument("--build-data", action="store_true", help="Build base dataset.")
    parser.add_argument("--run", action="store_true", help="Run training/evaluation.")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA))
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    parser.add_argument("--exclude-lap1", action="store_true")
    parser.add_argument("--tune", type=str, default="fast", choices=["off", "fast", "full"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-rounds", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow tracking.")
    parser.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI.")
    parser.add_argument("--mlflow-exp", type=str, default="f1-laptime", help="MLflow experiment name.")
    parser.add_argument("--mlflow-run-name", type=str, default=None, help="MLflow run name override.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    metadata_path = Path(args.metadata)

    if args.build_data:
        build_and_save_dataset(
            args.years,
            output_path=dataset_path,
            metadata_path=metadata_path,
            exclude_lap1=args.exclude_lap1,
            verbose=True,
        )

    if args.run or not args.build_data:
        if not dataset_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "Dataset or metadata not found. Run with --build-data first."
            )

        split_config = SplitConfig(test_rounds=args.test_rounds)
        run_training(
            dataset_path,
            metadata_path,
            tune_mode=args.tune,
            seed=args.seed,
            split_config=split_config,
            output_dir=Path(args.output_dir),
            mlflow_enabled=args.mlflow,
            mlflow_uri=args.mlflow_uri,
            mlflow_experiment=args.mlflow_exp,
            mlflow_run_name=args.mlflow_run_name,
        )


if __name__ == "__main__":
    main()
