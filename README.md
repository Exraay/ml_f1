# F1 Lap Time Prediction (FastF1)

Reproducible pipeline to predict F1 lap times using FastF1 data, with an apples-to-apples model comparison between **Linear**, **XGBoost**, and **Deep MLP**.

## Quickstart

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Build the dataset (base, split-safe features)

```bash
python -m src.train --build-data
```

This creates:
- `data/processed/laps_base.parquet` (cleaned + engineered features)
- `data/processed/laps_base.metadata.json` (feature lists, filters, seasons)

### Train + evaluate (with tuning)

```bash
python -m src.train --run --tune fast
```

Options:
- `--tune off` (no search)
- `--tune fast` (RandomizedSearchCV)
- `--tune full` (GridSearchCV, slower)
- `--test-rounds 6` (first N races of 2025 for test)
- `--mlflow` (enable MLflow tracking)

Outputs go to `reports/`:
- `model_comparison.png`
- `pred_vs_actual.png`
- `error_distribution.png`
- `error_by_circuit.png`

## MLflow tracking

MLflow is integrated into `src/train.py` and can be enabled via CLI:

```bash
python -m src.train --run --tune fast --mlflow
```

Optional flags:
- `--mlflow-uri` (set a remote tracking server; default is local `./mlruns`)
- `--mlflow-exp` (experiment name; default: `f1-laptime`)
- `--mlflow-run-name` (override run name)

When enabled, the script logs:
- params (tune mode, split config, dataset metadata, best params when tuning)
- metrics (MAE/RMSE/R2 for val and test)
- artifacts from `reports/` (plots + CSVs)

### MLflow UI (best practice for a shared view)

Use a repo-local tracking store so CLI + notebooks log to the same place:

```bash
powershell -ExecutionPolicy Bypass -File scripts/start_mlflow_ui.ps1
```

This starts MLflow at `http://127.0.0.1:5000` with `mlruns/` as the backend store.
The notebooks now default to `MLFLOW_TRACKING_URI = "<repo>/mlruns" (file:// URI)` so all runs show up in one place.

## Hyperparameter optimization

This repo uses `src.models.build_search()` to wrap each model in a CV search when tuning is enabled.

- **CLI runs (`python -m src.train`)**
  - `--tune off` uses the base model defaults in `src/models.py` (no search).
  - `--tune fast/full` uses the default `param_grid` inside `build_search()` unless you edit it.
  - There is **no YAML config**; adjust defaults or grids directly in code if you want different values.

- **Notebook runs**
  - In `notebooks/04_model_mlp.ipynb`, the first code cell sets `TUNE_MODE` and the MLP variables.
  - **Tuning on**: edit `MLP_PARAM_GRID` to change the search space.
  - **Tuning off**: edit the base model arguments passed into `make_mlp_model(...)` in that same cell
    (e.g., `hidden_layers`, `dropout`, `epochs`, `batch_size`, and any other params you add).
  - In `notebooks/03_model_xgboost.ipynb`, the first code cell sets `TUNE_MODE`.
  - **Tuning on (XGBoost)**: edit the default `param_grid` in `src/models.py` inside `build_search()`,
    or pass a custom `param_grid=...` into `build_search(...)` in the notebook.
  - **Tuning off (XGBoost)**: change values passed to `make_xgboost_model(...)` in that notebook cell
    (e.g., `learning_rate`, `max_depth`, `n_estimators`, etc.). The defaults live in `src/models.py`.

## Data split (apples-to-apples)

- **Train**: 2022–2023
- **Validation**: 2024
- **Test**: first N races of 2025 (default: 6)

## Notes on features

Base dataset includes:
- Lap history features (lags, rolling mean)
- Physics-inspired features (fuel load, tire degradation)
- Track evolution (cumulative field laps)
- Weather (if available in FastF1)

Train-only enhancements (no leakage):
- Driver/Team historical pace aggregates
- Stint-phase signals
- Target encoding for high-cardinality categories

## Future work (documented, not implemented)

- ERS / battery state of charge
- Wind speed and direction
- Traffic / gap to front car

These are not consistently available in FastF1 without external data sources.

## Archived notebooks

Legacy notebooks are preserved under `archive/notebooks/` to keep the repo slim.

## New notebook workflow (clean, focused)

Run in order:
1. `notebooks/00_setup.ipynb`
2. `notebooks/01_dataset_overview.ipynb`
3. `notebooks/02_model_linear.ipynb`
4. `notebooks/03_model_xgboost.ipynb`
5. `notebooks/04_model_mlp.ipynb`
6. `notebooks/05_model_comparison.ipynb`

These notebooks use Plotly and share helpers from `notebooks/_common.py`.
