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
- `data/processed/laps_base.csv` (same dataset as CSV)
- `data/processed/laps_base.metadata.json` (feature lists, filters, seasons)

You can switch dataset file types as you prefer:
- **Notebooks**: edit `DATA_FORMAT` in `notebooks/_common.py` (`"parquet"` or `"csv"`).
- **CLI**: pass `--dataset data/processed/laps_base.csv` to use the CSV file.

## Dataset (what is loaded and what happens)

**FastF1 implementation notes**
- Loading: `fastf1.get_session(year, round, "R")` and `session.load(laps=True, weather=True)`.
- Caching: `fastf1.Cache` is always enabled before loading (see `src.data.enable_cache()`).
- Merge: Weather data (`TrackTemp`, `AirTemp`, etc.) is joined onto laps using `LapStartTime`.

**Pipeline steps**
1. **Load** race laps for each season (2022-2025 by default).
2. **Clean**: drop inaccurate laps, pit in/out laps, SC/VSC/Red Flag laps, deleted laps, formation laps, and optional Lap 1; remove high-side outliers via robust MAD z-score.
3. **Feature engineering**:
   - Lags and rolling mean (`LapTimeLag1/2/3`, `RollingMean3`)
   - Physics-inspired features (fuel load, tire degradation, track evolution)
   - Tire age category buckets
4. **Categorical handling**:
   - Valid tire compounds only (SOFT/MEDIUM/HARD/INTERMEDIATE/WET); invalid/unknown rows are dropped.
   - Rare categories can be collapsed for `TrackStatusFlag` and `TireAgeCategory` only.
5. **Metadata**: saved alongside the dataset with feature lists and build parameters.

## Model-specific pipelines (data handling)

We keep one shared dataset, but use **model-specific preprocessing pipelines** so each model gets the right treatment:

- **Linear/Ridge (LinReg)**: median imputation + standardization + polynomial interactions
  - Pipeline: `SimpleImputer(median) -> StandardScaler -> PolynomialFeatures(degree=2, interaction_only=True)`
- **MLP**: median imputation + standardization
  - Pipeline: `SimpleImputer(median) -> StandardScaler`
- **XGBoost**: median imputation only (no scaling)
  - Pipeline: `SimpleImputer(median)`

This keeps the comparison fair (same dataset) while giving each model appropriate preprocessing.

## Base model parameters (SMALL_MODE = off)

These are the defaults used when SMALL_MODE is **off** (i.e., no notebook overrides):

**Linear / Ridge**
- `alpha = 1.0` (default in `make_linear_model`)

**XGBoost (`XGBRegressor`)**
- `objective = "reg:squarederror"`
- `n_estimators = 600`
- `max_depth = 6`
- `learning_rate = 0.05`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `min_child_weight = 3`
- `reg_alpha = 0.1`
- `reg_lambda = 2.0`
- `tree_method = "hist"`
- `n_jobs = -1`
- `random_state = 42`

**Deep MLP (`TorchMLPRegressor`)**
- `hidden_layers = (256, 128, 64, 32)`
- `dropout = 0.3`
- `batch_norm = True`
- `lr = 1e-3`
- `epochs = 150`
- `batch_size = 256`
- `patience = 20`
- `min_delta = 1e-4`
- `weight_decay = 1e-4`
- `lr_scheduler = True`
- `validation_fraction = 0.1`
- `random_state = 42`

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

- **Train**: 2022-2023
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
1. `notebooks/00_setup.ipynb` - Environment setup
2. `notebooks/01_dataset_overview.ipynb` - EDA, feature exploration, correlation analysis
3. `notebooks/02_model_linear.ipynb` - Linear/Ridge model training
4. `notebooks/03_model_xgboost.ipynb` - XGBoost training with early stopping
5. `notebooks/04_model_mlp.ipynb` - Deep MLP (PyTorch) training
6. `notebooks/05_model_comparison.ipynb` - Basic model comparison
7. `notebooks/06_model_comparison_full.ipynb` - **Full analysis with all metrics & plots for Seminararbeit**
8. `notebooks/07_scientific_validation_feature_interpretability.ipynb` - Defense document for examiner questions

These notebooks use Plotly and share helpers from `notebooks/_common.py`.

## Reports structure (MLOps best practices)

```
reports/
├── models/                          # Trained model artifacts (.joblib)
│   ├── linear.joblib
│   ├── xgboost.joblib
│   └── deep_mlp.joblib
└── notebooks/
    └── 06_model_comparison_full/    # Main output directory for paper
        ├── metrics_comprehensive_2025.csv   # All numerical metrics
        ├── predictions_2025.parquet         # Raw predictions
        ├── model_comparison_mae_rmse.png    # Model comparison plots
        ├── cumulative_error_distribution.png
        ├── feature_importance_xgboost.png
        ├── accuracy_thresholds.png
        ├── mae_by_round_2025.png
        ├── error_vs_lap_number.png
        └── ... (20+ plots)
```

All plots are saved as both `.html` (interactive) and `.png` (static for paper).
