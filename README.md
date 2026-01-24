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

Outputs go to `reports/`:
- `model_comparison.png`
- `pred_vs_actual.png`
- `error_distribution.png`
- `error_by_circuit.png`

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
