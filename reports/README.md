# Reports

This folder stores generated plots and evaluation artifacts.

Structure:
- `reports/notebooks/<notebook_name>/` -> plots created from notebooks
- `reports/cli/` -> plots and metrics created by `python -m src.train`
- `reports/archive/YYYY-MM-DD/` -> previous report outputs

Notes:
- Plotly figures are saved as `.html` (interactive) and `.png` (static).
- If PNG export fails, install `kaleido`.
