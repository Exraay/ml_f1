$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root ".venv\\Scripts\\python.exe"
$mlruns = Join-Path $root "mlruns"
$mlrunsUri = (Resolve-Path $mlruns).Path.Replace("\", "/")
$mlrunsUri = "file:///$mlrunsUri"

if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual env not found at $venvPython. Create it first (python -m venv .venv)."
    exit 1
}

Write-Host "Starting MLflow UI..."
& $venvPython -m mlflow ui --backend-store-uri $mlrunsUri --registry-store-uri $mlrunsUri --default-artifact-root $mlrunsUri --host 127.0.0.1 --port 5000
