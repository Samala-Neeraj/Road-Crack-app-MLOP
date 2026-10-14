$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $projectRoot ".venv-1\Scripts\python.exe"
$appPath = Join-Path $projectRoot "app.py"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project Python was not found at $pythonPath."
}

Write-Host "Streamlit app: http://127.0.0.1:8501"
Write-Host "Prometheus metrics: http://127.0.0.1:8000/metrics"
& $pythonPath -m streamlit run $appPath --server.address 127.0.0.1 --server.port 8501
if ($LASTEXITCODE -ne 0) {
    throw "Streamlit exited with code $LASTEXITCODE."
}
