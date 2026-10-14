param(
    [Parameter(Mandatory = $true)]
    [string]$PrometheusExe
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$prometheusPath = (Resolve-Path -LiteralPath $PrometheusExe).Path
$configPath = Join-Path $projectRoot "monitoring\prometheus.yml"
$storagePath = Join-Path $projectRoot "prometheus-data"

if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Prometheus config not found: $configPath"
}
New-Item -ItemType Directory -Force -Path $storagePath | Out-Null

Write-Host "Prometheus UI: http://127.0.0.1:9090"
& $prometheusPath `
    "--config.file=$configPath" `
    "--web.listen-address=127.0.0.1:9090" `
    "--storage.tsdb.path=$storagePath"
if ($LASTEXITCODE -ne 0) {
    throw "Prometheus exited with code $LASTEXITCODE."
}
