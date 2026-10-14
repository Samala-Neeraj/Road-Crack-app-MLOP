param(
    [Parameter(Mandatory = $true)]
    [string]$GrafanaServerExe
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$grafanaPath = (Resolve-Path -LiteralPath $GrafanaServerExe).Path
$grafanaBin = Split-Path -Parent $grafanaPath
$grafanaHome = Split-Path -Parent $grafanaBin
$provisioningPath = Join-Path $projectRoot "monitoring\grafana\provisioning"
$dashboardPath = Join-Path $projectRoot "monitoring\grafana\dashboards"
$dataPath = Join-Path $projectRoot "grafana-data"
$logsPath = Join-Path $projectRoot "grafana-logs"

if (-not (Test-Path -LiteralPath (Join-Path $grafanaHome "conf\defaults.ini") -PathType Leaf)) {
    throw "Could not find conf\defaults.ini under inferred Grafana home: $grafanaHome"
}

New-Item -ItemType Directory -Force -Path $dataPath, $logsPath | Out-Null
$env:GF_PATHS_DATA = $dataPath
$env:GF_PATHS_LOGS = $logsPath
$env:GF_PATHS_PROVISIONING = $provisioningPath
$env:ROAD_CRACK_GRAFANA_DASHBOARD_PATH = $dashboardPath

Write-Host "Grafana home: $grafanaHome"
Write-Host "Dashboard provisioned from: $dashboardPath"
Write-Host "Grafana UI: http://127.0.0.1:3000"
& $grafanaPath `
    "--homepath=$grafanaHome" `
    "--config=$grafanaHome\conf\defaults.ini" `
    "cfg:server.http_addr=127.0.0.1" `
    "cfg:server.http_port=3000"
if ($LASTEXITCODE -ne 0) {
    throw "Grafana exited with code $LASTEXITCODE."
}
