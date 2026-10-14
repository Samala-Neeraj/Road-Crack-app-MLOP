param(
    [Parameter(Mandatory = $true)]
    [string]$PrometheusExe,
    [Parameter(Mandatory = $true)]
    [string]$AlertmanagerExe,
    [Parameter(Mandatory = $true)]
    [string]$GrafanaServerExe
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$scriptRoot = $PSScriptRoot
$manifestPath = Join-Path $projectRoot "services.started.json"
$requiredFiles = @(
    (Join-Path $projectRoot ".env"),
    (Join-Path $projectRoot ".venv-1\Scripts\python.exe"),
    (Join-Path $scriptRoot "start_mlflow.ps1"),
    (Join-Path $scriptRoot "start_prometheus.ps1"),
    (Join-Path $scriptRoot "start_alertmanager.ps1"),
    (Join-Path $scriptRoot "start_grafana.ps1"),
    (Join-Path $scriptRoot "start_streamlit.ps1")
)

foreach ($requiredFile in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required project file is missing: $requiredFile"
    }
}
foreach ($binaryPath in @($PrometheusExe, $AlertmanagerExe, $GrafanaServerExe)) {
    if (-not (Test-Path -LiteralPath $binaryPath -PathType Leaf)) {
        throw "Service executable was not found: $binaryPath"
    }
}
if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
    throw "A services.started.json manifest already exists. Run scripts\stop_all.ps1 before starting another stack."
}

$occupiedPorts = foreach ($port in @(5000, 8000, 8501, 9090, 9093, 3000)) {
    $listener = Get-NetTCPConnection `
        -LocalPort $port `
        -State Listen `
        -ErrorAction SilentlyContinue
    if ($listener) {
        $owners = ($listener | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
        "$port (PID $owners)"
    }
}
if ($occupiedPorts) {
    throw "Cannot start the full stack because these ports are already in use: $($occupiedPorts -join '; '). Stop those services first."
}

$services = @(
    [pscustomobject]@{
        Name = "MLflow"
        Script = Join-Path $scriptRoot "start_mlflow.ps1"
        Extra = @()
    },
    [pscustomobject]@{
        Name = "Prometheus"
        Script = Join-Path $scriptRoot "start_prometheus.ps1"
        Extra = @("-PrometheusExe", (Resolve-Path -LiteralPath $PrometheusExe).Path)
    },
    [pscustomobject]@{
        Name = "Alertmanager"
        Script = Join-Path $scriptRoot "start_alertmanager.ps1"
        Extra = @("-AlertmanagerExe", (Resolve-Path -LiteralPath $AlertmanagerExe).Path)
    },
    [pscustomobject]@{
        Name = "Grafana"
        Script = Join-Path $scriptRoot "start_grafana.ps1"
        Extra = @("-GrafanaServerExe", (Resolve-Path -LiteralPath $GrafanaServerExe).Path)
    },
    [pscustomobject]@{
        Name = "Streamlit"
        Script = Join-Path $scriptRoot "start_streamlit.ps1"
        Extra = @()
    }
)

$started = [System.Collections.Generic.List[object]]::new()
try {
    foreach ($service in $services) {
        $quotedScript = '"' + $service.Script + '"'
        $quotedExtra = @(
            foreach ($argument in $service.Extra) {
                '"' + $argument.Replace('"', '\"') + '"'
            }
        )
        $argumentLine = "-NoExit -ExecutionPolicy Bypass -File $quotedScript"
        if ($quotedExtra.Count -gt 0) {
            $argumentLine += " " + ($quotedExtra -join " ")
        }

        $process = Start-Process `
            -FilePath "powershell.exe" `
            -ArgumentList $argumentLine `
            -WorkingDirectory $projectRoot `
            -PassThru
        $started.Add(
            [pscustomobject]@{
                Name = $service.Name
                RootPid = $process.Id
                Script = $service.Script
            }
        )
        Write-Host "Started $($service.Name) terminal (PID $($process.Id))."
        Start-Sleep -Milliseconds 500
        $process.Refresh()
        if ($process.HasExited) {
            throw "$($service.Name) launcher exited immediately. Check its terminal output."
        }
    }

    $started | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath $manifestPath -Encoding utf8
}
catch {
    foreach ($service in $started) {
        Stop-Process -Id $service.RootPid -Force -ErrorAction SilentlyContinue
    }
    throw
}

Write-Host ""
Write-Host "Service URLs:"
Write-Host "  Streamlit    http://127.0.0.1:8501"
Write-Host "  Metrics      http://127.0.0.1:8000/metrics"
Write-Host "  MLflow       http://127.0.0.1:5000"
Write-Host "  Prometheus   http://127.0.0.1:9090"
Write-Host "  Alertmanager http://127.0.0.1:9093"
Write-Host "  Grafana      http://127.0.0.1:3000"
Write-Host ""
Write-Host "Run scripts\stop_all.ps1 to stop only the service process trees recorded by this launcher."
