$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv-1\Scripts\python.exe"
$PidFile = Join-Path $ProjectRoot "mlflow-server.pid"
$StdoutLog = Join-Path $ProjectRoot "mlflow-server.log"
$StderrLog = Join-Path $ProjectRoot "mlflow-server-error.log"
$BaseUrl = "http://127.0.0.1:5000"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project venv Python was not found at '$Python'. Create the venv first."
}

if (Test-Path -LiteralPath $PidFile -PathType Leaf) {
    $ExistingPidText = (Get-Content -LiteralPath $PidFile -Raw).Trim()
    $ExistingPid = 0
    if ([int]::TryParse($ExistingPidText, [ref]$ExistingPid)) {
        $ExistingProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $ExistingPid" -ErrorAction SilentlyContinue
        if ($null -ne $ExistingProcess -and $ExistingProcess.CommandLine -match "mlflow\s+server") {
            try {
                $Health = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 2
                Write-Output "MLflow is already running at $BaseUrl (PID $ExistingPid)."
                exit 0
            }
            catch {
                throw "An MLflow server process (PID $ExistingPid) exists but its health check failed. Check '$StderrLog' before restarting it."
            }
        }
    }
    Remove-Item -LiteralPath $PidFile -Force
}

$PortListener = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue
if ($PortListener) {
    $ProcessIds = ($PortListener | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
    throw "Port 5000 is already in use by PID(s) $ProcessIds. Stop that service or choose another port."
}

New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "mlartifacts") | Out-Null

$Arguments = @(
    "-m", "mlflow", "server",
    "--backend-store-uri", "sqlite:///mlflow.db",
    "--default-artifact-root", "./mlartifacts",
    "--host", "127.0.0.1",
    "--port", "5000",
    "--workers", "1"
)

$ServerProcess = Start-Process `
    -FilePath $Python `
    -ArgumentList $Arguments `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -PassThru

Set-Content -LiteralPath $PidFile -Value $ServerProcess.Id -NoNewline

for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
    Start-Sleep -Seconds 1
    $ServerProcess.Refresh()
    if ($ServerProcess.HasExited) {
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        throw "MLflow exited during startup (exit code $($ServerProcess.ExitCode)). Read '$StderrLog'."
    }

    try {
        $null = Invoke-RestMethod -Uri "$BaseUrl/health" -TimeoutSec 2
        Write-Output "MLflow is healthy at $BaseUrl (PID $($ServerProcess.Id))."
        Write-Output "Tracking UI: $BaseUrl"
        Write-Output "Standard output: $StdoutLog"
        Write-Output "Errors: $StderrLog"
        exit 0
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}

throw "MLflow did not become healthy at $BaseUrl. Read '$StderrLog' and '$StdoutLog'."
