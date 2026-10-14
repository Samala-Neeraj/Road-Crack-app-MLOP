$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PidFile = Join-Path $ProjectRoot "mlflow-server.pid"
$PythonPath = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot ".venv-1\Scripts\python.exe"))
$PythonPathPattern = [regex]::Escape($PythonPath)

if (Test-Path -LiteralPath $PidFile -PathType Leaf) {
    $PidText = (Get-Content -LiteralPath $PidFile -Raw).Trim()
    $ServerPid = 0
    if (-not [int]::TryParse($PidText, [ref]$ServerPid)) {
        throw "The MLflow PID file '$PidFile' does not contain a valid process ID. Remove it only after checking whether MLflow is running."
    }
    $ProcessInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $ServerPid" -ErrorAction SilentlyContinue
    if ($null -eq $ProcessInfo) {
        Remove-Item -LiteralPath $PidFile -Force
        Write-Output "The recorded MLflow process is already stopped; removed its stale PID file."
        exit 0
    }
    if ($ProcessInfo.CommandLine -notmatch "mlflow\s+server") {
        throw "PID $ServerPid is not identifiable as the MLflow server. Refusing to stop it; inspect '$PidFile' manually."
    }
}
else {
    $Listener = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue
    if (-not $Listener) {
        Write-Output "MLflow is not listening on port 5000; nothing to stop."
        exit 0
    }

    $Processes = @(Get-CimInstance Win32_Process)
    $ListenerProcess = $Processes | Where-Object {
        $_.ProcessId -eq $Listener[0].OwningProcess
    } | Select-Object -First 1
    if (
        $null -eq $ListenerProcess -or
        $ListenerProcess.CommandLine -notmatch $PythonPathPattern -or
        $ListenerProcess.CommandLine -notmatch "uvicorn"
    ) {
        throw "Port 5000 is occupied by a process that is not an identifiable project MLflow worker. Refusing to stop it."
    }

    $CurrentParentId = [int]$ListenerProcess.ParentProcessId
    $ProcessInfo = $null
    while ($CurrentParentId -gt 0) {
        $Parent = $Processes | Where-Object {
            $_.ProcessId -eq $CurrentParentId
        } | Select-Object -First 1
        if ($null -eq $Parent) {
            break
        }
        if (
            $Parent.CommandLine -match $PythonPathPattern -and
            $Parent.CommandLine -match "mlflow\s+server"
        ) {
            $ProcessInfo = $Parent
            break
        }
        $CurrentParentId = [int]$Parent.ParentProcessId
    }
    if ($null -eq $ProcessInfo) {
        throw "Could not identify the project MLflow server that owns port 5000. Refusing to stop an unknown process."
    }

    $ServerPid = [int]$ProcessInfo.ProcessId
    Write-Output "Recovered the project MLflow server PID $ServerPid from the port 5000 process tree."
}

$Processes = @(Get-CimInstance Win32_Process)
$ProcessTree = [System.Collections.Generic.List[object]]::new()
$PendingParentIds = [System.Collections.Generic.List[int]]::new()
$PendingParentIds.Add($ServerPid)

while ($PendingParentIds.Count -gt 0) {
    $ParentId = $PendingParentIds[0]
    $PendingParentIds.RemoveAt(0)

    foreach ($Child in $Processes) {
        if (
            $Child.ParentProcessId -eq $ParentId -and
            (
                (
                    $Child.CommandLine -match $PythonPathPattern -and
                    $Child.CommandLine -match "mlflow\s+server|uvicorn"
                ) -or
                $Child.CommandLine -match "multiprocessing\.spawn.*spawn_main"
            )
        ) {
            $ProcessTree.Add($Child)
            $PendingParentIds.Add([int]$Child.ProcessId)
        }
    }
}

$ProcessesToStop = @($ProcessTree.ToArray()) + @($ProcessInfo)
for ($Index = $ProcessesToStop.Count - 1; $Index -ge 0; $Index--) {
    $ProcessId = [int]$ProcessesToStop[$Index].ProcessId
    $OriginalCommandLine = [string]$ProcessesToStop[$Index].CommandLine
    $CurrentProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if ($null -ne $CurrentProcess -and $CurrentProcess.CommandLine -eq $OriginalCommandLine) {
        Stop-Process -Id $ProcessId
    }

    $StillRunning = $false
    for ($Attempt = 0; $Attempt -lt 30; $Attempt++) {
        $CurrentProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
        $StillRunning = $null -ne $CurrentProcess -and $CurrentProcess.CommandLine -eq $OriginalCommandLine
        if (-not $StillRunning) {
            break
        }
        Start-Sleep -Milliseconds 500
    }

    if ($StillRunning) {
        throw "MLflow process $ProcessId did not stop within 15 seconds."
    }
}

if (Test-Path -LiteralPath $PidFile -PathType Leaf) {
    Remove-Item -LiteralPath $PidFile -Force
}
Write-Output "Stopped MLflow server process tree rooted at PID $ServerPid."
