$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$manifestPath = Join-Path $projectRoot "services.started.json"
$stopMlflowPath = Join-Path $PSScriptRoot "stop_mlflow.ps1"

if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    Write-Host "No services.started.json manifest found. No launcher-managed service terminals will be stopped."
    exit 0
}

$recordedServices = @(Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json)
$errors = [System.Collections.Generic.List[string]]::new()

try {
    & $stopMlflowPath
}
catch {
    $errors.Add("MLflow: $($_.Exception.Message)")
}

$processes = @(Get-CimInstance Win32_Process)
foreach ($service in $recordedServices) {
    $rootPid = [int]$service.RootPid
    $root = $processes | Where-Object { $_.ProcessId -eq $rootPid } | Select-Object -First 1
    if ($null -eq $root) {
        Write-Host "$($service.Name) launcher PID $rootPid is already stopped."
        continue
    }
    if ($root.CommandLine -notlike "*$($service.Script)*") {
        $errors.Add(
            "$($service.Name): PID $rootPid no longer identifies its recorded launcher; refusing to stop it."
        )
        continue
    }

    $tree = [System.Collections.Generic.List[object]]::new()
    $pending = [System.Collections.Generic.List[int]]::new()
    $pending.Add($rootPid)
    while ($pending.Count -gt 0) {
        $parentId = $pending[0]
        $pending.RemoveAt(0)
        foreach ($child in $processes) {
            if ($child.ParentProcessId -eq $parentId) {
                $tree.Add($child)
                $pending.Add([int]$child.ProcessId)
            }
        }
    }
    $tree.Add($root)

    for ($index = $tree.Count - 1; $index -ge 0; $index--) {
        $process = $tree[$index]
        $processId = [int]$process.ProcessId
        $current = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
        if ($null -ne $current -and $current.CommandLine -eq $process.CommandLine) {
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host "Stopped $($service.Name) launcher process tree rooted at PID $rootPid."
}

if ($errors.Count -gt 0) {
    throw ($errors -join [Environment]::NewLine)
}

Remove-Item -LiteralPath $manifestPath -Force
Write-Host "Stopped the launcher-managed services. Verify ports 5000, 8000, 8501, 9090, 9093, and 3000 if any service was already running separately."
