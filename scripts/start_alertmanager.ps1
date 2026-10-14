param(
    [Parameter(Mandatory = $true)]
    [string]$AlertmanagerExe
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$envFile = Join-Path $projectRoot ".env"
$templatePath = Join-Path $projectRoot "monitoring\alertmanager.yml"
$generatedConfig = Join-Path $projectRoot "monitoring\alertmanager.generated.yml"
$alertmanagerPath = (Resolve-Path -LiteralPath $AlertmanagerExe).Path

if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
    throw "Missing .env file at $envFile. Copy .env.example to .env and set Alertmanager email values."
}

$requiredVariables = @(
    "ALERTMANAGER_SMTP_SMARTHOST",
    "ALERTMANAGER_SMTP_FROM",
    "ALERTMANAGER_SMTP_USERNAME",
    "ALERTMANAGER_SMTP_PASSWORD",
    "ALERTMANAGER_EMAIL_TO"
)

foreach ($line in Get-Content -LiteralPath $envFile) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$' -and
        $Matches[1] -like "ALERTMANAGER_*") {
        $value = $Matches[2].Trim()
        if ($value.Length -ge 2 -and
            (($value.StartsWith('"') -and $value.EndsWith('"')) -or
             ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        [Environment]::SetEnvironmentVariable($Matches[1], $value, "Process")
    }
}

foreach ($name in $requiredVariables) {
    $configuredValue = [Environment]::GetEnvironmentVariable($name)
    if ([string]::IsNullOrWhiteSpace($configuredValue)) {
        throw "Set $name in .env before starting Alertmanager."
    }
    if (
        $configuredValue -match '(?i)replace-with|example\.com|smtp\.example\.com'
    ) {
        throw "Replace the example placeholder for $name in .env before starting Alertmanager."
    }
}

$configText = Get-Content -LiteralPath $templatePath -Raw
foreach ($name in $requiredVariables) {
    $value = [Environment]::GetEnvironmentVariable($name)
    $yamlScalar = ConvertTo-Json -InputObject $value -Compress
    $placeholder = '${' + $name + '}'
    $configText = $configText.Replace($placeholder, $yamlScalar)
}
if ($configText -match '\$\{ALERTMANAGER_[A-Z0-9_]+\}') {
    throw "The generated Alertmanager config still contains an unresolved environment placeholder."
}

Set-Content -LiteralPath $generatedConfig -Value $configText -Encoding utf8
Write-Host "Using generated config: $generatedConfig"
Write-Host "Alertmanager UI: http://127.0.0.1:9093"
& $alertmanagerPath `
    "--config.file=$generatedConfig" `
    "--web.listen-address=127.0.0.1:9093"
if ($LASTEXITCODE -ne 0) {
    throw "Alertmanager exited with code $LASTEXITCODE."
}
