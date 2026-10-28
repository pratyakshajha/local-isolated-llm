param(
    [string]$Config,
    [switch]$Plain
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Virtual environment missing. Run .\scripts\setup.ps1 first." }

$env:HF_HOME = if ($env:HF_HOME) { $env:HF_HOME } else { Join-Path $HOME ".cache\huggingface" }
New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null
Push-Location $Root
try {
    $CliArgs = @()
    if ($Config) { $CliArgs += @("--config", $Config) }
    if ($Plain) { $CliArgs += "--plain" }
    & $Python run_llm.py @CliArgs
    $Status = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $Status
