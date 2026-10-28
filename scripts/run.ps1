param([string]$Config)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Virtual environment missing. Run .\scripts\setup.ps1 first." }

$env:HF_HOME = if ($env:HF_HOME) { $env:HF_HOME } else { Join-Path $HOME ".cache\huggingface" }
New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null
Push-Location $Root
try {
    $ServerArgs = @()
    if ($Config) { $ServerArgs = @("--config", $Config) }
    & $Python llm_server.py @ServerArgs
    $Status = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $Status
