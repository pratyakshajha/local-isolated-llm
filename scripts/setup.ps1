$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $SystemPython = Get-Command python -ErrorAction Stop
    & $SystemPython.Source -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment." }
}

$env:HF_HOME = if ($env:HF_HOME) { $env:HF_HOME } else { Join-Path $HOME ".cache\huggingface" }
New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null

Push-Location $Root
try {
    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }
    & $Python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Could not install requirements.txt." }
    & $Python -m pip install -e . --no-deps
    if ($LASTEXITCODE -ne 0) { throw "Could not install Wonderdesk." }
    & $Python -m pip install httpx2==2.16.0 ruff==0.12.12
    if ($LASTEXITCODE -ne 0) { throw "Could not install test and lint tools." }
    & $Python -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Dependency conflicts found. Use a clean .venv or resolve the conflicts shown above." }

    Push-Location (Join-Path $Root "frontend")
    try {
        & npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw "npm ci failed." }
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
    }
    finally {
        Pop-Location
    }
}
finally {
    Pop-Location
}

Write-Host "Setup complete. Run .\scripts\run.ps1 to start Wonderdesk."
