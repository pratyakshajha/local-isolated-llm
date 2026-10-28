$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Virtual environment missing. Run .\scripts\setup.ps1 first." }

Push-Location $Root
try {
    & $Python -m unittest discover -v
    if ($LASTEXITCODE -ne 0) { throw "Backend tests failed." }
    & $Python -m ruff check app tests
    if ($LASTEXITCODE -ne 0) { throw "Python lint failed." }

    Push-Location (Join-Path $Root "frontend")
    try {
        & npm.cmd test
        if ($LASTEXITCODE -ne 0) { throw "Frontend tests failed." }
        & npm.cmd run typecheck
        if ($LASTEXITCODE -ne 0) { throw "Frontend typecheck failed." }
        & npm.cmd run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
        & npm.cmd audit --audit-level=high
        if ($LASTEXITCODE -ne 0) { throw "Frontend dependency audit failed." }
    }
    finally {
        Pop-Location
    }
}
finally {
    Pop-Location
}
