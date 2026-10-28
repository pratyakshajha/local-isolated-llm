param([string]$Config)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Node = (Get-Command node -ErrorAction Stop).Source
$Vite = Join-Path $Root "frontend\node_modules\vite\bin\vite.js"
if (-not (Test-Path $Python) -or -not (Test-Path $Vite)) {
    throw "Setup is incomplete. Run .\scripts\setup.ps1 first."
}

$env:HF_HOME = if ($env:HF_HOME) { $env:HF_HOME } else { Join-Path $HOME ".cache\huggingface" }
New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null
$Backend = $null
$Frontend = $null
$Status = 0
try {
    $ServerArgs = @("llm_server.py")
    if ($Config) { $ServerArgs += @("--config", $Config) }
    $Backend = Start-Process -FilePath $Python -ArgumentList $ServerArgs -WorkingDirectory $Root -NoNewWindow -PassThru
    $Frontend = Start-Process -FilePath $Node -ArgumentList @($Vite) -WorkingDirectory (Join-Path $Root "frontend") -NoNewWindow -PassThru
    Write-Host "Backend: http://127.0.0.1:8080/  |  Vite: http://127.0.0.1:5173/"

    while (-not $Backend.HasExited -and -not $Frontend.HasExited) {
        Start-Sleep -Seconds 1
    }
    if ($Backend.HasExited) { $Status = $Backend.ExitCode }
    elseif ($Frontend.HasExited) { $Status = $Frontend.ExitCode }
}
finally {
    foreach ($Process in @($Frontend, $Backend)) {
        if ($null -ne $Process -and -not $Process.HasExited) {
            Stop-Process -Id $Process.Id -Force
        }
    }
}
exit $Status
