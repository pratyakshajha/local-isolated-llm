param(
    [Parameter(Position = 0)]
    [ValidateSet("build", "run")]
    [string]$Action,
    [switch]$Gpu
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Image = "wonderdesk:local"

if ($Action -eq "build") {
    docker build --tag $Image $Root
    exit $LASTEXITCODE
}
if ($Action -ne "run") {
    Write-Host "Usage: .\scripts\docker.ps1 build|run [-Gpu]"
    exit 2
}

$DockerArgs = @(
    "run", "--rm", "--name", "wonderdesk",
    "--publish", "127.0.0.1:8080:8080",
    "--volume", "wonderdesk-model-cache:/home/llm/.cache/huggingface"
)
if ($Gpu) { $DockerArgs += @("--gpus", "all") }
$HFHome = if ($env:HF_HOME) { $env:HF_HOME } else { Join-Path $HOME ".cache\huggingface" }
$TokenFile = Join-Path $HFHome "token"
if (Test-Path $TokenFile) {
    $DockerArgs += @(
        "--mount", "type=bind,source=$TokenFile,target=/run/secrets/hf_token,readonly",
        "--env", "HF_TOKEN_PATH=/run/secrets/hf_token"
    )
}
$DockerArgs += $Image
docker @DockerArgs
exit $LASTEXITCODE
