param(
  [switch]$SetupOnly
)

$ErrorActionPreference = "Stop"

Write-Output "[DGX] Loading environment from .env if present"
if (Test-Path .env) {
  Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    $parts = $_ -split '=', 2
    if ($parts.Count -eq 2) {
      [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
    }
  }
}

if ([string]::IsNullOrWhiteSpace($env:GPU_HOST) -or [string]::IsNullOrWhiteSpace($env:GPU_SSH_USER)) {
  throw "[DGX] GPU_HOST and GPU_SSH_USER must be set in .env"
}

if (-not $env:GPU_SSH_PORT) { $env:GPU_SSH_PORT = "22" }
if (-not $env:GPU_DEVICE_ID) { $env:GPU_DEVICE_ID = "0" }
if (-not $env:DGX_PROJECT_PATH) { $env:DGX_PROJECT_PATH = "/raid/home/$($env:GPU_SSH_USER)/Contexta-AI" }
if (-not $env:DGX_REPO_URL) { $env:DGX_REPO_URL = "https://github.com/Srivardhan04/Contexta-AI.git" }
if (-not $env:DGX_PYTHON_BIN) { $env:DGX_PYTHON_BIN = "python3" }
if (-not $env:DGX_REMOTE_PORT) { $env:DGX_REMOTE_PORT = "8000" }
if (-not $env:DGX_LOCAL_TUNNEL_PORT) { $env:DGX_LOCAL_TUNNEL_PORT = $env:DGX_REMOTE_PORT }
if (-not $env:DGX_SKIP_PIP_INSTALL) { $env:DGX_SKIP_PIP_INSTALL = "false" }
if (-not $env:DGX_FORCE_TORCH_GPU) { $env:DGX_FORCE_TORCH_GPU = "true" }
if (-not $env:DGX_TORCH_INDEX_URL) { $env:DGX_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121" }
if (-not $env:DGX_TORCH_VERSION) { $env:DGX_TORCH_VERSION = "2.4.1" }

$plink = "C:\Program Files\PuTTY\plink.exe"
$pscp = "C:\Program Files\PuTTY\pscp.exe"
if (-not (Test-Path $plink)) { throw "[DGX] plink was not found. Install PuTTY first." }
if (-not (Test-Path $pscp)) { throw "[DGX] pscp was not found. Install PuTTY first." }

$remoteLogin = "$($env:GPU_SSH_USER)@$($env:GPU_HOST)"
$plinkBase = @("-batch", "-P", $env:GPU_SSH_PORT)

if (-not [string]::IsNullOrWhiteSpace($env:GPU_SSH_HOST_KEY)) {
  $plinkBase += @("-hostkey", $env:GPU_SSH_HOST_KEY)
}
if (-not [string]::IsNullOrWhiteSpace($env:GPU_SSH_PASSWORD)) {
  $plinkBase += @("-pw", $env:GPU_SSH_PASSWORD)
}

Write-Output "[DGX] Ensuring remote project directory exists"
& $plink @plinkBase $remoteLogin "mkdir -p '$($env:DGX_PROJECT_PATH)'"
if ($LASTEXITCODE -ne 0) { throw "[DGX] Failed to create remote project directory" }

if (Test-Path .env) {
  Write-Output "[DGX] Syncing local .env to DGX project directory"
  $pscpArgs = @("-batch", "-P", $env:GPU_SSH_PORT)
  if (-not [string]::IsNullOrWhiteSpace($env:GPU_SSH_HOST_KEY)) {
    $pscpArgs += @("-hostkey", $env:GPU_SSH_HOST_KEY)
  }
  if (-not [string]::IsNullOrWhiteSpace($env:GPU_SSH_PASSWORD)) {
    $pscpArgs += @("-pw", $env:GPU_SSH_PASSWORD)
  }
  $pscpArgs += @(".env", "${remoteLogin}:$($env:DGX_PROJECT_PATH)/.env")
  & $pscp @pscpArgs
  if ($LASTEXITCODE -ne 0) { throw "[DGX] Failed to sync .env to DGX" }
}

function Invoke-DgxCommand {
  param([string]$Command)
  & $plink @plinkBase $remoteLogin $Command
  if ($LASTEXITCODE -ne 0) {
    throw "[DGX] Remote command failed: $Command"
  }
}

Write-Output "[DGX] Bootstrapping Python environment and verifying CUDA on DGX"
Invoke-DgxCommand "if [ ! -d '$($env:DGX_PROJECT_PATH)/.git' ]; then git init '$($env:DGX_PROJECT_PATH)'; fi; if ! git -C '$($env:DGX_PROJECT_PATH)' remote get-url origin >/dev/null 2>&1; then git -C '$($env:DGX_PROJECT_PATH)' remote add origin '$($env:DGX_REPO_URL)'; else git -C '$($env:DGX_PROJECT_PATH)' remote set-url origin '$($env:DGX_REPO_URL)'; fi; git -C '$($env:DGX_PROJECT_PATH)' fetch --depth 1 origin HEAD; git -C '$($env:DGX_PROJECT_PATH)' checkout -f FETCH_HEAD"
Invoke-DgxCommand "$($env:DGX_PYTHON_BIN) -m venv '$($env:DGX_PROJECT_PATH)/.venv'"
if ($env:DGX_SKIP_PIP_INSTALL -ne "true") {
  Invoke-DgxCommand ". '$($env:DGX_PROJECT_PATH)/.venv/bin/activate' && python -m pip install --upgrade pip"
}
if ($env:DGX_FORCE_TORCH_GPU -eq "true") {
  Invoke-DgxCommand ". '$($env:DGX_PROJECT_PATH)/.venv/bin/activate' && python -m pip install --upgrade --index-url '$($env:DGX_TORCH_INDEX_URL)' 'torch==$($env:DGX_TORCH_VERSION)'"
}
if ($env:DGX_SKIP_PIP_INSTALL -ne "true") {
  Invoke-DgxCommand ". '$($env:DGX_PROJECT_PATH)/.venv/bin/activate' && python -m pip install -r '$($env:DGX_PROJECT_PATH)/requirements.txt'"
}
$cudaProbe = 'import torch; print(int(torch.cuda.is_available())); print(torch.cuda.device_count())'
Invoke-DgxCommand ". '$($env:DGX_PROJECT_PATH)/.venv/bin/activate' && export GPU_ENABLED=true GPU_DEVICE_ID='$($env:GPU_DEVICE_ID)' EMBEDDING_DEVICE=auto RERANKER_DEVICE=auto && python -c '$cudaProbe'"

if ($SetupOnly) {
  Write-Output "[DGX] Setup-only mode complete."
  exit 0
}

$tunnelArgs = @("-N", "-L", "$($env:DGX_LOCAL_TUNNEL_PORT):127.0.0.1:$($env:DGX_REMOTE_PORT)")
Write-Output "[DGX] Starting SSH tunnel localhost:$($env:DGX_LOCAL_TUNNEL_PORT) -> DGX:$($env:DGX_REMOTE_PORT)"
Start-Process -FilePath $plink -ArgumentList ($plinkBase + $tunnelArgs + $remoteLogin) -WindowStyle Hidden | Out-Null

$startScript = @"
set -e
cd '$($env:DGX_PROJECT_PATH)'
. .venv/bin/activate
if [ -f .env ]; then
  while IFS= read -r line || [ -n "`$line" ]; do
    case "`$line" in
      ''|\#*) continue ;;
    esac
    key="`${line%%=*}"
    value="`${line#*=}"
    key="`$(printf '%s' "`$key" | tr -d ' \r')"
    value="`${value%`$'\r'}"
    if [ -n "`$key" ]; then
      export "`$key=`$value"
    fi
  done < .env
fi
export GPU_ENABLED=true
export GPU_DEVICE_ID='$($env:GPU_DEVICE_ID)'
export EMBEDDING_DEVICE=auto
export RERANKER_DEVICE=auto
export CUDA_VISIBLE_DEVICES='$($env:GPU_DEVICE_ID)'
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OLLAMA_NUM_GPU='99'
export OLLAMA_GPU_ONLY='true'
export OLLAMA_LLM_LIBRARY='cuda'
export OLLAMA_HOST='127.0.0.1:11435'
pkill -u "`$(whoami)" -f "ollama serve" >/dev/null 2>&1 || true
nohup env OLLAMA_HOST="`$OLLAMA_HOST" OLLAMA_LLM_LIBRARY="`$OLLAMA_LLM_LIBRARY" OLLAMA_NUM_GPU="`$OLLAMA_NUM_GPU" ollama serve >/tmp/contexta_ollama.log 2>&1 &
export OLLAMA_BASE_URL='http://127.0.0.1:11435'
python main.py
"@

$startScript = $startScript -replace "`r`n", "`n"

Write-Output "[DGX] Starting backend on DGX (foreground)."
Write-Output "[DGX] Access API via http://localhost:$($env:DGX_LOCAL_TUNNEL_PORT)"
& $plink @plinkBase $remoteLogin $startScript
