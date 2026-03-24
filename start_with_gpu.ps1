$ErrorActionPreference = "Stop"

Write-Output "[GPU] Loading environment from .env if present"
if (Test-Path .env) {
  Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    $parts = $_ -split '=', 2
    if ($parts.Count -eq 2) {
      [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
    }
  }
}

$gpuEnabled = (($env:GPU_ENABLED | ForEach-Object { $_.ToLower() }) -eq "true")
if ($gpuEnabled) {
  if (-not $env:CUDA_VISIBLE_DEVICES) {
    if ($env:GPU_DEVICE_ID) {
      $env:CUDA_VISIBLE_DEVICES = $env:GPU_DEVICE_ID
    } else {
      $env:CUDA_VISIBLE_DEVICES = "0"
    }
  }
  if (-not $env:CUDA_DEVICE_ORDER) {
    $env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"
  }
}

Write-Output "[GPU] Running health check"
& .\.venv\Scripts\python.exe .\gpu_health_check.py

$localCuda = $false
try {
  $cudaResult = & .\.venv\Scripts\python.exe -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2>$null
  $localCuda = (($cudaResult | Select-Object -Last 1).Trim() -eq "1")
} catch {
  $localCuda = $false
}

if ($localCuda) {
  Write-Output "[GPU] Local CUDA detected. Starting backend locally."
  & .\.venv\Scripts\python.exe .\main.py
  exit $LASTEXITCODE
}

$useRemoteGpu = (($env:GPU_ENABLED | ForEach-Object { $_.ToLower() }) -eq "true") -and
                (-not [string]::IsNullOrWhiteSpace($env:GPU_HOST)) -and
                (-not [string]::IsNullOrWhiteSpace($env:GPU_SSH_USER))

if ($useRemoteGpu) {
  Write-Output "[GPU] Local CUDA not available. Switching to DGX remote GPU runtime."
  & .\start_on_dgx.ps1
  exit $LASTEXITCODE
}

Write-Output "[GPU] GPU is not available locally and DGX remote settings are incomplete."
Write-Output "[GPU] Starting backend locally on CPU."
& .\.venv\Scripts\python.exe .\main.py
