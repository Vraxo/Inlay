$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher (py.exe) was not found. Install 64-bit Python 3.12."
}
py -3.12 --version
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12 was not found. Install 64-bit Python 3.12 and rerun this script."
}
$environment = Join-Path $PSScriptRoot ".venv"
if (-not (Test-Path $environment)) {
    py -3.12 -m venv $environment
}
$python = Join-Path $environment "Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "The Qwen 3 VL experiment virtual environment could not be created."
}
Write-Host ""
Write-Host "Installing the lightweight experiment dependencies..."
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
& $python -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install the experiment dependencies." }
Write-Host ""
Write-Host "Checking the Python runtime..."
& $python -c "import requests; import PIL; from inlay_qwen3vl.client import LMStudioVisionClient; from inlay_qwen3vl.runner import run_experiment; print('Requests:', requests.__version__); print('Pillow:', PIL.__version__); print('Qwen 3 VL experiment imports: OK')"
if ($LASTEXITCODE -ne 0) { throw "The Qwen 3 VL experiment import check failed." }
Pop-Location
Write-Host ""
Write-Host "Setup complete."
Write-Host "Run: .venv\Scripts\python.exe -m inlay_qwen3vl <image-path>"
Write-Host "This experiment uses the Qwen model already served by LM Studio; it does not download model weights."
