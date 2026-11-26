$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher (py.exe) was not found. Install Python 3.12 x64 first."
}

py -3.12 --version
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12 was not found. Install Python 3.12 x64 and rerun this script."
}

if (-not (Test-Path ".venv")) {
    py -3.12 -m venv .venv
}
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }

& $python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "Failed to install CUDA-enabled PyTorch." }

& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Failed to install experiment dependencies." }

& $python -m pip install git+https://github.com/facebookresearch/sam3.git
if ($LASTEXITCODE -ne 0) {
    throw "SAM 3 installation failed. Check the official SAM 3 install requirements and error above."
}

& $python -m pip install -e .
if ($LASTEXITCODE -ne 0) { throw "Failed to install Inlay experiment." }

Write-Host ""
Write-Host "Python/PyTorch environment installed."
Write-Host "Authenticate with: .venv\Scripts\hf.exe auth login"
Write-Host "Run with: .venv\Scripts\inlay-sam3.exe <image-path>"
