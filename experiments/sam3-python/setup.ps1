$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher (py.exe) was not found. Install 64-bit Python 3.12."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git was not found. Install Git for Windows; pip uses it to install Meta's official SAM 3 package."
}

py -3.12 --version
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.12 was not found. Install 64-bit Python 3.12 and rerun this script."
}

if (-not (Test-Path ".venv")) {
    py -3.12 -m venv .venv
}
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "The experiment virtual environment could not be created."
}

Write-Host ""
Write-Host "Installing PyTorch with its bundled CUDA 12.8 runtime..."
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }

& $python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "Failed to install CUDA-enabled PyTorch." }

Write-Host ""
Write-Host "Installing the experiment and Meta's official SAM 3 package..."
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Failed to install experiment dependencies." }

& $python -m pip install git+https://github.com/facebookresearch/sam3.git
if ($LASTEXITCODE -ne 0) {
    throw "Official SAM 3 installation failed. Check the Python, pip, and Git errors above."
}

& $python -m pip install -e .
if ($LASTEXITCODE -ne 0) { throw "Failed to install the Inlay SAM 3 command." }

Write-Host ""
& $python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
if ($LASTEXITCODE -ne 0) { throw "The PyTorch installation check failed." }

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next: .venv\Scripts\hf.exe auth login"
Write-Host "Then: .venv\Scripts\inlay-sam3.exe <image-path>"
Write-Host ""
Write-Host "No CUDA Toolkit, CMake, Visual Studio C++ tools, GGUF model, or native build is needed."
