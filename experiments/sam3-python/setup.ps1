$ErrorActionPreference = "Stop"

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
    throw "The experiment virtual environment could not be created."
}

Write-Host ""
Write-Host "Installing PyTorch with its bundled CUDA 12.8 runtime..."
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }

& $python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw "Failed to install CUDA-enabled PyTorch." }

Write-Host ""
Write-Host "Installing the experiment dependencies..."
$requirements = Join-Path $PSScriptRoot "requirements.txt"
& $python -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Failed to install experiment dependencies." }

Write-Host ""
Write-Host "Checking the local Python build tools..."
& $python -m pip install wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to install the wheel build tool." }

& $python -c "import setuptools, wheel; from packaging.version import Version; print('setuptools:', setuptools.__version__); print('wheel:', wheel.__version__); assert Version(setuptools.__version__) >= Version('68'), 'setuptools 68 or newer is required.'"
if ($LASTEXITCODE -ne 0) {
    throw "The environment needs setuptools 68 or newer and wheel. Check the build-tool installation above."
}

Write-Host ""
Write-Host "Downloading Meta's official SAM 3 source archive..."
# Pin to the current official main revision so setup does not need a Git clone.
$sam3Commit = "31cb7e624e4fe95301b1be1d4e809111593d2e45"
$sam3ArchiveUrl = "https://codeload.github.com/facebookresearch/sam3/zip/$sam3Commit"
$sam3ArchivePath = Join-Path ([IO.Path]::GetTempPath()) "sam3-$sam3Commit.zip"
$downloaded = $false
$downloadError = ""

for ($attempt = 1; $attempt -le 3; $attempt++) {
    try {
        Invoke-WebRequest -Uri $sam3ArchiveUrl -OutFile $sam3ArchivePath -TimeoutSec 300
        $archive = Get-Item -LiteralPath $sam3ArchivePath
        if ($archive.Length -lt 100KB) {
            throw "The downloaded SAM 3 source archive is unexpectedly small ($($archive.Length) bytes)."
        }
        $downloaded = $true
        break
    }
    catch {
        $downloadError = $_.Exception.Message
        Remove-Item -LiteralPath $sam3ArchivePath -Force -ErrorAction SilentlyContinue
        if ($attempt -lt 3) {
            Write-Warning "SAM 3 source download attempt $attempt failed: $downloadError"
            Start-Sleep -Seconds (2 * $attempt)
        }
    }
}

if (-not $downloaded) {
    throw "Could not download Meta's official SAM 3 source archive from $sam3ArchiveUrl after 3 attempts. Check access to codeload.github.com. Details: $downloadError"
}

Write-Host ""
Write-Host "Installing Meta's official SAM 3 package from the downloaded source archive..."
Write-Host "Build isolation is disabled so pip reuses the build tools already installed above."
try {
    & $python -m pip install --no-build-isolation $sam3ArchivePath
    if ($LASTEXITCODE -ne 0) {
        throw "Official SAM 3 installation failed. Check access to the Python package index for its runtime dependencies."
    }
}
finally {
    Remove-Item -LiteralPath $sam3ArchivePath -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Installing the Inlay command..."
& $python -m pip install --no-build-isolation -e $PSScriptRoot
if ($LASTEXITCODE -ne 0) { throw "Failed to install the Inlay SAM 3 command." }

Write-Host ""
Write-Host "Checking the SAM 3 support runtimes..."
& $python -c "import psutil, triton; print('Triton:', triton.__version__); print('psutil:', psutil.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "A required SAM 3 runtime dependency is missing or could not be loaded."
}

Write-Host ""
Write-Host "Checking the SAM 3 inference imports..."
& $python -c "from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf; from sam3.model.sam3_image_processor import Sam3Processor; print('SAM 3 inference imports: OK')"
if ($LASTEXITCODE -ne 0) {
    throw "SAM 3 was installed, but its inference imports failed. Review the missing dependency or import error above."
}

Write-Host ""
& $python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
if ($LASTEXITCODE -ne 0) { throw "The PyTorch installation check failed." }

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next: .venv\Scripts\hf.exe auth login"
Write-Host "Then: .venv\Scripts\inlay-sam3.exe <image-path>"
Write-Host ""
Write-Host "No Git, CUDA Toolkit, CMake, Visual Studio C++ tools, GGUF model, or native build is needed."
