$ErrorActionPreference = "Stop"

$experimentRoot = $PSScriptRoot
$runtimeRoot = Join-Path $experimentRoot ".deps\sam3-ggml"
$buildRoot = Join-Path $runtimeRoot "build-inlay"
$repositoryUrl = "https://github.com/Asher-1/sam3-ggml.git"

function Require-Command([string]$Name, [string]$InstallHint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found. $InstallHint"
    }
}

Require-Command "git" "Install Git for Windows."
Require-Command "cmake" "Install CMake 3.24 or newer and add it to PATH."
$cmakeVersionText = (& cmake --version | Select-Object -First 1)
if ($cmakeVersionText -notmatch "cmake version (\d+)\.(\d+)") {
    throw "Could not determine the installed CMake version."
}
$cmakeVersion = [version]::new([int]$Matches[1], [int]$Matches[2])
if ($cmakeVersion -lt [version]::new(3, 24)) {
    throw "CMake 3.24 or newer is required. Found $cmakeVersionText."
}
Require-Command "nvcc" "Install the NVIDIA CUDA Toolkit 12.6 or newer and add nvcc to PATH."

$gitCommand = Get-Command git
$gitDirectory = Split-Path -Parent $gitCommand.Source
$gitBin = Join-Path (Split-Path -Parent $gitDirectory) "bin"
$gitBash = Join-Path $gitBin "bash.exe"
if (Test-Path $gitBash) {
    $env:PATH = "$gitBin;$env:PATH"
}
Require-Command "bash" "Git Bash is required by the upstream CMake patch step. Reinstall Git for Windows with Git Bash."

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "Visual Studio 2022 C++ Build Tools were not found. Install Visual Studio 2022 Build Tools with the 'Desktop development with C++' workload."
}
$vsInstall = & $vswhere -latest -products "*" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$vsVersion = & $vswhere -latest -products "*" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion
if ($LASTEXITCODE -ne 0 -or -not $vsInstall -or -not $vsVersion) {
    throw "MSVC C++ build tools were not found. Install Visual Studio 2022 or 2026 with the 'Desktop development with C++' workload."
}

$vsMajor = [int](($vsVersion -split "\.")[0])
$generator = switch ($vsMajor) {
    18 { "Visual Studio 18 2026" }
    17 { "Visual Studio 17 2022" }
    default { throw "Unsupported Visual Studio version $vsVersion. Install Visual Studio 2022 or 2026." }
}
Write-Host "Visual Studio: $vsVersion ($generator)"

$cudaVersion = (& nvcc --version | Select-String "release").ToString()
Write-Host "CUDA compiler: $cudaVersion"
Write-Host "Target GPU architecture: 86 (RTX 3060)"

if (-not (Test-Path (Join-Path $runtimeRoot ".git"))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $runtimeRoot) | Out-Null
    Write-Host "Cloning native SAM 3 runtime..."
    & git -c core.autocrlf=false clone --recurse-submodules $repositoryUrl $runtimeRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone the SAM 3 C++ runtime." }
}
else {
    Write-Host "Updating native SAM 3 runtime..."
    & git -C $runtimeRoot fetch origin main
    if ($LASTEXITCODE -ne 0) { throw "Failed to fetch the SAM 3 C++ runtime." }
    & git -C $runtimeRoot checkout main
    if ($LASTEXITCODE -ne 0) { throw "Failed to select the runtime main branch." }
    & git -C $runtimeRoot pull --ff-only origin main
    if ($LASTEXITCODE -ne 0) { throw "Failed to update the SAM 3 C++ runtime." }
}

& git -C $runtimeRoot submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { throw "Failed to initialize the ggml submodule." }

# The upstream patch series is generated against LF Git blobs. Windows
# autocrlf conversion changes the checked-out text and makes git apply fail.
# Disable conversion in both repositories and rewrite tracked files from
# their indexes before CMake applies the patches.
& git -C $runtimeRoot config core.autocrlf false
if ($LASTEXITCODE -ne 0) { throw "Could not disable line-ending conversion in the native runtime checkout." }
& git -C $runtimeRoot checkout-index --all --force
if ($LASTEXITCODE -ne 0) { throw "Could not normalize the native runtime working tree." }

$ggmlRoot = Join-Path $runtimeRoot "ggml"

# The SAM 3 fork's patch series is rebased against ggml v0.21.0.
# Its moving submodule pointer can advance independently, making those
# patches fail against newer ggml sources. Pin the verified compatible commit.
& git -C $ggmlRoot fetch origin 8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b
if ($LASTEXITCODE -ne 0) { throw "Failed to fetch the ggml v0.21.0 revision required by the upstream SAM 3 patches." }
& git -C $ggmlRoot checkout --detach 8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b
if ($LASTEXITCODE -ne 0) { throw "Could not select the ggml v0.21.0 revision required by the upstream SAM 3 patches." }

& git -C $ggmlRoot config core.autocrlf false
if ($LASTEXITCODE -ne 0) { throw "Could not disable line-ending conversion in the ggml checkout." }
& git -C $ggmlRoot checkout-index --all --force
if ($LASTEXITCODE -ne 0) { throw "Could not normalize the ggml working tree." }

Write-Host "Configuring native CUDA build..."
& cmake -S $runtimeRoot -B $buildRoot -G $generator -A x64 -DSAM3_CUDA=ON -DSAM3_METAL=OFF -DSAM3_BUILD_EXAMPLES=ON -DCMAKE_CUDA_ARCHITECTURES=86
if ($LASTEXITCODE -ne 0) {
    throw "CMake configuration failed. Check that CUDA Toolkit, Visual Studio 2022 C++ tools, CMake, and Git Bash are installed."
}

Write-Host "Building sam3_vis (no GUI or Python runtime required)..."
& cmake --build $buildRoot --config Release --target sam3_vis --parallel
if ($LASTEXITCODE -ne 0) { throw "The native SAM 3 build failed." }

$executable = Get-ChildItem -Path (Join-Path $buildRoot "examples") -Filter "sam3_vis.exe" -File -Recurse |
    Select-Object -First 1
if (-not $executable) {
    throw "Build completed but sam3_vis.exe was not found under $buildRoot."
}

Write-Host ""
Write-Host "Native GGUF runtime is ready."
Write-Host "Executable: $($executable.FullName)"
Write-Host "Model folder: $(Join-Path $experimentRoot 'models')"
Write-Host "Run: .\run-gguf.ps1 -Image C:\path\to\screenshot.png"
Write-Host "No Python, PyTorch, torchvision, or Hugging Face login is used by this path."
