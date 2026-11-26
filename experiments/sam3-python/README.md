# Inlay — SAM 3.1 segmentation experiments

A deliberately small experiment to find out whether SAM 3.1 can segment the **entire foreground photo panel** in Inlay screenshots—not merely the person or object inside that photo.

This is isolated from the existing .NET application. The folder name is historical; **the recommended path is now the native GGUF/C++ runtime**.

## 1. Recommended: native GGUF runtime (no PyTorch)

The GGUF route uses [Asher-1/sam3-ggml](https://github.com/Asher-1/sam3-ggml), a native C++/ggml implementation. It loads the .gguf model directly. It does **not** install or run Python, PyTorch, torchvision, or the official Python SAM package.

### Put the model here

Copy your downloaded SAM 3.1 GGUF file into this directory:

```text
experiments/sam3-python/
└── models/
    └── sam3.1-f16.gguf
```

The runner also recognizes sam3.1-q8_0.gguf, sam3-f16.gguf, and sam3-q8_0.gguf. If you have a differently named model, pass its path with -Model. Keep model files local; they are ignored by Git.

### Windows prerequisites

- Git for Windows (including Git Bash)
- CMake 3.24+
- Visual Studio 2022 Build Tools with **Desktop development with C++**
- NVIDIA CUDA Toolkit 12.6+ (nvcc on PATH)

The build targets CUDA architecture 8.6 for the RTX 3060. It builds only the offline visualization CLI, so SDL2 and the optional interactive GUI are not needed. The C++ runtime source and ggml submodule are cloned into the ignored .deps/ directory.

### Build once

From this directory in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

This configures and compiles sam3_vis.exe with CUDA support. It does not download model weights; use the GGUF file you already downloaded.

### Run the panel-detection experiment

```powershell
.\run-gguf.ps1 -Image C:\path\to\screenshot.png
```

By default, the runner tests three descriptions of the complete foreground photo panel. It uses automatic backend selection (CUDA first, with the runtime's supported fallbacks). To force CUDA:

```powershell
.\run-gguf.ps1 -Image C:\path\to\screenshot.png -Device cuda
```

To test a custom prompt:

```powershell
.\run-gguf.ps1 -Image C:\path\to\screenshot.png -Prompt "the complete rectangular photo panel in the foreground"
```

Output is written beside the source image in a directory ending in .sam3-gguf. Each prompt produces a rendered PNG overlay and a log containing the detection count, scores, and boxes. report.json records the prompt, output path, selected device, and process result.

**The overlay is the important evidence:** a mask around the person/object *inside* the photograph is a failure for Inlay's purpose. A detection score is not a calibrated probability and does not prove that the outer photo panel was found.

## 2. Optional: official Python/PyTorch comparison

The original official Meta Python experiment remains available for comparison. This path is separate and **does** install a large CUDA-enabled PyTorch stack.

1. Request access to the official [Meta SAM 3.1 Hugging Face model](https://huggingface.co/facebook/sam3.1), accept the terms, and create a read token.
2. Use 64-bit Python 3.12. From this directory, run:

```powershell
.\setup-python.ps1
.\.venv\Scripts\hf.exe auth login
```

3. Run the Python experiment:

```powershell
.\.venv\Scripts\inlay-sam3.exe C:\path\to\screenshot.png
```

The Python experiment saves masks, transparent masked crops, overlays, raw model scores, bounding rectangles, and inference time. Native Windows support for the official Python project is not explicitly guaranteed; WSL2 is the fallback if a native dependency fails.

## 3. Tests

The lightweight Python geometry tests do not install PyTorch:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m pytest
```

The PowerShell setup and runner scripts are syntax-checked in Windows CI. CI also compiles the native SAM 3 CLI with the Windows MSVC toolchain (CPU backend); the CUDA-enabled build still needs to be validated on a Windows machine with the prerequisites above.

## Important interpretation

- SAM scores are not calibrated probabilities.
- A high score does not prove that the full panel was selected.
- A mask of the person/object *inside* the photograph is a failed result for Inlay's purpose.
- The decisive test is visual: does the mask follow the outer photo panel across varied screenshots?
- No model is assumed to achieve zero errors until it has been evaluated on a labelled, held-out set.
