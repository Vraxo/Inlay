# Inlay — local SAM 3.1 photo-panel experiment

This experiment tests whether Meta's **official SAM 3.1** can segment the complete foreground photo panel in an Inlay screenshot—not merely the person or object inside the photo.

It runs locally in Python with PyTorch. There is no custom C++ runtime, GGUF conversion, model server, or image upload.

## Setup (Windows)

Requirements:

- 64-bit Python 3.12
- An NVIDIA GPU and a compatible NVIDIA driver for CUDA acceleration

From this directory in PowerShell, run:

    .\setup.ps1

The script creates a local .venv, installs PyTorch with the CUDA 12.8 runtime, and installs Meta's official SAM 3 source archive plus the Inlay runner. The source archive is downloaded over HTTPS from GitHub's archive host; Git itself is not used. The `einops` and Windows Triton dependencies are installed from pinned official wheel URLs with SHA-256 checks, without querying the PyPI package index. The Triton wheel is matched to PyTorch 2.10 and Python 3.12; Linux environments skip this Windows-only wheel.

**You do not need to install Git, the CUDA Toolkit, CMake, Visual Studio C++ Build Tools, or any separate native compiler.** PyTorch's CUDA wheel includes the runtime libraries; an NVIDIA driver is still required for GPU inference. If CUDA is unavailable, the runner falls back to CPU, which can be very slow.

## Model access (one time)

SAM 3.1 weights are hosted behind Hugging Face access approval.

1. Open the official [facebook/sam3.1 model page](https://huggingface.co/facebook/sam3.1), sign in, and request/accept access.
2. After access is approved, authenticate from this directory:

    .\.venv\Scripts\hf.exe auth login

The runner downloads the official sam3.1_multiplex.pt checkpoint automatically on first use and Hugging Face caches it locally. **Do not place a GGUF file or manually copy model weights into this project.**

## Run

    .\.venv\Scripts\inlay-sam3.exe C:\path\to\screenshot.png

Try a custom prompt (repeat --prompt to compare several):

    .\.venv\Scripts\inlay-sam3.exe screenshot.png --prompt "the complete foreground photo panel, including its outer edges"

To provide an approximate rectangle as an additional visual prompt:

    .\.venv\Scripts\inlay-sam3.exe screenshot.png --box 120,180,900,700

By default, three panel-focused prompts are tested. The image is encoded once and its embeddings are reused between text prompts. The best text candidate's bounding rectangle is also tested as a visual box prompt.

## Output

A directory named <image-name>.sam3 is created beside the input image. It contains:

- *-overlay.png: the predicted mask over the original screenshot
- *-mask.png: the binary mask
- *-crop.png: the masked crop with transparency outside the predicted region
- report.json: prompts, raw model scores, crop bounds, device, and timings

**Inspect the overlay first.** A mask around the subject inside the photo is a failure for Inlay's purpose; the mask must follow the outer photo panel. Model scores are not calibrated probabilities and do not prove crop correctness.

## Tests

The geometry tests do not load SAM or download model weights:

    python -m pip install -r requirements.txt
    python -m pytest -q

The experiment is intentionally separate from Inlay's .NET detector and browser app. It is an evaluation path, not yet a production integration. No zero-error claim should be made until it has been tested on a labelled, held-out screenshot set.
