# Inlay — Qwen 3 VL crop-refinement experiment

This experiment tests whether Qwen 3 VL through LM Studio, combined with a small Python crop-and-review loop, can identify the complete foreground photo/image panel in a screenshot.

It is deliberately separate from Inlay's .NET detector and the SAM 3.1 experiment. It does not install PyTorch, download model weights, or change either existing implementation.

## How it works

1. Start the LM Studio local API server if it is not already running.
2. Check whether the configured Qwen model is loaded. If not, inspect downloaded LLMs using lms ls --llm --json, find the matching local model key, and load it with the configured API identifier.
3. Send the screenshot to the Qwen3-VL Instruct model through LM Studio's native `/api/v1/chat` endpoint.
4. Ask for the outer rectangle of the complete foreground image panel as a compact JSON object.
5. Convert the returned normalized coordinates into pixel coordinates and create a crop from the original, full-resolution image using Pillow.
6. Send Qwen the original screenshot, a preview with the current rectangle drawn over it, and the extracted crop.
7. Ask Qwen to verify the crop or adjust its rectangle. Repeat until it confirms an unchanged rectangle, makes no progress, or reaches the iteration limit.
8. Save the crop candidate, visual diagnostics, and a JSON report.

The model returns coordinates normalized to the range 0–1000 relative to the original image. Python validates the rectangle and performs the crop using the original pixels. The model never reconstructs or redraws the image.

A model's done response is only its own assessment, not proof of correctness. Inspect the overlays and compare the results with labelled ground truth before considering automatic extraction reliable.

## Requirements

- 64-bit Python 3.12
- A current LM Studio version with the native REST chat API
- LM Studio installed
- The Qwen3-VL-8B Instruct model downloaded in LM Studio
- The LM Studio lms CLI available on PATH if the script should start the server or load a model automatically

From this directory in PowerShell:

    .\setup.ps1

This creates a separate .venv and installs only Requests, Pillow, and pytest. It does not touch experiments/sam3-python/.venv.

## Model choice

**Use Qwen3-VL-8B Instruct, not Qwen3-VL-8B Thinking.**

Qwen publishes these as separate checkpoints. The Thinking checkpoint is a reasoning-enhanced variant, not the Instruct checkpoint with a toggle that can reliably be switched off. It can spend the entire output-token budget reasoning without returning the compact JSON needed by this experiment.

In LM Studio, search for **Qwen3-VL-8B** (catalog model key `qwen/qwen3-vl-8b`). Its available GGUF and MLX source files are the Instruct variant. Do not select the separately named Qwen3-VL-8B-Thinking model.

The model must already be downloaded; the experiment does not fetch large model weights automatically.

## Run

With the Instruct model downloaded in LM Studio:

    .\.venv\Scripts\python.exe -m inlay_qwen3vl C:\path\to\screenshot.jpg

The default API identifier is:

    qwen3vl-8b

When that model is not already loaded, Inlay searches downloaded LLM model keys and loads the matching Instruct model automatically. It will not silently substitute a model whose key identifies it as a Thinking checkpoint.

If the local model key cannot be matched automatically, list downloaded LLMs with:

    lms ls --llm

Then supply the exact Instruct model key shown by LM Studio:

    .\.venv\Scripts\python.exe -m inlay_qwen3vl screenshot.jpg --load-model <exact-instruct-model-key>

Useful options:

- --model qwen3vl-8b — API identifier assigned to the loaded model
- --load-model <exact-instruct-model-key> — override automatic local-model discovery
- --max-iterations 5 — maximum number of model refinement rounds (minimum 2)
- --output <directory> — diagnostics output directory
- --api-url http://localhost:1234/api/v1/chat — LM Studio native chat endpoint
- --max-image-dimension 2048 — maximum side length of images sent to Qwen

You can also configure INLAY_QWEN_MODEL, INLAY_LMSTUDIO_API_URL, INLAY_LMS_EXECUTABLE, and INLAY_LMS_LOAD_MODEL environment variables.

## Output

By default, the experiment creates a directory named <image-name>.qwen3vl beside the input image. It contains:

- crop.png when Qwen confirms a stable rectangle
- candidate.png when a rectangle was found but not confirmed
- overlay.png with the final rectangle drawn over the original screenshot
- iteration-XX-overlay.png and iteration-XX-candidate.png for each round
- report.json with the model's responses, rectangles, iteration status, and output paths

An unconfirmed candidate is intentionally not named crop.png. A result is only marked accepted when Qwen confirms the same rectangle on a subsequent visual review. Even then, that status is not a measured accuracy guarantee.

## Tests

The tests use synthetic images and a fake model client. They do not require LM Studio or a downloaded model:

    .\.venv\Scripts\python.exe -m pytest -q

## Current scope

This first experiment measures the quality of Qwen's rectangle proposal and iterative visual verification. It uses Pillow for exact pixel extraction but does not yet apply OpenCV edge snapping or other classical-CV boundary refinement. That should be added only after the raw Qwen results show where such refinement would help.
