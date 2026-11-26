# Inlay — SAM 3.1 experiment

A deliberately small Python experiment to find out whether Meta SAM 3.1 can segment the **entire foreground photo panel** in Inlay screenshots—not merely the person or object inside that photo.

This is an isolated experiment. It does not replace or modify the existing .NET application.

## 1. Checkpoint access (one-time)

1. Request access to the official [Meta SAM 3 Hugging Face model](https://huggingface.co/facebook/sam3.1).
2. Accept the model terms and wait for access approval.
3. Create a Hugging Face access token with read permission.

The model is downloaded on first inference and cached locally. Afterward, inference runs directly on this computer; no LM Studio server or hosted inference API is involved.

## 2. Install on Windows

Use 64-bit Python 3.12. From this directory, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

Authenticate after access is approved:

```powershell
.\.venv\Scripts\hf.exe auth login
```

The installer uses the official CUDA-enabled PyTorch wheel index and Meta's current SAM 3 repository and the SAM 3.1 checkpoint from the separate facebook/sam3.1 model repository. Meta documents Python 3.12+, PyTorch 2.7+, and CUDA 12.6+ as prerequisites. Native Windows support is not explicitly guaranteed by the upstream project; this first install is also a compatibility test. If a native Windows dependency fails, the fallback is WSL2 rather than changing the model or adding a remote service.

## 3. Run

```powershell
.\.venv\Scripts\inlay-sam3.exe C:\path\to\screenshot.png
```

The default run tests three descriptions of the complete foreground photo panel. For every prompt it saves all returned masks, transparent masked crops, overlays, raw model scores, bounding rectangles, and inference time.

To test a specific phrase:

```powershell
.\.venv\Scripts\inlay-sam3.exe screenshot.png --prompt "the complete rectangular photo panel in the foreground"
```

To add a known approximate rectangle as an independent visual prompt:

```powershell
.\.venv\Scripts\inlay-sam3.exe screenshot.png --box 120,240,900,700
```

The experiment also takes the best text-prompted rectangle and feeds it back as a box prompt. This tests whether geometry-guided refinement improves the panel boundary.

Output is written beside the source image in a directory ending in `.sam3`. Open the `*-overlay.png` files to judge what the model actually selected. The `report.json` contains raw results.

## 4. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The unit tests cover coordinate conversion and mask bounds without downloading or loading the model.

## Important interpretation

- SAM's scores are not calibrated probabilities.
- A high score does not prove that the full panel was selected.
- A mask of the person/object *inside* the photograph is a failed result for Inlay's purpose.
- The decisive test is visual: does the mask follow the outer photo panel across varied screenshots?
- No model is assumed to achieve zero errors until it has been evaluated on a labelled, held-out set.
