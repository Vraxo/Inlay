# Inlay segmentation model

This directory contains the first trainable foreground-image segmentation pipeline. It is intentionally separate from the .NET application until a trained model has passed validation and its inference runtime has been integrated.

The model predicts a pixel mask for the prominent image displayed over an application interface. Crop geometry is derived from the predicted mask; the existing C# detector remains the production path for now.

## 1. Prepare image assets

Provide a folder of varied, legally usable photographs or artwork. Use at least 20 distinct files; more variety is better. The generator splits the assets before creating samples, so training and validation do not share source images.

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r ml/requirements.txt
```

## 2. Generate labelled screenshots

```powershell
python ml/generate_synthetic.py --assets .\image-assets --output .\data
```

Defaults create 12,000 training and 1,200 validation examples. Each screenshot has a paired binary PNG mask. Positive samples include varied placement, size, aspect ratio, rounded corners, shadows, blurred backdrops, interface chrome, and occasional overlapping controls. Negative samples contain interface thumbnails but no dominant foreground image.

The validation split uses a disjoint pool of source images. This checks source-image generalization, but synthetic validation is not a substitute for real screenshots from the intended use cases.

## 3. Train

```powershell
python ml/train.py --data .\data --output .\artifacts\inlay-segmenter.pt
```

Training automatically uses CUDA when available. The default 384×384 model is deliberately compact enough for consumer GPUs. The best validation-IoU checkpoint is saved.

## 4. Export

```powershell
python ml/export_onnx.py .\artifacts\inlay-segmenter.pt --output .\artifacts\inlay-segmenter.onnx
```

The ONNX model accepts normalized RGB tensors shaped [batch, 3, 384, 384] and returns one-channel mask logits shaped [batch, 1, 384, 384]. Apply sigmoid to obtain per-pixel foreground scores.

## Validation gate

Do not replace the current detector or ship a model based only on synthetic validation. Before integration:

- Evaluate against a held-out set of real screenshots that are never used to generate training samples.
- Include negative screenshots, competing images, image viewers, rounded and unrounded edges, overlays, blur, and unusual aspect ratios.
- Report mask IoU, crop-box IoU, false-positive rate, and the fraction of cases safe to auto-extract.
- Calibrate an abstention threshold. Low-confidence or ambiguous results must remain in Review.
- Compare against the current C# detector on exactly the same cases.

The ONNX export is a model artifact, not a complete inference integration. Browser and CLI runtimes will be added only after model quality and latency are measured. No screenshot data needs to leave the user's device at runtime.
