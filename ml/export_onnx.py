"""Export a trained Inlay segmentation checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from inlay_model import InlaySegmenter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/inlay-segmenter.onnx"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = InlaySegmenter()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, 3, InlaySegmenter.input_size, InlaySegmenter.input_size)
    torch.onnx.export(
        model, sample, args.output, input_names=["image"], output_names=["mask_logits"],
        dynamic_axes={"image": {0: "batch"}, "mask_logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"Exported {args.output}")
    print(f"Validation IoU recorded in checkpoint: {checkpoint.get('validation_iou', 'unknown')}")


if __name__ == "__main__":
    main()
