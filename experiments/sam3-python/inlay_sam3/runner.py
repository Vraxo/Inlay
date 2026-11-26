from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .geometry import box_xyxy_to_xywh, mask_bounds, pixel_box_to_normalized_center


DEFAULT_PROMPTS = (
    "the complete foreground photograph or image panel, including its rectangular edges",
    "the entire rectangular picture displayed in front of the background",
    "the full image card or photo viewer panel, not the person or object inside it",
)


def load_model() -> tuple[Any, Any, str]:
    """Load the official Meta SAM 3.1 image model directly into this process."""
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as error:
        raise RuntimeError(
            "SAM 3 is not installed. Run setup.ps1 from this experiment directory."
        ) from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA is unavailable; inference will run on CPU and may be slow.")

    model = build_sam3_image_model(device=device, eval_mode=True)
    processor = Sam3Processor(model, device=device, confidence_threshold=0.15)
    return model, processor, device


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _save_candidate(
    image: Image.Image,
    mask: np.ndarray,
    score: float,
    method: str,
    index: int,
    output_dir: Path,
) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=bool).squeeze()
    if binary.shape != (image.height, image.width):
        raise ValueError(
            f"Model returned mask shape {binary.shape}; expected "
            f"{(image.height, image.width)}."
        )

    bounds = mask_bounds(binary)
    stem = f"{method}-{index:02d}"
    mask_path = output_dir / f"{stem}-mask.png"
    overlay_path = output_dir / f"{stem}-overlay.png"
    crop_path = output_dir / f"{stem}-crop.png"

    mask_image = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
    mask_image.save(mask_path)

    overlay = image.convert("RGBA")
    tint = Image.new("RGBA", image.size, (0, 220, 140, 0))
    tint.putalpha(mask_image.point(lambda value: int(value * 0.34)))
    Image.alpha_composite(overlay, tint).save(overlay_path)

    crop_bounds = None
    if bounds is not None:
        x, y, width, height = bounds
        crop = image.convert("RGBA").crop((x, y, x + width, y + height))
        crop.putalpha(mask_image.crop((x, y, x + width, y + height)))
        crop.save(crop_path)
        crop_bounds = {"x": x, "y": y, "width": width, "height": height}

    return {
        "method": method,
        "index": index,
        "score": float(score),
        "bounds": crop_bounds,
        "mask_pixels": int(binary.sum()),
        "mask_file": mask_path.name,
        "overlay_file": overlay_path.name,
        "crop_file": crop_path.name if bounds is not None else None,
    }


def _save_output(
    image: Image.Image,
    output: Any,
    method: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    masks = _to_numpy(output["masks"])
    scores = _to_numpy(output["scores"]).reshape(-1)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks[None, ...]

    candidates = []
    for index, mask in enumerate(masks):
        score = float(scores[index]) if index < len(scores) else 0.0
        candidates.append(
            _save_candidate(image, mask, score, method, index + 1, output_dir)
        )
    return candidates


def run_experiment(
    image_path: Path,
    output_dir: Path,
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
    manual_box: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    model, processor, device = load_model()
    results: list[dict[str, Any]] = []

    for prompt_index, prompt in enumerate(prompts, start=1):
        state = processor.set_image(image)
        _synchronize(device)
        started = time.perf_counter()
        output = processor.set_text_prompt(state=state, prompt=prompt)
        _synchronize(device)
        elapsed = time.perf_counter() - started
        candidates = _save_output(
            image, output, f"text-{prompt_index}", output_dir
        )
        results.append(
            {
                "method": f"text-{prompt_index}",
                "prompt": prompt,
                "inference_seconds": round(elapsed, 3),
                "candidates": candidates,
            }
        )
        print(
            f"Text prompt {prompt_index}: {len(candidates)} mask(s) "
            f"in {elapsed:.2f}s — {prompt}"
        )

        if candidates:
            best = max(candidates, key=lambda candidate: candidate["score"])
            if best["bounds"] is not None:
                box = tuple(best["bounds"].values())
                _run_box_prompt(
                    image, processor, device, box, output_dir,
                    f"text-{prompt_index}-box-refine", results
                )

    if manual_box is not None:
        _run_box_prompt(
            image, processor, device, manual_box, output_dir, "supplied-box", results
        )

    report = {
        "input": str(image_path.resolve()),
        "image_size": {"width": image.width, "height": image.height},
        "device": device,
        "results": results,
        "note": (
            "SAM scores are raw model outputs, not calibrated probabilities. "
            "A mask bounding rectangle is not proof that the complete photo panel "
            "was correctly identified."
        ),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved diagnostics to: {output_dir.resolve()}")
    print(f"Report: {report_path.resolve()}")
    return report


def _run_box_prompt(
    image: Image.Image,
    processor: Any,
    device: str,
    box: tuple[float, float, float, float],
    output_dir: Path,
    method: str,
    results: list[dict[str, Any]],
) -> None:
    state = processor.set_image(image)
    normalized = pixel_box_to_normalized_center(
        box, image.width, image.height
    )
    _synchronize(device)
    started = time.perf_counter()
    output = processor.add_geometric_prompt(
        state=state, box=normalized, label=True
    )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    candidates = _save_output(image, output, method, output_dir)
    results.append(
        {
            "method": method,
            "input_box_xywh": list(box),
            "inference_seconds": round(elapsed, 3),
            "candidates": candidates,
        }
    )
    print(f"Box prompt {method}: {len(candidates)} mask(s) in {elapsed:.2f}s")
