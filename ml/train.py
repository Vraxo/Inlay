"""Train Inlay's foreground-image segmentation model."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

from inlay_model import InlaySegmenter, segmentation_loss

SIZE = InlaySegmenter.input_size


class ScreenshotDataset(Dataset):
    def __init__(self, folder: Path, augment: bool) -> None:
        self.images = sorted(folder.glob("*.jpg"))
        self.augment = augment
        if not self.images:
            raise ValueError(f"No .jpg samples found in {folder}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.images[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        with Image.open(path.with_suffix(".png")) as source:
            mask = source.convert("L")

        if self.augment:
            if random.random() < 0.5:
                image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
            if random.random() < 0.25:
                image = ImageEnhance.Color(image).enhance(random.uniform(0.75, 1.25))
            if random.random() < 0.25:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))

        image = image.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        mask = mask.resize((SIZE, SIZE), Image.Resampling.NEAREST)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        labels = np.asarray(mask, dtype=np.float32) / 255.0
        return (torch.from_numpy(pixels.transpose(2, 0, 1).copy()),
                torch.from_numpy(labels[None].copy()))


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    intersections, unions = 0.0, 0.0
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            predictions = torch.sigmoid(logits) >= 0.5
            labels = targets.to(device) >= 0.5
            intersections += (predictions & labels).sum().item()
            unions += (predictions | labels).sum().item()
    return intersections / max(1.0, unions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/inlay-segmenter.pt"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7319)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = ScreenshotDataset(args.data / "train", augment=True)
    validation_data = ScreenshotDataset(args.data / "validation", augment=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=device.type == "cuda")

    model = InlaySegmenter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_iou = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(images), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        validation_iou = evaluate(model, validation_loader, device)
        mean_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch:02d}/{args.epochs} · loss {mean_loss:.4f} · validation IoU {validation_iou:.4f}")

        if validation_iou > best_iou:
            best_iou = validation_iou
            torch.save({"model": model.cpu().state_dict(), "input_size": SIZE,
                        "validation_iou": best_iou, "epoch": epoch}, args.output)
            model.to(device)

    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Checkpoint: {args.output}")


if __name__ == "__main__":
    main()
