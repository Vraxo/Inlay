from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from ml.generate_synthetic import make_sample


class SyntheticGeneratorTests(unittest.TestCase):
    def test_generates_foreground_masks_and_negative_examples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = []
            for index in range(24):
                path = Path(temporary) / f"asset-{index:02d}.png"
                Image.new("RGB", (96, 80), (index * 9 % 255, index * 17 % 255, index * 23 % 255)).save(path)
                assets.append(path)

            samples = [make_sample(assets, random.Random(seed)) for seed in range(100)]
            self.assertTrue(any(mask.getbbox() is not None for _, mask in samples))
            self.assertTrue(any(mask.getbbox() is None for _, mask in samples))
            for image, mask in samples:
                self.assertEqual(image.size, (768, 768))
                self.assertEqual(mask.size, (768, 768))


if __name__ == "__main__":
    unittest.main()
