import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain_mri.config import CLASS_NAMES


def _fake_mri(seed: int, label: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:96, :96]
    brain = ((xx - 48) ** 2 / 38 ** 2 + (yy - 48) ** 2 / 42 ** 2) < 1
    img = brain * 110 + rng.normal(0, 12, (96, 96))
    cx, cy = 30 + 12 * label, 35 + 8 * label
    img += 120 * (((xx - cx) ** 2 + (yy - cy) ** 2) < (6 + 2 * label) ** 2)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGB")


@pytest.fixture(scope="session")
def fake_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("mri")
    for split, n in (("Training", 12), ("Testing", 4)):
        for label, name in enumerate(CLASS_NAMES):
            d = root / split / name
            d.mkdir(parents=True)
            for i in range(n):
                _fake_mri((0 if split == "Training" else 5000) + 100 * label + i, label).save(d / f"{split[:2]}_{name}_{i}.jpg")
    return root
