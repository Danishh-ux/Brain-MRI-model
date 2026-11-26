from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from brain_mri.config import DEFAULT_CHECKPOINT, DISPLAY_NAMES
from brain_mri.data import build_transforms
from brain_mri.evaluate import predictive_entropy
from brain_mri.gradcam import GradCAM, focus_box
from brain_mri.models import count_parameters, gradcam_target_layer, load_checkpoint


@dataclass
class Prediction:
    probs: dict
    label: str
    display_label: str
    confidence: float
    entropy: float
    cam: np.ndarray
    focus_box: tuple | None
    grayscale_score: float
    tta: bool

    @property
    def has_tumor(self) -> bool:
        return self.label != "notumor"

    def is_reliable(self, threshold: float) -> bool:
        return self.confidence >= threshold and self.looks_like_mri

    @property
    def looks_like_mri(self) -> bool:
        return self.grayscale_score >= 0.9


def grayscale_score(image: Image.Image) -> float:
    arr = np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.float32)
    spread = arr.max(axis=2) - arr.min(axis=2)
    return float(1.0 - np.clip(spread.mean() / 40.0, 0, 1))


class Predictor:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.meta = load_checkpoint(checkpoint, self.device)
        self.class_names = self.meta["class_names"]
        _, self.transform = build_transforms(self.meta["img_size"], self.meta["mean"], self.meta["std"])
        self.target_layer = gradcam_target_layer(self.model, self.meta["arch"])
        self.num_params = count_parameters(self.model)

    def _tensor(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image, tta: bool = False, cam_method: str = "gradcam") -> Prediction:
        image = image.convert("RGB")
        x = self._tensor(image)
        with GradCAM(self.model, self.target_layer, plus_plus=cam_method == "gradcam++") as cam_fn:
            cam, _, probs = cam_fn(x)
        if tta:
            with torch.inference_mode():
                flipped = torch.softmax(self.model(torch.flip(x, dims=[3])), dim=1)[0].cpu().numpy()
            probs = (probs + flipped) / 2
        idx = int(np.argmax(probs))
        label = self.class_names[idx]
        return Prediction(
            probs={name: float(p) for name, p in zip(self.class_names, probs)},
            label=label,
            display_label=DISPLAY_NAMES.get(label, label),
            confidence=float(probs[idx]),
            entropy=float(predictive_entropy(probs[None])[0]),
            cam=cam,
            focus_box=focus_box(cam),
            grayscale_score=grayscale_score(image),
            tta=tta,
        )
