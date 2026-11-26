from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SAMPLES_DIR = ROOT / "assets" / "samples"
DEFAULT_CHECKPOINT = MODELS_DIR / "best_model.pth"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}
CLASS_COLORS = {
    "glioma": "#E4572E",
    "meningioma": "#F3A712",
    "notumor": "#29B6A4",
    "pituitary": "#5B6EE1",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LEGACY_MEAN = (0.5, 0.5, 0.5)
LEGACY_STD = (0.5, 0.5, 0.5)


@dataclass
class TrainConfig:
    data_dir: str = "data"
    arch: str = "resnet18"
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    lr: float = 3e-4
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    warmup_epochs: int = 1
    freeze_epochs: int = 1
    val_fraction: float = 0.15
    patience: int = 6
    num_workers: int = 2
    seed: int = 42
    use_amp: bool = True
    max_train_batches: int | None = None
    output_dir: str = str(MODELS_DIR)
    extra: dict = field(default_factory=dict)
