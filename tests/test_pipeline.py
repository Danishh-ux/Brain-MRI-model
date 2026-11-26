import numpy as np
import pytest
import torch
from PIL import Image

from brain_mri.config import CLASS_NAMES, TrainConfig
from brain_mri.data import build_loaders, find_duplicates, list_images, remove_leaked, stratified_split
from brain_mri.evaluate import compute_metrics, expected_calibration_error, predictive_entropy
from brain_mri.gradcam import GradCAM, focus_box, overlay_heatmap
from brain_mri.inference import Predictor, grayscale_score
from brain_mri.models import (SUPPORTED_ARCHS, build_model, gradcam_target_layer, load_checkpoint,
                              save_checkpoint)
from brain_mri.train import train


def test_stratified_split_keeps_class_balance(fake_dataset):
    samples = list_images(fake_dataset / "Training")
    train_s, val_s = stratified_split(samples, 0.25, seed=1)
    assert len(train_s) + len(val_s) == len(samples)
    assert {label for _, label in val_s} == set(range(len(CLASS_NAMES)))
    assert not {p for p, _ in train_s} & {p for p, _ in val_s}


def test_duplicate_detection_finds_copied_image(fake_dataset, tmp_path):
    train_s = list_images(fake_dataset / "Training")
    leaked_path = tmp_path / "copy.jpg"
    Image.open(train_s[0][0]).save(leaked_path)
    duplicates = find_duplicates(train_s, [(leaked_path, train_s[0][1])])
    assert len(duplicates) == 1
    assert len(remove_leaked(train_s, duplicates)) < len(train_s)


@pytest.mark.parametrize("arch", ["resnet18", "efficientnet_b0"])
def test_models_forward_and_gradcam(arch):
    model = build_model(arch, pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)
    assert model(x).shape == (1, 4)
    for plus_plus in (False, True):
        with GradCAM(model, gradcam_target_layer(model, arch), plus_plus=plus_plus) as cam_fn:
            cam, idx, probs = cam_fn(x)
        assert cam.shape == (224, 224)
        assert 0.0 <= cam.min() and cam.max() <= 1.0
        assert 0 <= idx < 4 and np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_supported_archs_build():
    for arch in SUPPORTED_ARCHS:
        assert build_model(arch, pretrained=False) is not None


def test_checkpoint_roundtrip_and_legacy_format(tmp_path):
    model = build_model("resnet18", pretrained=False)
    save_checkpoint(tmp_path / "v2.pth", model, "resnet18", metrics={"val_f1": 0.9})
    loaded, meta = load_checkpoint(tmp_path / "v2.pth")
    assert meta["format_version"] == 2 and meta["metrics"]["val_f1"] == 0.9
    torch.save(model.state_dict(), tmp_path / "legacy.pth")
    legacy, legacy_meta = load_checkpoint(tmp_path / "legacy.pth")
    assert legacy_meta["format_version"] == 1 and legacy_meta["mean"] == [0.5, 0.5, 0.5]
    x = torch.randn(2, 3, 224, 224)
    assert torch.allclose(loaded(x), legacy(x), atol=1e-5)


def test_metrics_on_perfect_and_random_predictions():
    labels = np.repeat(np.arange(4), 25)
    perfect = np.eye(4)[labels] * 0.97 + 0.01
    m = compute_metrics(perfect, labels)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["macro_auc"] == 1.0
    assert np.array(m["confusion_matrix"]).trace() == 100
    ece, rows = expected_calibration_error(perfect, labels)
    assert ece < 0.05 and rows
    uniform = np.full((3, 4), 0.25)
    assert np.allclose(predictive_entropy(uniform), 1.0)


def test_overlay_and_focus_box():
    image = Image.new("RGB", (120, 100), "gray")
    cam = np.zeros((224, 224))
    cam[50:100, 60:120] = 1.0
    assert overlay_heatmap(image, cam).size == image.size
    assert focus_box(cam) == (60, 50, 119, 99)
    assert focus_box(np.zeros((10, 10))) is None


def test_grayscale_score_flags_colour_photos():
    gray = Image.new("RGB", (64, 64), (90, 90, 90))
    colour = Image.fromarray(np.dstack([np.full((64, 64), v, np.uint8) for v in (230, 40, 60)]))
    assert grayscale_score(gray) > 0.95
    assert grayscale_score(colour) < 0.5


def test_training_smoke_and_predictor(fake_dataset, tmp_path):
    cfg = TrainConfig(data_dir=str(fake_dataset), arch="resnet18", epochs=2, batch_size=8, num_workers=0,
                      freeze_epochs=1, warmup_epochs=1, val_fraction=0.25, output_dir=str(tmp_path), use_amp=False)
    loaders, _ = build_loaders(fake_dataset, 224, 8, 0.25, 0, num_workers=0)
    model, history, ckpt = train(cfg, loaders=loaders, log=lambda *_: None)
    assert len(history["train_loss"]) == 2 and ckpt.exists()
    assert (tmp_path / "resnet18_history.json").exists()

    predictor = Predictor(ckpt, device="cpu")
    image = Image.open(next((fake_dataset / "Testing" / "glioma").iterdir()))
    pred = predictor.predict(image, tta=True, cam_method="gradcam++")
    assert pred.label in CLASS_NAMES and np.isclose(sum(pred.probs.values()), 1.0, atol=1e-4)
    assert pred.cam.shape == (224, 224) and pred.looks_like_mri
