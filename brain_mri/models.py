from pathlib import Path

import torch
from torch import nn
from torchvision import models

from brain_mri.config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, LEGACY_MEAN, LEGACY_STD

SUPPORTED_ARCHS = ("resnet18", "resnet50", "efficientnet_b0", "convnext_tiny")


def build_model(arch: str = "resnet18", num_classes: int = 4, pretrained: bool = True,
                dropout: float = 0.3) -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier[1].in_features, num_classes))
    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture '{arch}'. Choose from {SUPPORTED_ARCHS}.")
    return model


def head_parameters(model: nn.Module, arch: str):
    if arch.startswith("resnet"):
        return list(model.fc.parameters())
    return list(model.classifier.parameters())


def backbone_parameters(model: nn.Module, arch: str):
    head_ids = {id(p) for p in head_parameters(model, arch)}
    return [p for p in model.parameters() if id(p) not in head_ids]


def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool) -> None:
    for p in backbone_parameters(model, arch):
        p.requires_grad = trainable


def gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    if arch.startswith("resnet"):
        return model.layer4[-1]
    if arch.startswith("efficientnet"):
        return model.features[-1]
    if arch.startswith("convnext"):
        return model.features[-1][-1]
    raise ValueError(f"No Grad-CAM layer defined for '{arch}'.")


def embedding_extractor(model: nn.Module, arch: str):
    if arch.startswith("resnet"):
        return lambda x: torch.flatten(model.avgpool(_resnet_features(model, x)), 1)
    return lambda x: torch.flatten(model.avgpool(model.features(x)), 1)


def _resnet_features(model, x):
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    return model.layer4(model.layer3(model.layer2(model.layer1(x))))


def save_checkpoint(path, model, arch, img_size=224, metrics=None, history=None,
                    class_names=CLASS_NAMES, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 2,
        "arch": arch,
        "img_size": img_size,
        "class_names": list(class_names),
        "mean": list(mean),
        "std": list(std),
        "metrics": metrics or {},
        "history": history or {},
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path, device: str | torch.device = "cpu"):
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and raw.get("format_version") == 2:
        meta = {k: v for k, v in raw.items() if k != "state_dict"}
        state_dict = raw["state_dict"]
    else:
        meta = {"format_version": 1, "arch": "resnet18", "img_size": 224, "class_names": list(CLASS_NAMES),
                "mean": list(LEGACY_MEAN), "std": list(LEGACY_STD), "metrics": {}, "history": {}}
        state_dict = raw
    model = build_model(meta["arch"], num_classes=len(meta["class_names"]), pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, meta


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
