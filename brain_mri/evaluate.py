import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score, confusion_matrix,
                             f1_score, roc_auc_score)

from brain_mri.config import CLASS_NAMES
from brain_mri.models import embedding_extractor


@torch.inference_mode()
def predict_loader(model, loader, device="cpu", tta: bool = False, arch: str | None = None,
                   with_embeddings: bool = False):
    model.eval()
    probs_all, labels_all, emb_all = [], [], []
    extract = embedding_extractor(model, arch) if with_embeddings and arch else None
    start = time.time()
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        if tta:
            probs = (probs + F.softmax(model(torch.flip(x, dims=[3])), dim=1)) / 2
        probs_all.append(probs.cpu())
        labels_all.append(y)
        if extract is not None:
            emb_all.append(extract(x).cpu())
    elapsed = time.time() - start
    probs = torch.cat(probs_all).numpy()
    labels = torch.cat(labels_all).numpy()
    result = {"probs": probs, "labels": labels, "preds": probs.argmax(1),
              "ms_per_image": 1000 * elapsed / len(labels)}
    if extract is not None:
        result["embeddings"] = torch.cat(emb_all).numpy()
    return result


def expected_calibration_error(probs, labels, n_bins: int = 15):
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    bins = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            acc, avg_conf = correct[mask].mean(), conf[mask].mean()
            ece += mask.mean() * abs(acc - avg_conf)
            rows.append({"bin_low": float(lo), "bin_high": float(hi), "accuracy": float(acc),
                         "confidence": float(avg_conf), "count": int(mask.sum())})
    return float(ece), rows


def compute_metrics(probs, labels, class_names=CLASS_NAMES) -> dict:
    preds = probs.argmax(1)
    report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
    ece, reliability = expected_calibration_error(probs, labels)
    one_hot = np.eye(len(class_names))[labels]
    per_class_auc = {name: float(roc_auc_score(one_hot[:, i], probs[:, i]))
                     for i, name in enumerate(class_names) if 0 < one_hot[:, i].sum() < len(labels)}
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
        "cohen_kappa": float(cohen_kappa_score(labels, preds)),
        "macro_auc": float(np.mean(list(per_class_auc.values()))) if per_class_auc else float("nan"),
        "per_class_auc": per_class_auc,
        "ece": ece,
        "reliability": reliability,
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(class_names)))).tolist(),
        "report": {k: v for k, v in report.items() if k in class_names or k in ("macro avg", "weighted avg")},
        "n_samples": int(len(labels)),
    }


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(-1) / np.log(p.shape[-1])
