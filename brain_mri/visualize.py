from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from brain_mri.config import CLASS_COLORS, CLASS_NAMES, DISPLAY_NAMES
from brain_mri.data import load_rgb
from brain_mri.gradcam import overlay_heatmap

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
                     "axes.titleweight": "bold", "axes.titlesize": 13})

PALETTE = [CLASS_COLORS[c] for c in CLASS_NAMES]
LABELS = [DISPLAY_NAMES[c] for c in CLASS_NAMES]


def save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


def plot_class_distribution(counts_by_split: dict, path=None):
    splits = list(counts_by_split)
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / len(splits)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    shades = sns.color_palette("crest", len(splits))
    for i, split in enumerate(splits):
        values = [counts_by_split[split][c] for c in CLASS_NAMES]
        bars = ax.bar(x + i * width - 0.4 + width / 2, values, width, label=split.title(), color=shades[i])
        ax.bar_label(bars, fontsize=8, padding=2)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("Images")
    ax.set_title("Class distribution per split")
    ax.legend(frameon=False)
    return save(fig, path)


def plot_sample_grid(samples, n_per_class: int = 5, path=None, seed: int = 0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(CLASS_NAMES), n_per_class, figsize=(2.2 * n_per_class, 2.3 * len(CLASS_NAMES)))
    for row, name in enumerate(CLASS_NAMES):
        paths = [p for p, label in samples if label == row]
        picks = rng.choice(len(paths), size=min(n_per_class, len(paths)), replace=False) if paths else []
        for col in range(n_per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(picks):
                ax.imshow(load_rgb(paths[picks[col]]))
        axes[row, 0].set_title(DISPLAY_NAMES[name], loc="left", fontsize=11, color=CLASS_COLORS[name])
    fig.suptitle("Sample MRI scans per class", fontweight="bold", y=1.0)
    fig.tight_layout()
    return save(fig, path)


def plot_image_sizes(samples, path=None, max_images: int = 1500):
    from PIL import Image
    idx = np.random.default_rng(0).permutation(len(samples))[:max_images]
    sizes = []
    for i in idx:
        with Image.open(samples[i][0]) as img:
            sizes.append((*img.size, samples[i][1]))
    sizes = np.array(sizes)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for label, name in enumerate(CLASS_NAMES):
        m = sizes[:, 2] == label
        ax.scatter(sizes[m, 0], sizes[m, 1], s=10, alpha=0.5, color=CLASS_COLORS[name], label=DISPLAY_NAMES[name])
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Raw image resolutions")
    ax.legend(frameon=False, markerscale=2)
    return save(fig, path)


def plot_intensity_histograms(samples, path=None, per_class: int = 60):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rng = np.random.default_rng(0)
    for label, name in enumerate(CLASS_NAMES):
        paths = [p for p, l in samples if l == label]
        chosen = rng.choice(len(paths), size=min(per_class, len(paths)), replace=False)
        values = np.concatenate([np.asarray(load_rgb(paths[i]).convert("L").resize((128, 128))).ravel()
                                 for i in chosen])
        sns.kdeplot(values[values > 5], ax=ax, color=CLASS_COLORS[name], label=DISPLAY_NAMES[name], linewidth=2)
    ax.set_xlabel("Pixel intensity (background removed)")
    ax.set_title("Pixel intensity distribution by class")
    ax.legend(frameon=False)
    return save(fig, path)


def plot_augmentations(path_to_image, transform, n: int = 8, path=None, mean=None, std=None):
    from brain_mri.config import IMAGENET_MEAN, IMAGENET_STD
    mean = np.array(mean or IMAGENET_MEAN)[:, None, None]
    std = np.array(std or IMAGENET_STD)[:, None, None]
    image = load_rgb(path_to_image)
    fig, axes = plt.subplots(1, n + 1, figsize=(2.3 * (n + 1), 2.6))
    axes[0].imshow(image.resize((224, 224)))
    axes[0].set_title("Original", fontsize=10)
    for ax in axes[1:]:
        t = transform(image).numpy() * std + mean
        ax.imshow(np.clip(t.transpose(1, 2, 0), 0, 1))
        ax.set_title("Augmented", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Training augmentations", fontweight="bold")
    fig.tight_layout()
    return save(fig, path)


def plot_training_curves(history: dict, title: str = "", path=None):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    best = int(np.argmax(history["val_f1"])) + 1
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    specs = [("loss", "Loss"), ("acc", "Accuracy"), ("f1", "Macro F1")]
    for ax, (key, label) in zip(axes, specs):
        ax.plot(epochs, history[f"train_{key}"], "o-", ms=3, label="Train", color="#5B6EE1")
        ax.plot(epochs, history[f"val_{key}"], "o-", ms=3, label="Validation", color="#E4572E")
        ax.axvline(best, ls="--", color="grey", lw=1)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False)
    ax2 = axes[0].twinx()
    ax2.plot(epochs, history["lr"], color="#29B6A4", alpha=0.35, lw=1.5)
    ax2.set_ylabel("Learning rate", color="#29B6A4")
    ax2.grid(False)
    fig.suptitle(f"Training curves {title}".strip(), fontweight="bold")
    fig.tight_layout()
    return save(fig, path)


def plot_confusion_matrices(cm, path=None, title="Confusion matrix"):
    cm = np.asarray(cm)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], xticklabels=LABELS, yticklabels=LABELS,
                cbar=False, annot_kws={"size": 12})
    sns.heatmap(norm, annot=True, fmt=".1%", cmap="Blues", ax=axes[1], xticklabels=LABELS, yticklabels=LABELS,
                vmin=0, vmax=1, annot_kws={"size": 11})
    for ax, sub in zip(axes, ("Counts", "Row-normalised (recall)")):
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(sub)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return save(fig, path)


def plot_classification_report(report: dict, path=None):
    rows = CLASS_NAMES + ["macro avg", "weighted avg"]
    data = np.array([[report[r]["precision"], report[r]["recall"], report[r]["f1-score"]] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.7, vmax=1.0, ax=ax,
                xticklabels=["Precision", "Recall", "F1-score"],
                yticklabels=LABELS + ["Macro avg", "Weighted avg"])
    ax.set_title("Per-class metrics")
    return save(fig, path)


def plot_roc_pr(probs, labels, path=None):
    one_hot = np.eye(len(CLASS_NAMES))[labels]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(one_hot[:, i], probs[:, i])
        prec, rec, _ = precision_recall_curve(one_hot[:, i], probs[:, i])
        axes[0].plot(fpr, tpr, lw=2, color=CLASS_COLORS[name], label=f"{DISPLAY_NAMES[name]} (AUC {auc(fpr, tpr):.3f})")
        axes[1].plot(rec, prec, lw=2, color=CLASS_COLORS[name], label=f"{DISPLAY_NAMES[name]} (AP {auc(rec, prec):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (one-vs-rest)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curves", ylim=(0, 1.02))
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "lower left", frameon=False)
    fig.tight_layout()
    return save(fig, path)


def plot_calibration(probs, labels, reliability, ece, path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    centers = [(r["bin_low"] + r["bin_high"]) / 2 for r in reliability]
    axes[0].bar(centers, [r["accuracy"] for r in reliability], width=1 / 15, edgecolor="white",
                color="#5B6EE1", label="Accuracy")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    axes[0].set(xlabel="Confidence", ylabel="Accuracy", title=f"Reliability diagram (ECE = {ece:.3f})",
                xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(frameon=False)
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    bins = np.linspace(conf.min(), 1, 30)
    axes[1].hist(conf[correct], bins=bins, alpha=0.75, color="#29B6A4", label="Correct")
    axes[1].hist(conf[~correct], bins=bins, alpha=0.85, color="#E4572E", label="Wrong")
    axes[1].set(xlabel="Predicted confidence", ylabel="Images", title="Confidence: correct vs wrong")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return save(fig, path)


def plot_embeddings(embeddings, labels, path=None, seed: int = 42):
    coords = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for i, name in enumerate(CLASS_NAMES):
        m = labels == i
        ax.scatter(coords[m, 0], coords[m, 1], s=9, alpha=0.7, color=CLASS_COLORS[name], label=DISPLAY_NAMES[name])
    ax.set_title("t-SNE of learned image embeddings")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, markerscale=2.5)
    return save(fig, path)


def plot_gradcam_grid(items, path=None, title="Grad-CAM: where the model looks"):
    cols = 4
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(3.3 * cols, 3.4 * rows * 2))
    axes = np.atleast_2d(axes)
    for k in range(rows * cols):
        r, c = (k // cols) * 2, k % cols
        for ax in (axes[r, c], axes[r + 1, c]):
            ax.axis("off")
        if k >= len(items):
            continue
        item = items[k]
        axes[r, c].imshow(item["image"])
        axes[r + 1, c].imshow(overlay_heatmap(item["image"], item["cam"]))
        ok = item["true"] == item["pred"]
        axes[r, c].set_title(f"True: {DISPLAY_NAMES[CLASS_NAMES[item['true']]]}", fontsize=10)
        axes[r + 1, c].set_title(f"Pred: {DISPLAY_NAMES[CLASS_NAMES[item['pred']]]} ({item['conf']:.0%})",
                                 fontsize=10, color="#1B7F3B" if ok else "#C62828")
    fig.suptitle(title, fontweight="bold", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return save(fig, path)


def plot_misclassified(samples, probs, labels, path=None, limit: int = 12):
    preds = probs.argmax(1)
    wrong = np.where(preds != labels)[0]
    wrong = wrong[np.argsort(-probs[wrong].max(1))][:limit]
    if len(wrong) == 0:
        return None
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, wrong):
        ax.imshow(load_rgb(samples[i][0]))
        ax.set_title(f"True: {DISPLAY_NAMES[CLASS_NAMES[labels[i]]]}\n"
                     f"Pred: {DISPLAY_NAMES[CLASS_NAMES[preds[i]]]} ({probs[i].max():.0%})",
                     fontsize=9.5, color="#C62828")
    fig.suptitle("Most confident mistakes", fontweight="bold")
    fig.tight_layout()
    return save(fig, path)


def plot_model_comparison(results: dict, path=None):
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("macro_auc", "Macro AUC"),
               ("cohen_kappa", "Cohen's kappa")]
    names = list(results)
    x = np.arange(len(metrics))
    width = 0.8 / len(names)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [2.2, 1]})
    colors = sns.color_palette("Set2", len(names))
    for i, name in enumerate(names):
        vals = [results[name][m] for m, _ in metrics]
        bars = axes[0].bar(x + i * width - 0.4 + width / 2, vals, width, label=name, color=colors[i])
        axes[0].bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    axes[0].set_xticks(x, [label for _, label in metrics])
    lows = [results[n][m] for n in names for m, _ in metrics]
    axes[0].set_ylim(max(0.0, min(lows) - 0.05), 1.01)
    axes[0].set_title("Test-set performance")
    axes[0].legend(frameon=False)
    for i, name in enumerate(names):
        axes[1].scatter(results[name]["params_m"], results[name]["accuracy"], s=60 + 25 * results[name]["ms_per_image"],
                        color=colors[i], label=name, alpha=0.85, edgecolor="black")
        axes[1].annotate(f"{results[name]['ms_per_image']:.1f} ms/img", (results[name]["params_m"],
                         results[name]["accuracy"]), textcoords="offset points", xytext=(8, -4), fontsize=8)
    axes[1].set(xlabel="Parameters (millions)", ylabel="Accuracy", title="Accuracy vs model size")
    fig.tight_layout()
    return save(fig, path)
