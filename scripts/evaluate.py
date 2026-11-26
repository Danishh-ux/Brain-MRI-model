import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import torch
from torch.utils.data import DataLoader

from brain_mri.config import CLASS_NAMES, DEFAULT_CHECKPOINT, REPORTS_DIR
from brain_mri.data import MRIDataset, build_transforms, list_images
from brain_mri.evaluate import compute_metrics, predict_loader
from brain_mri.gradcam import gradcam_examples
from brain_mri.models import count_parameters, load_checkpoint
from brain_mri import visualize as viz


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and generate report figures.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--out", default=str(REPORTS_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--no-tsne", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model, meta = load_checkpoint(args.checkpoint, device)
    _, eval_tf = build_transforms(meta["img_size"], meta["mean"], meta["std"])
    samples = list_images(args.test_dir)
    loader = DataLoader(MRIDataset(samples, eval_tf), batch_size=args.batch_size, shuffle=False)

    out = Path(args.out)
    figures = out / "figures"
    print(f"Evaluating {meta['arch']} on {len(samples)} images ({device}, TTA={args.tta})")
    res = predict_loader(model, loader, device, tta=args.tta, arch=meta["arch"], with_embeddings=not args.no_tsne)
    metrics = compute_metrics(res["probs"], res["labels"])
    metrics.update({"arch": meta["arch"], "params_m": count_parameters(model) / 1e6,
                    "ms_per_image": res["ms_per_image"], "tta": args.tta, "device": str(device),
                    "checkpoint_format": meta["format_version"]})

    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({
        "path": [str(Path(p).relative_to(Path(args.test_dir))) for p, _ in samples],
        "label": [CLASS_NAMES[l] for l in res["labels"]],
        "pred": [CLASS_NAMES[p] for p in res["preds"]],
        **{f"p_{c}": res["probs"][:, i] for i, c in enumerate(CLASS_NAMES)},
    }).to_csv(out / "test_predictions.csv", index=False)

    viz.plot_confusion_matrices(metrics["confusion_matrix"], figures / "confusion_matrix.png",
                                f"Confusion matrix - {meta['arch']} (accuracy {metrics['accuracy']:.2%})")
    viz.plot_classification_report(metrics["report"], figures / "classification_report.png")
    viz.plot_roc_pr(res["probs"], res["labels"], figures / "roc_pr_curves.png")
    viz.plot_calibration(res["probs"], res["labels"], metrics["reliability"], metrics["ece"],
                         figures / "calibration.png")
    viz.plot_misclassified(samples, res["probs"], res["labels"], figures / "misclassified.png")
    viz.plot_gradcam_grid(gradcam_examples(model, meta, samples, res["probs"], res["labels"], eval_tf, device),
                          figures / "gradcam_examples.png")
    if "embeddings" in res:
        viz.plot_embeddings(res["embeddings"], res["labels"], figures / "tsne_embeddings.png")
    if meta.get("history", {}).get("train_loss"):
        viz.plot_training_curves(meta["history"], f"- {meta['arch']}", figures / "training_curves.png")

    print(f"Accuracy {metrics['accuracy']:.4f} | Macro F1 {metrics['macro_f1']:.4f} | "
          f"Macro AUC {metrics['macro_auc']:.4f} | ECE {metrics['ece']:.4f} | {res['ms_per_image']:.1f} ms/img")
    print(f"Saved metrics and figures to {out}")


if __name__ == "__main__":
    main()
