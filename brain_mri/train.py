import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from brain_mri.config import TrainConfig
from brain_mri.data import build_loaders, set_seed
from brain_mri.models import (backbone_parameters, build_model, head_parameters, save_checkpoint,
                              set_backbone_trainable)


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def run_epoch(model, loader, loss_fn, device, optimizer=None, scheduler=None, scaler=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    total_loss, labels_all, preds_all = 0.0, [], []
    use_amp = scaler is not None and scaler.is_enabled()
    with torch.set_grad_enabled(training):
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            total_loss += loss.item() * y.size(0)
            labels_all.append(y.cpu())
            preds_all.append(logits.argmax(1).cpu())
    labels = torch.cat(labels_all).numpy()
    preds = torch.cat(preds_all).numpy()
    return {
        "loss": total_loss / len(labels),
        "acc": float((labels == preds).mean()),
        "f1": float(f1_score(labels, preds, average="macro")),
    }


def train(cfg: TrainConfig, loaders=None, log=print):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if loaders is None:
        loaders, _ = build_loaders(cfg.data_dir, cfg.img_size, cfg.batch_size, cfg.val_fraction,
                                   cfg.seed, cfg.num_workers)

    model = build_model(cfg.arch).to(device)
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters(model, cfg.arch), "lr": cfg.lr * cfg.backbone_lr_mult},
        {"params": head_parameters(model, cfg.arch), "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)
    steps_per_epoch = len(loaders["train"]) if cfg.max_train_batches is None else min(
        len(loaders["train"]), cfg.max_train_batches)
    scheduler = warmup_cosine(optimizer, cfg.warmup_epochs * steps_per_epoch, cfg.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.use_amp and device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    eval_loss_fn = nn.CrossEntropyLoss()

    history = {k: [] for k in ("train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1",
                               "lr", "epoch_time")}
    best_f1, best_epoch, bad_epochs = -1.0, 0, 0
    ckpt_path = Path(cfg.output_dir) / f"{cfg.arch}_best.pth"

    for epoch in range(1, cfg.epochs + 1):
        set_backbone_trainable(model, cfg.arch, epoch > cfg.freeze_epochs)
        start = time.time()
        tr = run_epoch(model, loaders["train"], loss_fn, device, optimizer, scheduler, scaler, cfg.max_train_batches)
        va = run_epoch(model, loaders["val"], eval_loss_fn, device, max_batches=cfg.max_train_batches)
        elapsed = time.time() - start

        for key, value in (("train_loss", tr["loss"]), ("train_acc", tr["acc"]), ("train_f1", tr["f1"]),
                           ("val_loss", va["loss"]), ("val_acc", va["acc"]), ("val_f1", va["f1"]),
                           ("lr", optimizer.param_groups[1]["lr"]), ("epoch_time", elapsed)):
            history[key].append(float(value))

        improved = va["f1"] > best_f1
        if improved:
            best_f1, best_epoch, bad_epochs = va["f1"], epoch, 0
            save_checkpoint(ckpt_path, model, cfg.arch, cfg.img_size,
                            metrics={"val_f1": best_f1, "val_acc": va["acc"], "best_epoch": epoch})
        else:
            bad_epochs += 1

        log(f"[{cfg.arch}] epoch {epoch:02d}/{cfg.epochs} | train loss {tr['loss']:.4f} acc {tr['acc']:.3f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.3f} f1 {va['f1']:.3f} | {elapsed:.0f}s"
            f"{'  * best' if improved else ''}")

        if bad_epochs >= cfg.patience:
            log(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, val F1 {best_f1:.4f})")
            break

    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    save_checkpoint(ckpt_path, model, cfg.arch, cfg.img_size, metrics=best["metrics"], history=history)
    (Path(cfg.output_dir) / f"{cfg.arch}_history.json").write_text(json.dumps(history, indent=2))
    return model, history, ckpt_path


def summarize_history(history: dict) -> dict:
    best = int(np.argmax(history["val_f1"]))
    return {"best_epoch": best + 1, "val_f1": history["val_f1"][best], "val_acc": history["val_acc"][best],
            "epochs_run": len(history["val_f1"]), "train_time_s": float(sum(history["epoch_time"]))}
