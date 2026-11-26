import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain_mri.config import TrainConfig
from brain_mri.models import SUPPORTED_ARCHS
from brain_mri.train import summarize_history, train


def main():
    parser = argparse.ArgumentParser(description="Train a brain MRI tumor classifier.")
    parser.add_argument("--arch", default="resnet18", choices=SUPPORTED_ARCHS)
    for f in fields(TrainConfig):
        if f.name in ("arch", "extra", "use_amp"):
            continue
        kind = int if f.type in (int, "int") else float if f.type in (float, "float") else str
        if f.name == "max_train_batches":
            kind = int
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=kind, default=f.default)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    values = {f.name: getattr(args, f.name) for f in fields(TrainConfig) if hasattr(args, f.name)}
    cfg = TrainConfig(**values, use_amp=not args.no_amp)
    _, history, ckpt = train(cfg)
    print(summarize_history(history))
    print(f"Best checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
