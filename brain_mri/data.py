import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from brain_mri.config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(split_dir: str | Path) -> list[tuple[Path, int]]:
    split_dir = Path(split_dir)
    samples = []
    for label, name in enumerate(CLASS_NAMES):
        class_dir = split_dir / name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))
    return samples


def stratified_split(samples, val_fraction: float, seed: int):
    by_class = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(by_class):
        items = by_class[label][:]
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_fraction)))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def class_counts(samples) -> dict[str, int]:
    counts = Counter(label for _, label in samples)
    return {CLASS_NAMES[i]: counts.get(i, 0) for i in range(len(CLASS_NAMES))}


def build_transforms(img_size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def load_rgb(path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


class MRIDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(load_rgb(path)), label


def make_balanced_sampler(samples) -> WeightedRandomSampler:
    counts = Counter(label for _, label in samples)
    weights = [1.0 / counts[label] for _, label in samples]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)


def build_loaders(data_dir, img_size=224, batch_size=32, val_fraction=0.15, seed=42,
                  num_workers=2, balanced=False):
    data_dir = Path(data_dir)
    train_all = list_images(data_dir / "Training")
    test = list_images(data_dir / "Testing")
    train, val = stratified_split(train_all, val_fraction, seed)
    return build_loaders_from_samples(train, val, test, img_size, batch_size, num_workers, balanced)


def build_loaders_from_samples(train, val, test, img_size=224, batch_size=32, num_workers=2, balanced=False):
    train_tf, eval_tf = build_transforms(img_size)
    pin = torch.cuda.is_available()
    sampler = make_balanced_sampler(train) if balanced else None
    loaders = {
        "train": DataLoader(MRIDataset(train, train_tf), batch_size=batch_size, shuffle=sampler is None,
                            sampler=sampler, num_workers=num_workers, pin_memory=pin, drop_last=True),
        "val": DataLoader(MRIDataset(val, eval_tf), batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin),
        "test": DataLoader(MRIDataset(test, eval_tf), batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin),
    }
    splits = {"train": train, "val": val, "test": test}
    return loaders, splits


def average_hash(path, hash_size: int = 16) -> str:
    img = load_rgb(path).convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(img, dtype=np.float32)
    bits = (pixels > pixels.mean()).flatten()
    return np.packbits(bits).tobytes().hex()


def find_duplicates(train_samples, test_samples):
    train_hashes = defaultdict(list)
    for path, label in train_samples:
        train_hashes[average_hash(path)].append((path, label))
    duplicates = []
    for path, label in test_samples:
        matches = train_hashes.get(average_hash(path))
        if matches:
            duplicates.append({"test": path, "test_label": label, "train": [m[0] for m in matches],
                               "label_conflict": any(m[1] != label for m in matches)})
    return duplicates


def remove_leaked(train_samples, duplicates):
    leaked = {Path(p) for d in duplicates for p in d["train"]}
    return [s for s in train_samples if Path(s[0]) not in leaked]
