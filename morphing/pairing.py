"""Explicit identity pairing and leakage-free train/val splitting.

The dataset stores one image per identity (image_N.jpg). A morph pair is an
unordered pair of two *different* images. To avoid train/val leakage we first
split the underlying images into disjoint train/val sets, then form pairs only
within each set - so no image (identity) ever appears in both splits.
"""
import random


def split_files(files, val_split=0.15, seed=42):
    """Split the list of image files into disjoint (train, val) file lists."""
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(round(len(files) * val_split)))
    val_files = files[:n_val]
    train_files = files[n_val:]
    return train_files, val_files


def build_pairs(files, max_pairs=None, seed=42):
    """Random distinct unordered pairs (no self-pairs, no duplicates).

    Override this to encode a deliberate policy (e.g. same gender/pose) once the
    dataset carries that metadata."""
    files = list(files)
    if len(files) < 2:
        return []

    rng = random.Random(seed)
    total_possible = len(files) * (len(files) - 1) // 2
    target = total_possible if max_pairs is None else min(max_pairs, total_possible)

    pairs = set()
    attempts = 0
    while len(pairs) < target and attempts < target * 50:
        a, b = rng.sample(files, 2)
        pairs.add(tuple(sorted((a, b))))
        attempts += 1
    return [list(p) for p in sorted(pairs)]
