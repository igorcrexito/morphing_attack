import os
import yaml

from morphing.pairing import split_files, build_pairs
from morphing.dataset import build_cache


if __name__ == "__main__":
    with open("execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    dp = params["dataset_parameters"]
    output_dir = str(dp["output_dir"])
    cache_dir = str(dp["cache_dir"])
    val_split = float(dp["val_split"])
    seed = int(dp["seed"])
    max_pairs = dp["max_pairs"]
    max_val_pairs = dp["max_val_pairs"]
    alpha = float(params["morphing_parameters"]["alpha"])
    width = int(params["image_parameters"]["image_width"])
    height = int(params["image_parameters"]["image_height"])

    files = sorted(f for f in os.listdir(output_dir) if f.endswith(".jpg"))
    print(f"Found {len(files)} landmarked images in {output_dir}")

    train_files, val_files = split_files(files, val_split=val_split, seed=seed)
    print(f"Identity split -> train: {len(train_files)}, val: {len(val_files)}")

    train_pairs = build_pairs(train_files, max_pairs=max_pairs, seed=seed)
    val_pairs = build_pairs(val_files, max_pairs=max_val_pairs, seed=seed + 1)
    print(f"Pairs -> train: {len(train_pairs)}, val: {len(val_pairs)}")

    os.makedirs(cache_dir, exist_ok=True)
    train_cache = os.path.join(cache_dir, "train.npz")
    val_cache = os.path.join(cache_dir, "val.npz")

    print("Warping + caching train pairs...")
    build_cache(output_dir, train_pairs, alpha, width, height, train_cache)
    print(f"  wrote {train_cache}")

    print("Warping + caching val pairs...")
    build_cache(output_dir, val_pairs, alpha, width, height, val_cache)
    print(f"  wrote {val_cache}")

    print("Done.")
