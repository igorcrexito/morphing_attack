"""Detect landmarks on the source faces and split them into train/val.

Loads the images under ``image_parameters.dataset_path`` (the FRLL frontal set),
splits them into a training and a validation set *by identity* (so the same person
never appears in both - FRLL stores several images per person, e.g. 172_03.jpg), and
for each image writes a 224x224 copy plus its 68-point landmark .csv into the train
or val output directory.
"""
import os
import random
import yaml
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image

from image_utils.image_loader import ImageLoader
from landmarks.landmark import Landmarks

tf.get_logger().setLevel("ERROR")


def _identity_of(path):
    """FRLL identity = filename prefix before the first underscore (172_03 -> 172)."""
    return os.path.splitext(os.path.basename(str(path)))[0].split("_")[0]


if __name__ == "__main__":
    with open("execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    channels = int(params["image_parameters"]["image_channels"])
    width = int(params["image_parameters"]["image_width"])
    height = int(params["image_parameters"]["image_height"])
    dataset_path = str(params["image_parameters"]["dataset_path"])

    dp = params["dataset_parameters"]
    train_dir = str(dp["train_dir"])
    val_dir = str(dp["val_dir"])
    val_split = float(dp["val_split"])
    seed = int(dp["seed"])

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    ### load source images (keeping paths so we can split by identity)
    image_loader = ImageLoader(channels=channels, base_path=dataset_path)
    items = image_loader.load_images_with_paths()
    print(f"Loaded {len(items)} images from {dataset_path}")

    ### leakage-free split: hold out whole identities for validation
    identities = sorted({_identity_of(p) for p, _ in items})
    rng = random.Random(seed)
    rng.shuffle(identities)
    n_val = max(1, int(round(len(identities) * val_split)))
    val_ids = set(identities[:n_val])
    print(f"Identities -> train: {len(identities) - n_val}, val: {n_val}")

    landmark_descriptor = Landmarks(
        number_of_landmarks=int(params["landmark_parameters"]["number_of_landmarks"]))

    counts = {"train": 0, "val": 0}
    for path, image in tqdm(items, desc="Landmarking", unit="img"):
        split = "val" if _identity_of(path) in val_ids else "train"
        out_dir = val_dir if split == "val" else train_dir
        try:
            _, landmarks = landmark_descriptor.generate_landmarks(
                image=image, channels=channels, width=width, height=height)

            counts[split] += 1
            idx = counts[split]

            resized = Image.fromarray(image).resize((width, height))
            resized.save(os.path.join(out_dir, f"image_{idx}.jpg"), format="JPEG")
            np.savetxt(os.path.join(out_dir, f"image_{idx}.csv"),
                       landmarks, delimiter=",", header="x,y", comments="",
                       fmt="%.4f")
        except Exception:                              # noqa: BLE001 - no face etc.
            print(f"no face detected for {os.path.basename(str(path))}")

    print(f"Done. Saved train: {counts['train']} -> {train_dir}, "
          f"val: {counts['val']} -> {val_dir}")
