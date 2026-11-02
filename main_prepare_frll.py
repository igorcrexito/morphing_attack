"""Landmark the FRLL frontal set while PRESERVING subject identity.

The original ``main_save_landmarks.py`` renames every image to ``image_<counter>``
and throws the identity prefix away, which makes it impossible to (a) forbid
same-identity morph pairs and (b) verify a morph against a *held-out* photo of the
same subject. Both are required by the SYN-MAD 2022 / DCMorph evaluation protocol.

FRLL frontal filenames are ``<id>_03.jpg`` (neutral) and ``<id>_08.jpg`` (smiling),
with ``<id>`` a zero-padded numeric subject id (001..102). Because the ids are
numeric and unique we keep the ``image_<n>.jpg`` convention used by the rest of the
pipeline but set ``n = int(id)``, so the number in a filename *is* the identity and
``morph_<a>_<b>.jpg`` names the two contributing subjects.

Outputs:

    output_dataset/frll/image_<id>.jpg + image_<id>.csv    # neutral -> morph parents
    output_dataset/frll/index.csv                          # id, gender, expression, source
    output_dataset/frll_probe/image_<id>.jpg + .csv        # smiling -> held-out probes

    python main_prepare_frll.py
"""
import csv
import os

import numpy as np
import tensorflow as tf
import yaml
from PIL import Image
from tqdm import tqdm

from landmarks.landmark import Landmarks

tf.get_logger().setLevel("ERROR")

FRLL_ROOT = "/home/igor/datasets/FRLL"
NEUTRAL_DIR = os.path.join(FRLL_ROOT, "neutral_front", "neutral_front")
SMILING_DIR = os.path.join(FRLL_ROOT, "smiling_front", "smiling_front")
INFO_CSV = os.path.join(FRLL_ROOT, "london_faces_info.csv")

OUT_MORPH_DIR = os.path.join("output_dataset", "frll")        # parents (neutral)
OUT_PROBE_DIR = os.path.join("output_dataset", "frll_probe")  # probes (smiling)


def _load_gender():
    """face_id -> gender ('female'/'male') from london_faces_info.csv."""
    with open(INFO_CSV, newline="") as f:
        return {row["face_id"].strip(): row["face_gender"].strip()
                for row in csv.DictReader(f)}


def _identity_of(filename):
    """'001_03.jpg' -> '001'."""
    return os.path.splitext(os.path.basename(filename))[0].split("_")[0]


def _landmark_dir(src_dir, out_dir, descriptor, channels, width, height):
    """Landmark every jpg in src_dir into out_dir as image_<int(identity)>.*.

    Returns {identity: source_filename} for the images that succeeded.
    """
    os.makedirs(out_dir, exist_ok=True)
    kept = {}
    files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".jpg"))
    for name in tqdm(files, desc=f"Landmarking {os.path.basename(out_dir)}",
                     unit="img"):
        identity = _identity_of(name)
        image = np.array(Image.open(os.path.join(src_dir, name)).convert("RGB"))
        try:
            out_image, landmarks = descriptor.generate_landmarks(
                image=image, channels=channels, width=width, height=height)
        except Exception:                                  # noqa: BLE001 - no face
            print(f"no face detected for {name}")
            continue

        stem = os.path.join(out_dir, f"image_{int(identity)}")
        out_image.save(stem + ".jpg", format="JPEG")
        np.savetxt(stem + ".csv", landmarks, delimiter=",", header="x,y",
                   comments="", fmt="%.4f")
        kept[identity] = name
    return kept


def main():
    with open("execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    channels = int(params["image_parameters"]["image_channels"])
    width = int(params["image_parameters"]["image_width"])
    height = int(params["image_parameters"]["image_height"])
    quality = params.get("quality_parameters", {}) or {}

    descriptor = Landmarks(
        number_of_landmarks=int(params["landmark_parameters"]["number_of_landmarks"]),
        align=bool(quality.get("align_faces", False)))

    gender = _load_gender()

    neutral = _landmark_dir(NEUTRAL_DIR, OUT_MORPH_DIR, descriptor,
                            channels, width, height)
    smiling = _landmark_dir(SMILING_DIR, OUT_PROBE_DIR, descriptor,
                            channels, width, height)

    # index.csv describes the morph-parent set. `has_probe` marks the subjects for
    # which a held-out (smiling) capture is available for uncorrelated verification.
    index_path = os.path.join(OUT_MORPH_DIR, "index.csv")
    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "identity", "gender", "expression",
                         "source", "has_probe"])
        for identity in sorted(neutral, key=int):
            writer.writerow([f"image_{int(identity)}.jpg", identity,
                             gender.get(identity, "unknown"), "neutral",
                             neutral[identity], int(identity in smiling)])

    n_probe = sum(1 for i in neutral if i in smiling)
    print(f"\nParents (neutral): {len(neutral)}/102 -> {OUT_MORPH_DIR}")
    print(f"Probes  (smiling): {len(smiling)}/102 -> {OUT_PROBE_DIR}")
    print(f"Subjects with both parent and held-out probe: {n_probe}")
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
