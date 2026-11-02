"""Landmark the Utrecht (ECVP) face set while PRESERVING subject identity,
exactly mirroring main_prepare_frll.py so the SAME evaluation suite
(evaluation/01_calibrate_thresholds.py, 02_vulnerability_eval.py, ...) can be
pointed at this dataset with zero changes.

Utrecht raw filenames (/home/igor/datasets/utrecht) are ``<gender><id>.jpg``
(neutral, e.g. "f4001.jpg", "m4012.jpg") and ``<gender><id>s.jpg`` (smiling,
e.g. "f4001s.jpg"). Unlike FRLL's purely numeric ids, the identity string here
mixes a gender letter and a number, so a synthetic sequential integer id is
assigned per unique identity (recorded in index.csv alongside the original
string) - the rest of the pipeline only ever sees ``image_<int>.jpg``.

A few identities (e.g. "f4007_1.jpg" / "f4007_2.jpg") have no plain
"<identity>.jpg" neutral capture; the first numbered variant is used as the
neutral image for those.

Outputs (same shape as output_dataset/frll + frll_probe):

    output_dataset/utrecht/image_<n>.jpg + .csv        # neutral -> morph parents
    output_dataset/utrecht/index.csv                   # n, identity_str, gender, expression, source, has_probe
    output_dataset/utrecht_probe/image_<n>.jpg + .csv  # smiling -> held-out probes

    python main_prepare_utrecht.py
"""
import csv
import glob
import os
import re

import numpy as np
import tensorflow as tf
import yaml
from PIL import Image
from tqdm import tqdm

from landmarks.landmark import Landmarks

tf.get_logger().setLevel("ERROR")

UTRECHT_ROOT = "/home/igor/datasets/utrecht"
OUT_MORPH_DIR = os.path.join("output_dataset", "utrecht")        # parents (neutral)
OUT_PROBE_DIR = os.path.join("output_dataset", "utrecht_probe")  # probes (smiling)

_IDENTITY_RE = re.compile(r"^([fm]\d+)(?:_(\d+))?(s)?\.jpg$", re.IGNORECASE)


def _scan_identities():
    """{identity_str: {"neutral": filename or None, "smiling": filename or None}}"""
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(UTRECHT_ROOT, "*.jpg")))
    subjects = {}
    for name in files:
        m = _IDENTITY_RE.match(name)
        if not m:
            print(f"[skip] unrecognized filename pattern: {name}")
            continue
        identity, variant, smiling = m.groups()
        entry = subjects.setdefault(identity, {"neutral": None, "neutral_variant": None,
                                               "smiling": None})
        if smiling:
            entry["smiling"] = name
        elif variant is None:
            entry["neutral"] = name
        else:
            # e.g. f4007_1.jpg - only used if no plain "<identity>.jpg" exists
            if entry["neutral_variant"] is None:
                entry["neutral_variant"] = name
    for identity, entry in subjects.items():
        if entry["neutral"] is None:
            entry["neutral"] = entry["neutral_variant"]
    return subjects


def _landmark_one(descriptor, src_path, out_stem, channels, width, height):
    image = np.array(Image.open(src_path).convert("RGB"))
    out_image, landmarks = descriptor.generate_landmarks(
        image=image, channels=channels, width=width, height=height)
    out_image.save(out_stem + ".jpg", format="JPEG")
    np.savetxt(out_stem + ".csv", landmarks, delimiter=",", header="x,y",
              comments="", fmt="%.4f")


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

    os.makedirs(OUT_MORPH_DIR, exist_ok=True)
    os.makedirs(OUT_PROBE_DIR, exist_ok=True)

    subjects = _scan_identities()
    print(f"Found {len(subjects)} unique identities in {UTRECHT_ROOT}")

    rows = []
    n_probe = 0
    for i, identity in enumerate(sorted(subjects), start=1):
        entry = subjects[identity]
        gender = "female" if identity.startswith("f") else "male"

        if entry["neutral"] is None:
            print(f"[skip] {identity}: no neutral capture at all")
            continue
        try:
            _landmark_one(descriptor, os.path.join(UTRECHT_ROOT, entry["neutral"]),
                         os.path.join(OUT_MORPH_DIR, f"image_{i}"), channels, width, height)
        except Exception as e:                              # noqa: BLE001 - no face
            print(f"[skip] {identity}: no face detected in neutral ({e})")
            continue

        has_probe = 0
        if entry["smiling"] is not None:
            try:
                _landmark_one(descriptor, os.path.join(UTRECHT_ROOT, entry["smiling"]),
                             os.path.join(OUT_PROBE_DIR, f"image_{i}"), channels, width, height)
                has_probe = 1
                n_probe += 1
            except Exception as e:                          # noqa: BLE001 - no face
                print(f"[note] {identity}: neutral OK, smiling probe failed ({e})")

        rows.append([f"image_{i}.jpg", i, identity, gender, "neutral",
                     entry["neutral"], has_probe])

    index_path = os.path.join(OUT_MORPH_DIR, "index.csv")
    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "identity", "identity_str", "gender",
                         "expression", "source", "has_probe"])
        writer.writerows(rows)

    print(f"\nParents (neutral): {len(rows)} -> {OUT_MORPH_DIR}")
    print(f"Probes  (smiling): {n_probe} -> {OUT_PROBE_DIR}")
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
