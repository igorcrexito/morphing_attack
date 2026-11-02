"""evaluation/export_arcface_embeddings.py

Exports ArcFace embeddings (insightface's "buffalo_l" pack - the standard
ArcFace-R50 recognition model widely used in the morphing-attack literature,
e.g. SYN-MAD 2022 / DCMorph-style benchmarks) for every image the evaluation
suite needs: the bona fide FRLL/FRLL-probe photos AND the morphs from every
pipeline variant (original, balanced, balanced+poseblend, ablations, ...).

*** MUST BE RUN WITH THE SEPARATE `arcface_env` VIRTUALENV, NOT `morphing_env` ***

Why a separate venv: insightface depends on onnxruntime, which (at the time
this was set up) pulls in numpy>=2.0 and protobuf>=5, both INCOMPATIBLE with
this project's TensorFlow 2.15 (needs numpy<2.0, protobuf<4.21). Installing
insightface into morphing_env broke TensorFlow entirely on the first attempt -
do not repeat that. `arcface_env` is a throwaway, isolated venv used only to
run this one script; nothing else in the project should ever import from it.

Setup (already done once, kept here for reference / re-creation):
    python3 -m venv arcface_env
    source arcface_env/bin/activate
    pip install onnxruntime insightface numpy pillow

Usage (always with arcface_env active, run from the PROJECT ROOT):
    source arcface_env/bin/activate
    python3 evaluation/export_arcface_embeddings.py
    deactivate   # then go back to `source morphing_env/bin/activate` for everything else

Output: cache/arcface.npz with 'paths' (absolute image paths) and
'embeddings' (Nx512 float32, L2-normalized) - the exact format
evaluation/common/fr_systems.py already knows how to load automatically for
ANY cache/*.npz file. No other evaluation script needs to change: once this
file exists, 01_calibrate_thresholds.py and 02_vulnerability_eval.py pick up
an "arcface" system on their own next run.

Safe to re-run at any time (e.g. after more morphs have been generated): it
recomputes embeddings for every currently-existing image under the configured
directories and overwrites cache/arcface.npz with the full, current set.
"""
import glob
import os
import sys

import numpy as np
import cv2

# Directories to embed. Add a new pipeline variant's output folder here and
# re-run this script to make it show up in evaluation/02_vulnerability_eval.py
# automatically (as long as it's a cache/*.npz-backed system, no other code
# changes needed - see common/fr_systems.py).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGE_DIRS = [
    os.path.join(PROJECT_ROOT, "output_dataset", "frll"),
    os.path.join(PROJECT_ROOT, "output_dataset", "frll_probe"),
    os.path.join(PROJECT_ROOT, "morphed_dataset", "frll_synmad", "neutral"),
    os.path.join(PROJECT_ROOT, "morphed_dataset", "frll_synmad", "smiling"),
    os.path.join(PROJECT_ROOT, "morphed_dataset_balanced", "frll_synmad", "neutral"),
    os.path.join(PROJECT_ROOT, "morphed_dataset_balanced", "frll_synmad", "smiling"),
    os.path.join(PROJECT_ROOT, "morphed_dataset_balanced", "frll_synmad_poseblend", "neutral"),
    os.path.join(PROJECT_ROOT, "morphed_dataset_balanced", "frll_synmad_poseblend", "smiling"),
    # official_pipeline's promoted output (fixes #1-3 + pose-blend) - the
    # canonical dataset to report going forward:
    os.path.join(PROJECT_ROOT, "morphed_dataset_official", "frll_synmad", "neutral"),
    os.path.join(PROJECT_ROOT, "morphed_dataset_official", "frll_synmad", "smiling"),
]
OUT_PATH = os.path.join(PROJECT_ROOT, "cache", "arcface.npz")


def _collect_paths():
    paths = []
    for d in IMAGE_DIRS:
        if os.path.isdir(d):
            paths += sorted(glob.glob(os.path.join(d, "*.jpg")))
        else:
            print(f"[skip] directory not found (yet?): {d}")
    return paths


def main():
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise SystemExit(
            "insightface not importable - this script must be run with "
            "arcface_env active, not morphing_env:\n"
            "    source arcface_env/bin/activate\n"
            "    python3 evaluation/export_arcface_embeddings.py")

    paths = _collect_paths()
    print(f"Found {len(paths)} images to embed across {len(IMAGE_DIRS)} directories.")
    if not paths:
        raise SystemExit("Nothing to embed - check IMAGE_DIRS in this script.")

    # buffalo_l = insightface's standard pack: RetinaFace detector + ArcFace
    # (w600k_r50) recognizer. CPU-only provider since this machine has no GPU
    # runtime configured for onnxruntime.
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(224, 224))

    kept_paths, embeddings = [], []
    no_face, multi_face = 0, 0
    for i, p in enumerate(paths):
        img = cv2.imread(p)  # BGR, as insightface expects
        if img is None:
            print(f"[skip] unreadable image: {p}")
            continue
        faces = app.get(img)
        if len(faces) == 0:
            no_face += 1
            print(f"[skip] no face detected: {p}")
            continue
        if len(faces) > 1:
            multi_face += 1  # keep the largest/first detection, insightface
            # already sorts by detection score in most versions; harmless
            # either way since these are single-subject portrait crops.
        kept_paths.append(os.path.abspath(p))
        embeddings.append(faces[0].normed_embedding.astype(np.float32))

        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(paths)} embedded")

    embeddings = np.stack(embeddings).astype(np.float32)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(OUT_PATH, paths=np.array(kept_paths), embeddings=embeddings)

    print(f"\nWrote {len(kept_paths)} embeddings to {OUT_PATH}")
    print(f"  skipped: {no_face} no-face, {len(paths) - len(kept_paths) - no_face} "
          f"unreadable | {multi_face} images had >1 face detected (kept top match)")


if __name__ == "__main__":
    main()
