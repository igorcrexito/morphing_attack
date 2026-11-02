"""evaluation/calibrate_lfw_threshold.py

Calibrates ArcFace's FMR-1%/FMR-0.1% decision thresholds on the STANDARD LFW
verification protocol (the 10-fold, 6000-pair "View 2" split - 3000 genuine +
3000 impostor pairs), the exact benchmark the SYN-MAD 2022 / DCMorph papers
use to set their own thresholds (see the paper excerpt quoted in the
evaluation chat history: "The FMR1000 and FMR100 decision thresholds used in
the vulnerability study were calculated on the LFW benchmark").

This exists because 01_calibrate_thresholds.py calibrates on FRLL instead
(the same dataset the morphs are built from) - a smaller, more homogeneous
set than LFW. The two thresholds differ (FRLL: 0.1722 @ FMR1%, LFW: 0.1589 @
FMR1%, about 7.7% lower) because LFW is a larger, more "in the wild" dataset
with a slightly heavier impostor-score tail. Neither is "wrong" - FRLL
calibration is arguably more relevant to THIS specific benchmark (same
population as the morphs), while LFW calibration is what makes the resulting
MMPMR numbers directly comparable to the literature's own reported figures.
Both are reported side by side in evaluation/artifacts/FINAL_REPORT.md.

*** MUST BE RUN WITH arcface_env, NOT morphing_env *** (same reason as
export_arcface_embeddings.py - insightface's onnxruntime dependency conflicts
with this project's TensorFlow). Also needs scikit-learn (for the LFW
fetcher): `pip install scikit-learn` inside arcface_env.

Usage:
    source arcface_env/bin/activate
    python3 evaluation/calibrate_lfw_threshold.py

Output: cache/lfw_arcface_thresholds.json (kept separate from
artifacts/thresholds.json, which holds the FRLL-calibrated numbers used by
default everywhere else in this suite).

Downloads ~230MB of LFW images to ~/scikit_learn_data on first run (cached
after that).
"""
import numpy as np
from sklearn.datasets import fetch_lfw_pairs
from insightface.app import FaceAnalysis
import cv2, json

data = fetch_lfw_pairs(subset='10_folds', color=True, resize=1.0, download_if_missing=True)
pairs = data.pairs  # (6000, 2, 125, 94, 3), float in [0,1]
target = data.target  # 1 = same person, 0 = different

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(160, 160))

def embed(img_float):
    img = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # upscale a bit - LFW pair crops are small (125x94), helps detection
    img_bgr = cv2.resize(img_bgr, (250, 188), interpolation=cv2.INTER_CUBIC)
    faces = app.get(img_bgr)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding.astype(np.float32)

genuine, impostor = [], []
no_face = 0
for i in range(len(pairs)):
    e1 = embed(pairs[i, 0])
    e2 = embed(pairs[i, 1])
    if e1 is None or e2 is None:
        no_face += 1
        continue
    sim = float(np.dot(e1, e2))
    if target[i] == 1:
        genuine.append(sim)
    else:
        impostor.append(sim)
    if (i + 1) % 500 == 0:
        print(f"  ... {i+1}/{len(pairs)} processed, no_face={no_face}")

genuine = np.array(genuine)
impostor = np.array(impostor)
print(f"genuine n={len(genuine)}  impostor n={len(impostor)}  no_face={no_face}")

def calibrate(genuine, impostor, target_fmr):
    imp_sorted = np.sort(impostor)[::-1]
    k = int(round(target_fmr * len(imp_sorted)))
    k = min(max(k, 0), len(imp_sorted) - 1)
    return float(imp_sorted[k])

thr_1pct = calibrate(genuine, impostor, 0.01)
thr_01pct = calibrate(genuine, impostor, 0.001)
print(f"LFW ArcFace threshold @ FMR1%: {thr_1pct:.4f}")
print(f"LFW ArcFace threshold @ FMR0.1%: {thr_01pct:.4f}")

with open("cache/lfw_arcface_thresholds.json", "w") as f:
    json.dump({"fmr_1pct": thr_1pct, "fmr_0.1pct": thr_01pct,
               "n_genuine": len(genuine), "n_impostor": len(impostor),
               "genuine_mean": float(genuine.mean()), "impostor_mean": float(impostor.mean())}, f, indent=2)
print("saved to cache/lfw_arcface_thresholds.json")
