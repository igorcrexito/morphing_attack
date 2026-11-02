"""evaluation/utrecht_mmpmr.py

MMPMR on the Utrecht (ECVP) dataset, scored against ArcFace at the LFW-
calibrated threshold (cache/lfw_arcface_thresholds.json - run
calibrate_lfw_threshold.py first). Separate from 02_vulnerability_eval.py
because Utrecht's pairing (pairs_utrecht.csv, built by a one-off top-150-by-
similarity ranking - see main_prepare_utrecht.py and the project root's
generate_utrecht_official.py) isn't the SYN-MAD 2022 protocol, so it's kept
out of the main evaluation script's assumptions (expression-split directory
layout, thresholds.json's FRLL-based calibration, etc.).

IMPORTANT CAVEAT: pairs_utrecht.csv keeps only the top 150 of 1596 possible
pairs among the 57 Utrecht subjects with both a neutral and a smiling capture
(~9.4% most-similar retained) - much more selective than SYN-MAD 2022's ~77%
retention on FRLL. More-similar identity pairs are inherently easier morphing
targets, so Utrecht's MMPMR is NOT directly comparable to the FRLL number
without accounting for this - see evaluation/README.md's Utrecht section.

*** MUST BE RUN WITH arcface_env, NOT morphing_env ***

Usage:
    source arcface_env/bin/activate
    python3 evaluation/utrecht_mmpmr.py

Output: evaluation/artifacts/utrecht_mmpmr_result.json
"""
import csv, os, json
import numpy as np
import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(224, 224))

def embed(path):
    img = cv2.imread(path)
    if img is None:
        return None
    faces = app.get(img)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding.astype(np.float32)

with open("pairs_utrecht.csv", newline="") as f:
    pairs = list(csv.DictReader(f))

with open("cache/lfw_arcface_thresholds.json") as f:
    lfw_thr = json.load(f)["fmr_1pct"]

rows = []
for r in pairs:
    a, b = int(r["id_a"]), int(r["id_b"])
    morph_path = f"morphed_dataset_official/utrecht/morph_{a}_{b}.jpg"
    proba_path = f"output_dataset/utrecht_probe/image_{a}.jpg"
    probb_path = f"output_dataset/utrecht_probe/image_{b}.jpg"
    if not (os.path.exists(morph_path) and os.path.exists(proba_path) and os.path.exists(probb_path)):
        continue
    em = embed(morph_path)
    ea = embed(proba_path)
    eb = embed(probb_path)
    if em is None or ea is None or eb is None:
        continue
    sim_a = float(np.dot(em, ea))
    sim_b = float(np.dot(em, eb))
    rows.append({"id_a": a, "id_b": b, "sim_a": sim_a, "sim_b": sim_b, "worst": min(sim_a, sim_b)})

n = len(rows)
successes = sum(1 for r in rows if r["worst"] >= lfw_thr)
mmpmr = successes / n if n else float("nan")
mean_a = np.mean([r["sim_a"] for r in rows])
mean_b = np.mean([r["sim_b"] for r in rows])
gap = mean_a - mean_b

print(f"n={n}  MMPMR@LFW-thr={mmpmr:.4f}  sim_A={mean_a:.3f}  sim_B={mean_b:.3f}  gap={gap:.3f}")
os.makedirs("evaluation/artifacts", exist_ok=True)
with open("evaluation/artifacts/utrecht_mmpmr_result.json", "w") as f:
    json.dump({"n": n, "mmpmr": mmpmr, "sim_a": mean_a, "sim_b": mean_b, "gap": gap,
               "lfw_threshold": lfw_thr, "rows": rows}, f, indent=2)
