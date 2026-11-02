"""Build the SYN-MAD 2022 morph pair list on FRLL.

Reproduces the pair-selection protocol used by SYN-MAD 2022 and followed by
DCMorph (CVPRW 2026), so that MMPMR numbers computed on the resulting morphs are
directly comparable to the published tables:

  * restrict to the FRLL *frontal* images,
  * split them by **gender x expression** -> 4 groups
    (female-neutral, female-smiling, male-neutral, male-smiling),
  * embed every image with a face-recognition model and score all within-group
    pairs by cosine similarity,
  * keep the **250 most similar pairs per group** -> 1000 morph pairs.

Same-identity pairs are impossible by construction (one image per subject per
expression), which is exactly the bug this replaces in ``_select_partner``.

The paper ranks with ElasticFace-Arc. That model is not available in this
TensorFlow environment, so by default we rank with the project's FaceNet
embedder and *report* the deviation. To rank with a margin-loss FR model
instead, precompute embeddings elsewhere and pass them in:

    # in face_morph_verification (torch venv):

    python export_embeddings.py \
        --images <morphing_attack>/output_dataset/frll \
                 <morphing_attack>/output_dataset/frll_probe \
        --model adaface_ir101_wf12m --out /tmp/frll_adaface.npz

    python main_syn_mad_pairs.py --embeddings /tmp/frll_adaface.npz

Output: ``pairs_synmad.csv`` with one row per morph pair:

    group,expression,gender,id_a,id_b,similarity,shape_distance,rank
"""
import argparse
import csv
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from morphing.warp import shape_distance
from landmarks.landmark import Landmarks

tf.get_logger().setLevel("ERROR")

INDEX_CSV = os.path.join("output_dataset", "frll", "index.csv")
DIR_BY_EXPRESSION = {
    "neutral": os.path.join("output_dataset", "frll"),
    "smiling": os.path.join("output_dataset", "frll_probe"),
}
PAIRS_PER_GROUP = 250


def _read_index():
    """[(identity, gender)] for every landmarked FRLL subject, sorted by id."""
    with open(INDEX_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    return [(r["identity"], r["gender"]) for r in sorted(rows, key=lambda r: int(r["identity"]))]


def _paths(identities, expression):
    d = DIR_BY_EXPRESSION[expression]
    return [os.path.join(d, f"image_{int(i)}.jpg") for i in identities]


def _load_landmarks(jpg_path):
    csv_path = os.path.splitext(jpg_path)[0] + ".csv"
    return np.loadtxt(csv_path, delimiter=",", skiprows=1).astype(np.float32)


def _embed(paths, size, external):
    """L2-normalized embeddings for `paths`, from an npz or from FaceNet."""
    if external is not None:
        data = np.load(external, allow_pickle=True)
        # keyed by absolute path: the neutral and smiling dirs reuse the same
        # image_<n>.jpg names, so basenames alone would collide.
        lookup = {os.path.abspath(str(k)): v
                  for k, v in zip(data["paths"], data["embeddings"])}
        wanted = [os.path.abspath(p) for p in paths]
        missing = [p for p in wanted if p not in lookup]
        if missing:
            raise SystemExit(f"{len(missing)} paths missing from {external}, "
                             f"e.g. {missing[0]}")
        emb = np.stack([lookup[p] for p in wanted]).astype(np.float32)
        return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    from image_utils.face_embedding import embed_paths
    return embed_paths(paths, size)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="pairs_synmad.csv")
    ap.add_argument("--pairs-per-group", type=int, default=PAIRS_PER_GROUP)
    ap.add_argument("--size", type=int, default=224,
                    help="resize used when embedding with FaceNet")
    ap.add_argument("--embeddings", default=None,
                    help="npz with 'paths' and 'embeddings' arrays (e.g. AdaFace/"
                         "ElasticFace), used instead of FaceNet for pair ranking")
    args = ap.parse_args()

    index = _read_index()
    print(f"{len(index)} FRLL subjects from {INDEX_CSV}")

    rows = []
    for expression in ("neutral", "smiling"):
        for gender in ("female", "male"):
            members = [i for i, g in index if g == gender]
            paths = _paths(members, expression)
            keep = [(m, p) for m, p in zip(members, paths) if os.path.exists(p)]
            members = [m for m, _ in keep]
            paths = [p for _, p in keep]

            emb = _embed(paths, args.size, args.embeddings)
            landmarks = [_load_landmarks(p) for p in paths]
            sims = emb @ emb.T

            n = len(members)
            candidates = [(float(sims[a, b]), a, b)
                          for a in range(n) for b in range(a + 1, n)]
            candidates.sort(reverse=True)
            chosen = candidates[:args.pairs_per_group]

            group = f"{gender}-{expression}"
            print(f"{group:16s}: {n:3d} subjects, {len(candidates):5d} possible "
                  f"pairs -> keeping {len(chosen)} "
                  f"(cos {chosen[-1][0]:.3f}..{chosen[0][0]:.3f})")
            if len(chosen) < args.pairs_per_group:
                print(f"  NOTE: only {len(chosen)} pairs available in this group "
                      f"(FRLL has fewer subjects per gender than requested).")

            for rank, (sim, a, b) in enumerate(chosen, start=1):
                rows.append({
                    "group": group, "expression": expression, "gender": gender,
                    "id_a": members[a], "id_b": members[b],
                    "similarity": round(sim, 4),
                    "shape_distance": round(
                        float(shape_distance(landmarks[a], landmarks[b])), 4),
                    "rank": rank,
                })

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sd = np.array([r["shape_distance"] for r in rows])
    print(f"\nWrote {len(rows)} pairs to {args.out}")
    print(f"Ranked with: {'external ' + args.embeddings if args.embeddings else 'FaceNet (deviates from the paper, which uses ElasticFace-Arc)'}")
    print(f"shape_distance over the chosen pairs: median {np.median(sd):.3f}, "
          f"p90 {np.percentile(sd, 90):.3f}, max {sd.max():.3f}")
    for t in (0.09, 0.12, 0.15, 0.20):
        print(f"  would survive shape_threshold={t}: {100 * (sd <= t).mean():.1f}%")


if __name__ == "__main__":
    main()
