"""Pluggable face-recognition "systems" used to score attack success.

Every system exposes the same interface: `embed(paths) -> (N, D) float32,
L2-normalized`. Two kinds of system are supported:

  * "facenet"  - always available. Runs the project's own FaceNet embedder
    (image_utils/face_embedding.py), the same identity anchor the morph
    generator's training loss uses. Works on ANY image path, including the
    freshly generated morphs, with no extra setup.

  * external (e.g. "adaface") - loaded from a precomputed .npz produced OUTSIDE
    this (TensorFlow) environment by a model this repo cannot run directly
    (ArcFace, ElasticFace, AdaFace, CurricularFace, ...). This mirrors the
    convention already used by ../main_syn_mad_pairs.py. An external system can
    only score paths present in its npz - see README.md "Adding a published FR
    system" for how to extend coverage to newly generated morphs.

Cosine similarity is used throughout since every embedding here is L2-normalized
(similarity == dot product).
"""
import os
import numpy as np

from evaluation.common import paths as P


def _load_npz_lookup(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    emb = data["embeddings"].astype(np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    return {os.path.abspath(str(p)): e for p, e in zip(data["paths"], emb)}


class FaceNetSystem:
    name = "facenet"
    #: FaceNet is the training-time identity anchor (see model/diffusion_model.py),
    #: so it can embed anything, but a paper should not claim this as an
    #: independent, published vulnerability benchmark on its own - see README.
    coverage = "universal"

    def embed(self, img_paths, size=224):
        from image_utils.face_embedding import embed_paths
        return embed_paths(list(img_paths), size)


class ExternalNpzSystem:
    """Wraps a precomputed embeddings file for a system this repo can't run."""

    coverage = "npz-only"

    def __init__(self, name, npz_path):
        self.name = name
        self._lookup = _load_npz_lookup(npz_path)

    def embed(self, img_paths):
        wanted = [os.path.abspath(p) for p in img_paths]
        missing = [p for p in wanted if p not in self._lookup]
        if missing:
            raise KeyError(
                f"{self.name}: {len(missing)}/{len(wanted)} paths have no "
                f"precomputed embedding (first missing: {missing[0]}). This "
                f"system can only score images it was exported for - see "
                f"README.md 'Adding a published FR system'.")
        return np.stack([self._lookup[p] for p in wanted]).astype(np.float32)

    def covers(self, img_paths):
        """True if every path already has a precomputed embedding."""
        wanted = (os.path.abspath(p) for p in img_paths)
        return all(p in self._lookup for p in wanted)


def available_systems():
    """{name: system} for FaceNet plus every external .npz found in cache/."""
    systems = {"facenet": FaceNetSystem()}
    registered_paths = set()
    for name, npz_path in P.KNOWN_EXTERNAL_EMBEDDINGS.items():
        if os.path.exists(npz_path):
            systems[name] = ExternalNpzSystem(name, npz_path)
            registered_paths.add(os.path.abspath(npz_path))
    # also pick up any cache/<name>.npz not already registered (by path, so the
    # same file isn't loaded twice under two different names).
    if os.path.isdir(P.EXTERNAL_EMBEDDINGS_DIR):
        for fname in os.listdir(P.EXTERNAL_EMBEDDINGS_DIR):
            if not fname.endswith(".npz") or fname in ("train.npz", "val.npz"):
                continue
            full_path = os.path.abspath(os.path.join(P.EXTERNAL_EMBEDDINGS_DIR, fname))
            if full_path in registered_paths:
                continue
            name = os.path.splitext(fname)[0]
            if name not in systems:
                systems[name] = ExternalNpzSystem(name, full_path)
                registered_paths.add(full_path)
    return systems


def cosine_matrix(emb_a, emb_b):
    """emb_a: (N, D), emb_b: (M, D), both L2-normalized -> (N, M) cosine sims."""
    return emb_a @ emb_b.T
