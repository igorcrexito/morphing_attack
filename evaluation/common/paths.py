"""Central path configuration for the evaluation suite.

Every script in evaluation/ imports from here instead of hard-coding paths, so
the whole suite can be pointed at a different dataset/output layout by editing
one file. Paths are resolved relative to the project root (the parent of this
evaluation/ folder), so the scripts work regardless of the current working
directory they are launched from.
"""
import os

# evaluation/common/paths.py -> evaluation/common -> evaluation -> <project root>
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL_ROOT = os.path.join(PROJECT_ROOT, "evaluation")
ARTIFACTS_DIR = os.path.join(EVAL_ROOT, "artifacts")

# ---------------------------------------------------------------- source data
# FRLL, landmarked at model resolution by main_save_landmarks.py / main_prepare_frll.py.
# "frll" = neutral expression captures, "frll_probe" = smiling expression captures of
# the SAME 102 subjects (image_<id>.jpg is the same identity in both directories).
FRLL_NEUTRAL_DIR = os.path.join(PROJECT_ROOT, "output_dataset", "frll")
FRLL_SMILING_DIR = os.path.join(PROJECT_ROOT, "output_dataset", "frll_probe")
FRLL_INDEX_CSV = os.path.join(FRLL_NEUTRAL_DIR, "index.csv")

EXPRESSION_DIRS = {
    "neutral": FRLL_NEUTRAL_DIR,
    "smiling": FRLL_SMILING_DIR,
}
# For a morph built from expression E, the "other" expression's captures of the
# same two subjects are genuine probes never seen during morph construction.
OTHER_EXPRESSION = {"neutral": "smiling", "smiling": "neutral"}

# ---------------------------------------------------------------- pairing / morphs
PAIRS_CSV = os.path.join(PROJECT_ROOT, "pairs_synmad.csv")
SYNMAD_MORPH_ROOT = os.path.join(PROJECT_ROOT, "morphed_dataset", "frll_synmad")

# ---------------------------------------------------------------- model weights
MORPH_WEIGHTS = os.path.join(PROJECT_ROOT, "cache", "morph_model.weights.h5")

# ---------------------------------------------------------------- external embeddings
# Optional precomputed embeddings for FR systems this (TensorFlow) environment
# cannot run natively (ArcFace, ElasticFace, AdaFace, CurricularFace, ...). Each
# file is an .npz with 'paths' (absolute image paths) and 'embeddings' (NxD,
# L2-normalized), exactly the convention already used by main_syn_mad_pairs.py.
# Drop additional files here named "<system>.npz" (e.g. "elasticface.npz") and
# they are picked up automatically - see common/fr_systems.py and the README.
EXTERNAL_EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "cache")
KNOWN_EXTERNAL_EMBEDDINGS = {
    "adaface": os.path.join(EXTERNAL_EMBEDDINGS_DIR, "frll_adaface.npz"),
}


def ensure_artifacts_dir(*subdirs):
    """mkdir -p evaluation/artifacts/<subdirs...> and return the path."""
    path = os.path.join(ARTIFACTS_DIR, *subdirs)
    os.makedirs(path, exist_ok=True)
    return path
