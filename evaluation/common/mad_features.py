"""A self-contained, statistical S-MAD (single-image morph detection) baseline.

Design note (read this before trusting the numbers): the literature review
(morphing_literature_summaries.pdf, "Batskos, IWBF 2025") describes a strong,
simple published detector that re-compresses the questioned image at several
JPEG qualities, looks at how the compression error differs from a bona fide
photo, and feeds seven such statistics into an SVM (~5% EER on the BOEP
benchmark). That detector's exact feature set and trained weights are not
published/available in this environment, so this module is an independent,
SIMPLIFIED re-implementation of the same idea, not a reproduction of that paper's
numbers. It exists so this repo has a real, honestly-labeled, fully local
detector to evaluate against, instead of no detectability number at all.

Do not report this baseline's EER as "the field's MAD performance" in a paper -
report it as "a compression-error statistical baseline trained on our own
generated data", and, where possible, additionally submit morphs to a public
platform (BOEP / FATE MORPH) for a citable, independently-run number - see
README.md Section 4.
"""
import cv2
import numpy as np

from evaluation.common.degrade import jpeg_recompress

# Qualities the questioned image is re-compressed at; the response to
# progressively lossy re-encoding differs between a genuine photo (compressed
# once, by the camera/print pipeline) and a morph (already carrying blending /
# refinement-network artifacts that interact with quantization differently).
_QUALITIES = (90, 75, 60, 45, 30)


def _diff_stats(img_rgb01, recompressed_rgb01):
    diff = np.abs(img_rgb01 - recompressed_rgb01)
    gray_diff = cv2.cvtColor(diff.astype(np.float32), cv2.COLOR_RGB2GRAY)
    return float(gray_diff.mean()), float(gray_diff.std())


def compression_error_features(img_rgb01):
    """Seven-ish statistics of how the image responds to repeated JPEG loss.

    Returns a fixed-length 1-D float32 vector: for each quality in
    `_QUALITIES`, the mean and std of the per-pixel recompression error,
    plus the mean absolute difference between consecutive quality levels
    (how fast the error grows as quality drops).
    """
    img_rgb01 = np.clip(img_rgb01, 0.0, 1.0).astype(np.float32)
    means, stds = [], []
    for q in _QUALITIES:
        recompressed = jpeg_recompress(img_rgb01, quality=q)
        m, s = _diff_stats(img_rgb01, recompressed)
        means.append(m)
        stds.append(s)
    means = np.array(means, dtype=np.float32)
    stds = np.array(stds, dtype=np.float32)
    growth = np.diff(means)  # how much error increases as quality drops further

    # a simple high-frequency-energy feature via FFT, since generative/blending
    # artifacts often show up as unnatural spectral energy (see SelfMAD++ notes
    # in the literature review: "frequency-artifact generator ... FFT magnitude").
    gray = cv2.cvtColor(img_rgb01, cv2.COLOR_RGB2GRAY)
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    low = mag[cy - r:cy + r, cx - r:cx + r].mean()
    high = (mag.sum() - mag[cy - r:cy + r, cx - r:cx + r].sum()) / (mag.size - (2 * r) ** 2 + 1e-8)
    hf_ratio = float(high / (low + 1e-8))

    return np.concatenate([means, stds, growth, [hf_ratio]]).astype(np.float32)


FEATURE_NAMES = (
    [f"mean_err_q{q}" for q in _QUALITIES] +
    [f"std_err_q{q}" for q in _QUALITIES] +
    [f"err_growth_{i}" for i in range(len(_QUALITIES) - 1)] +
    ["hf_lf_ratio"]
)
