"""Metric definitions shared by every evaluation script.

Kept deliberately dependency-light (numpy + optionally sklearn for the ROC
curve) and each function is a direct, literal implementation of the definition
given in evaluation/README.md, so the README and the code cannot drift apart.
"""
import numpy as np


def calibrate_threshold(genuine_scores, impostor_scores, target_fmr):
    """Similarity threshold at which the *impostor* accept rate equals target_fmr.

    This is exactly how a deployed face-recognition system's operating point is
    set: pick the strictest threshold such that only `target_fmr` fraction of
    genuinely-different-people comparisons would still be (wrongly) accepted.
    Calibrated on bona fide genuine/impostor pairs only - the morphs are never
    involved in choosing the threshold, so the vulnerability numbers computed
    against it are not self-fulfilling.
    """
    impostor_scores = np.sort(np.asarray(impostor_scores))[::-1]  # descending
    if len(impostor_scores) == 0:
        raise ValueError("no impostor scores to calibrate against")
    k = int(round(target_fmr * len(impostor_scores)))
    k = min(max(k, 0), len(impostor_scores) - 1)
    return float(impostor_scores[k])


def mmpmr(scores_a, scores_b, threshold):
    """Mated Morph Presentation Match Rate.

    For every morph, the WORSE (minimum) of its two similarity scores against
    genuine probes of contributor A and contributor B must clear the threshold
    for the morph to count as a successful attack against BOTH identities at
    once. Returns (rate, successes, total).
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    worst = np.minimum(scores_a, scores_b)
    successes = int(np.sum(worst >= threshold))
    total = len(worst)
    rate = successes / total if total else float("nan")
    return rate, successes, total


def eer(genuine_scores, impostor_scores, n_thresholds=2000):
    """Equal Error Rate: the threshold where FAR (impostor accept) equals
    FRR (genuine reject), and that shared value.

    Returns (eer, threshold). Implemented by scanning thresholds rather than
    depending on sklearn, so it has no extra dependency and matches the
    textbook definition exactly.
    """
    genuine = np.asarray(genuine_scores, dtype=np.float64)
    impostor = np.asarray(impostor_scores, dtype=np.float64)
    lo = min(genuine.min(), impostor.min())
    hi = max(genuine.max(), impostor.max())
    thresholds = np.linspace(lo, hi, n_thresholds)

    far = np.array([(impostor >= t).mean() for t in thresholds])
    frr = np.array([(genuine < t).mean() for t in thresholds])
    diff = np.abs(far - frr)
    i = int(np.argmin(diff))
    return float((far[i] + frr[i]) / 2.0), float(thresholds[i])


def apcer_bpcer_at_threshold(attack_scores, bonafide_scores, threshold):
    """Detector operating point at a fixed decision threshold.

    `*_scores` are the detector's own "looks like an attack" score (higher =
    more suspicious). APCER = fraction of attacks the detector calls bona fide
    (misses); BPCER = fraction of bona fide samples the detector calls an
    attack (false alarms). Both must be reported together - a detector that
    flags everything has 0% APCER but 100% BPCER.
    """
    attack_scores = np.asarray(attack_scores, dtype=np.float64)
    bonafide_scores = np.asarray(bonafide_scores, dtype=np.float64)
    apcer = float((attack_scores < threshold).mean())
    bpcer = float((bonafide_scores >= threshold).mean())
    return apcer, bpcer


def bpcer_at_apcer(attack_scores, bonafide_scores, target_apcer, n_thresholds=2000):
    """BPCER at the threshold giving (at most) `target_apcer`.

    Standard way ISO/IEC 30107-3 tables report a detector: "BPCER @ APCER=5%"
    etc. Scans thresholds the same way `eer` does for consistency.
    """
    attack_scores = np.asarray(attack_scores, dtype=np.float64)
    bonafide_scores = np.asarray(bonafide_scores, dtype=np.float64)
    lo = min(attack_scores.min(), bonafide_scores.min())
    hi = max(attack_scores.max(), bonafide_scores.max())
    thresholds = np.linspace(lo, hi, n_thresholds)
    best_bpcer, best_apcer = 1.0, None
    for t in thresholds:
        apcer, bpcer = apcer_bpcer_at_threshold(attack_scores, bonafide_scores, t)
        if apcer <= target_apcer and bpcer < best_bpcer:
            best_bpcer, best_apcer = bpcer, apcer
    return best_bpcer, best_apcer
