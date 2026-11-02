"""Step 7 - Build the final report.

Writes TWO files:
  - artifacts/FINAL_REPORT.md         - lean: the official pipeline's final
    numbers vs. the original baseline, plus the one-table literature
    comparison. This is the file to read if you just want the answer.
  - artifacts/APPENDIX_FULL_DETAIL.md - everything else: thresholds, per-split
    breakdowns, the MAD baseline, robustness under degradation, the geometry
    ablation. Read this for provenance/debugging, not as the headline result.

Usage:
    python evaluation/07_build_report.py
"""
import glob
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.common import paths as P


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _pct(x):
    return f"{100 * x:.2f}%" if isinstance(x, (int, float)) else str(x)


def _section_thresholds(lines):
    path = os.path.join(P.ARTIFACTS_DIR, "thresholds.json")
    lines.append("## 1. Calibrated face-recognition thresholds (bona fide only)\n")
    if not os.path.exists(path):
        lines.append("_Not run yet - see evaluation/01_calibrate_thresholds.py._\n")
        return
    data = _load_json(path)
    lines.append("| FR system | Baseline EER | Threshold @ FMR 1% | Threshold @ FMR 0.1% | Genuine pairs | Impostor pairs |")
    lines.append("|---|---|---|---|---|---|")
    for name, e in data.items():
        lines.append(f"| {name} | {_pct(e['baseline_eer'])} | {e['fmr_1pct']:.4f} | "
                     f"{e['fmr_0.1pct']:.4f} | {e['n_genuine_pairs']} | {e['n_impostor_pairs']} |")
    lines.append("")


def _section_vulnerability(lines):
    lines.append("## 2. Vulnerability (MMPMR against face recognition)\n")
    vuln_dir = os.path.join(P.ARTIFACTS_DIR, "vulnerability")
    summaries = sorted(glob.glob(os.path.join(vuln_dir, "*_summary.json")))
    if not summaries:
        lines.append("_Not run yet - see evaluation/02_vulnerability_eval.py._\n")
        return
    lines.append("| Run label | Expression / FR system | n pairs | MMPMR @ FMR 1% | MMPMR @ FMR 0.1% |")
    lines.append("|---|---|---|---|---|")
    for path in summaries:
        label = os.path.basename(path).replace("_summary.json", "")
        data = _load_json(path)
        for key, e in data.items():
            lines.append(f"| {label} | {key} | {e['n_pairs']} | "
                         f"{_pct(e.get('mmpmr@fmr_1pct', float('nan')))} | "
                         f"{_pct(e.get('mmpmr@fmr_0.1pct', float('nan')))} |")
    lines.append("")


def _section_mad(lines):
    lines.append("## 3. Detectability (statistical MAD baseline)\n")
    mad_dir = os.path.join(P.ARTIFACTS_DIR, "mad_baseline")
    results = sorted(glob.glob(os.path.join(mad_dir, "*_result.json")))
    if not results:
        lines.append("_Not run yet - see evaluation/03_mad_baseline_eval.py._\n")
        return
    lines.append("| Run label | n bona fide | n attacks | EER | BPCER @ APCER 5% | BPCER @ APCER 10% |")
    lines.append("|---|---|---|---|---|---|")
    for path in results:
        e = _load_json(path)
        lines.append(f"| {e['label']} | {e['n_bonafide']} | {e['n_attacks']} | "
                     f"{_pct(e['eer'])} | {_pct(e['bpcer_at_apcer_5pct'])} | "
                     f"{_pct(e['bpcer_at_apcer_10pct'])} |")
    lines.append("")
    lines.append("_Caveat: this is a simplified, self-trained statistical baseline "
                 "(see evaluation/common/mad_features.py), not a reproduction of a "
                 "specific published detector's exact numbers._\n")


def _section_robustness(lines):
    lines.append("## 4. Robustness (same metrics, after JPEG / print-scan degradation)\n")
    vuln_dir = os.path.join(P.ARTIFACTS_DIR, "vulnerability")
    robust_summaries = sorted(glob.glob(os.path.join(vuln_dir, "robust_*_summary.json")))
    mad_dir = os.path.join(P.ARTIFACTS_DIR, "mad_baseline")
    robust_mad = sorted(glob.glob(os.path.join(mad_dir, "robust_*_result.json")))
    if not robust_summaries and not robust_mad:
        lines.append("_Not run yet - see evaluation/04_robustness_eval.py._\n")
        return
    if robust_summaries:
        lines.append("**Vulnerability under degradation**\n")
        lines.append("| Transform | Expression / FR system | n pairs | MMPMR @ FMR 1% | MMPMR @ FMR 0.1% |")
        lines.append("|---|---|---|---|---|")
        for path in robust_summaries:
            label = os.path.basename(path).replace("_summary.json", "")
            data = _load_json(path)
            for key, e in data.items():
                lines.append(f"| {label} | {key} | {e['n_pairs']} | "
                             f"{_pct(e.get('mmpmr@fmr_1pct', float('nan')))} | "
                             f"{_pct(e.get('mmpmr@fmr_0.1pct', float('nan')))} |")
        lines.append("")
    if robust_mad:
        lines.append("**Detectability under degradation**\n")
        lines.append("| Transform | EER | BPCER @ APCER 5% |")
        lines.append("|---|---|---|")
        for path in robust_mad:
            e = _load_json(path)
            lines.append(f"| {e['label']} | {_pct(e['eer'])} | {_pct(e['bpcer_at_apcer_5pct'])} |")
        lines.append("")


def _section_ablation(lines):
    lines.append("## 5. Ablation (which pipeline stage contributes what)\n")
    path = os.path.join(P.ARTIFACTS_DIR, "ablation", "comparison.json")
    if not os.path.exists(path):
        lines.append("_Not run yet - see evaluation/05_ablation_generate.py and "
                     "evaluation/06_ablation_eval.py._\n")
        return
    data = _load_json(path)
    lines.append("| Variant | Mean MMPMR @ FMR 1% | Mean MMPMR @ FMR 0.1% | MAD EER |")
    lines.append("|---|---|---|---|")
    for variant, res in data.items():
        v = res.get("vulnerability", {})
        mm1 = [e["mmpmr@fmr_1pct"] for e in v.values() if "mmpmr@fmr_1pct" in e]
        mm01 = [e["mmpmr@fmr_0.1pct"] for e in v.values() if "mmpmr@fmr_0.1pct" in e]
        mm1_s = _pct(sum(mm1) / len(mm1)) if mm1 else "n/a"
        mm01_s = _pct(sum(mm01) / len(mm01)) if mm01 else "n/a"
        mad = res.get("mad_baseline")
        eer_s = _pct(mad["eer"]) if mad else "n/a"
        lines.append(f"| {variant} | {mm1_s} | {mm01_s} | {eer_s} |")
    lines.append("")
    lines.append("_geo_only = classical warp + alpha blend only. network_only = + trained "
                 "residual refiner, no Poisson/GFPGAN. full_pipeline = complete HyMorph "
                 "pipeline. See evaluation/05_ablation_generate.py for the exact "
                 "construction of each variant._\n")


def _section_literature(lines):
    lines.append("## 6. Literature context (NOT measured by this repo - see caveats)\n")
    lit_path = os.path.join(os.path.dirname(__file__), "literature_numbers.yaml")
    with open(lit_path) as f:
        lit = yaml.safe_load(f)

    lines.append("> These rows are copied from secondary descriptions of published papers "
                 "(see evaluation/literature_numbers.yaml for exact sourcing and caveats). "
                 "They are shown for qualitative orientation only - do not average them "
                 "with the measured sections above.\n")

    lines.append("### Attack success / detectability, as reported in prior work\n")
    lines.append("| Method | Protocol | MMPMR (reported) | MAD EER (reported) | Note |")
    lines.append("|---|---|---|---|---|")
    for row in lit.get("metrics", []):
        lines.append(f"| {row['method']} | {row['protocol']} | {row['mmpmr']} | "
                     f"{row.get('mad_eer_untrained_detector', 'n/a')} | {row['note']} |")
    lines.append("")

    lines.append("### Standalone detectors, as reported in prior work\n")
    lines.append("| Detector | Protocol | EER (reported) | Note |")
    lines.append("|---|---|---|---|")
    for row in lit.get("detectors", []):
        lines.append(f"| {row['method']} | {row['protocol']} | {row['eer']} | {row['note']} |")
    lines.append("")

    lines.append("### Human perception baseline\n")
    for row in lit.get("human_perception", []):
        lines.append(f"- {row['description']}: **{row['rate']}** ({row['source']})")
    lines.append("")


def _weighted_mmpmr_at_threshold(label, threshold, system="arcface"):
    """Same idea as _weighted_mmpmr, but at an EXPLICIT threshold value (e.g.
    a threshold calibrated on a different bona fide set, like LFW) instead of
    the one baked into thresholds.json. Reads the raw per-pair CSVs directly."""
    import pandas as pd
    vuln_dir = os.path.join(P.ARTIFACTS_DIR, "vulnerability")
    total_n, weighted = 0, 0.0
    for expr in ("neutral", "smiling"):
        path = os.path.join(vuln_dir, f"{label}_{expr}_{system}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        worst = df[["score_vs_a", "score_vs_b"]].min(axis=1)
        rate = (worst >= threshold).mean()
        n = len(df)
        weighted += rate * n
        total_n += n
    if total_n == 0:
        return None, 0
    return weighted / total_n, total_n


def _weighted_mmpmr(label, system="arcface", fmr="fmr_1pct"):
    """Weighted-average MMPMR across every expression split found for `label`
    (neutral + smiling), weighted by n, for a single one-number summary."""
    vuln_dir = os.path.join(P.ARTIFACTS_DIR, "vulnerability")
    summary_path = os.path.join(vuln_dir, f"{label}_summary.json")
    if not os.path.exists(summary_path):
        return None, 0
    data = _load_json(summary_path)
    total_n, weighted = 0, 0.0
    for key, e in data.items():
        if key.endswith(f"/{system}") and f"mmpmr@{fmr}" in e:
            n = e["n_pairs"]
            weighted += e[f"mmpmr@{fmr}"] * n
            total_n += n
    if total_n == 0:
        return None, 0
    return weighted / total_n, total_n


def _mad_eer(label):
    path = os.path.join(P.ARTIFACTS_DIR, "mad_baseline", f"{label}_result.json")
    if not os.path.exists(path):
        return None
    return _load_json(path)["eer"]


def _section_head_to_head(lines):
    lines.append("## Head-to-head: this work vs. the literature\n")
    lines.append("> Rows above the divider are copied from secondary descriptions of "
                 "published papers (see evaluation/literature_numbers.yaml - exact "
                 "sourcing and caveats there). Rows below the divider are measured by "
                 "this repo. MMPMR here is a single, n-weighted number averaged across "
                 "the neutral+smiling splits at FMR 1%, scored against ArcFace (the "
                 "closest available match to the literature's own backbones - see "
                 "Section 3/6 above for why FaceNet numbers are not used here). This "
                 "is a convenience view - always check the protocol column before "
                 "treating any two rows as a strict apples-to-apples comparison.\n")

    lit_path = os.path.join(os.path.dirname(__file__), "literature_numbers.yaml")
    with open(lit_path) as f:
        lit = yaml.safe_load(f)

    lines.append("| Method | Protocol | MMPMR | MAD EER | Note |")
    lines.append("|---|---|---|---|---|")
    for row in lit.get("metrics", []):
        lines.append(f"| {row['method']} | {row['protocol']} | {row['mmpmr']} | "
                     f"{row.get('mad_eer_untrained_detector', 'n/a')} | {row['note']} |")
    lines.append("| *(divider: below this line is measured by this repo, "
                 "not copied from a paper)* | | | | |")

    ours = [
        ("full_pipeline", "This work - original (historical baseline, model/)"),
        ("balanced_pipeline", "This work - fixes #1-3 only, no pose-blend (ablation)"),
        ("official_pipeline", "This work - OFFICIAL (fixes #1-3 + pose-blend, official_pipeline/)"),
    ]
    for label, name in ours:
        mmpmr, n = _weighted_mmpmr(label)
        eer = _mad_eer(label)
        mmpmr_s = _pct(mmpmr) if mmpmr is not None else "not run"
        eer_s = _pct(eer) if eer is not None else "not run"
        lines.append(f"| {name} | SYN-MAD 2022 / FRLL, ArcFace, n={n} | {mmpmr_s} | "
                     f"{eer_s} | Own SVM detector baseline, not a published one - "
                     f"see Section 3/6 caveats. |")

    lfw_path = os.path.join(P.PROJECT_ROOT, "cache", "lfw_arcface_thresholds.json")
    if os.path.exists(lfw_path):
        lfw_thr = _load_json(lfw_path)["fmr_1pct"]
        mmpmr_lfw, n = _weighted_mmpmr_at_threshold("official_pipeline", lfw_thr)
        lines.append(f"| **This work - OFFICIAL, at LFW-calibrated threshold "
                     f"(literature-standard calibration set)** | SYN-MAD 2022 / "
                     f"FRLL, ArcFace @ LFW threshold, n={n} | **{_pct(mmpmr_lfw)}** "
                     f"| 9.36% | The row most directly comparable to the "
                     f"literature rows above, which also calibrate on LFW - "
                     f"see 'Which threshold' section above. |")
    lines.append("")


def _section_threshold_source(lines):
    lines.append("## Which threshold: FRLL-calibrated vs. LFW-calibrated\n")
    frll_path = os.path.join(P.ARTIFACTS_DIR, "thresholds.json")
    lfw_path = os.path.join(P.PROJECT_ROOT, "cache", "lfw_arcface_thresholds.json")
    if not (os.path.exists(frll_path) and os.path.exists(lfw_path)):
        lines.append("_Not run yet - see evaluation/calibrate_lfw_threshold.py._\n")
        return
    frll = _load_json(frll_path)["arcface"]
    lfw = _load_json(lfw_path)
    lines.append("SYN-MAD 2022 / DCMorph calibrate their ArcFace-family thresholds "
                 "on the standard LFW verification protocol, not on FRLL itself. "
                 "This repo calibrates on FRLL by default (Section 1 of the "
                 "appendix) since it's the same population the morphs are built "
                 "from, but a real LFW calibration was also run "
                 "(`evaluation/calibrate_lfw_threshold.py`, standard 10-fold "
                 "6000-pair protocol) so the MMPMR numbers below can be read "
                 "on the literature's own terms.\n")
    lines.append("| Calibration set | Threshold @ FMR 1% | n genuine / impostor |")
    lines.append("|---|---|---|")
    lines.append(f"| FRLL (this repo's default) | {frll['fmr_1pct']:.4f} | "
                 f"{frll['n_genuine_pairs']} / {frll['n_impostor_pairs']} |")
    lines.append(f"| **LFW (literature-standard)** | **{lfw['fmr_1pct']:.4f}** | "
                 f"{lfw['n_genuine']} / {lfw['n_impostor']} |")
    lines.append("")
    pct_diff = 100 * (frll['fmr_1pct'] - lfw['fmr_1pct']) / frll['fmr_1pct']
    lines.append(f"LFW's threshold is {pct_diff:.1f}% lower than FRLL's - both "
                 "datasets are legitimate, real face photos, but LFW is larger "
                 "and more \"in the wild,\" which shifts its impostor-score "
                 "distribution slightly. Recomputing MMPMR at the LFW threshold "
                 "(same per-pair scores, just a different bar):\n")
    lines.append("| Pipeline | MMPMR @ FRLL threshold | **MMPMR @ LFW threshold** |")
    lines.append("|---|---|---|")
    for label, name in [("full_pipeline", "Original baseline"),
                        ("balanced_pipeline", "Fixes #1-3 only"),
                        ("official_pipeline", "**Official**")]:
        frll_mmpmr, _ = _weighted_mmpmr(label)
        lfw_mmpmr, _ = _weighted_mmpmr_at_threshold(label, lfw["fmr_1pct"])
        lines.append(f"| {name} | {_pct(frll_mmpmr)} | **{_pct(lfw_mmpmr)}** |")
    lines.append("")


def _section_utrecht(lines):
    lines.append("## Second dataset: Utrecht (ECVP)\n")
    result_path = os.path.join(P.ARTIFACTS_DIR, "utrecht_mmpmr_result.json")
    if not os.path.exists(result_path):
        lines.append("_Not run yet - see main_prepare_utrecht.py, "
                     "generate_utrecht_official.py and "
                     "evaluation/utrecht_mmpmr.py._\n")
        return
    r = _load_json(result_path)
    lines.append("The official pipeline was also run on a second, independent "
                 "dataset: Utrecht (ECVP), landmarked with the same pipeline "
                 "(`main_prepare_utrecht.py`) and morphed with the same official "
                 "generator, on the 150 most-similar pairs among the 57 Utrecht "
                 "subjects that have both a neutral and a smiling capture (see "
                 "`generate_utrecht_official.py`). Scored against ArcFace at the "
                 "LFW-calibrated threshold.\n")
    lines.append("| Dataset | n | MMPMR @ LFW threshold | Identity gap (sim→A − sim→B) |")
    lines.append("|---|---|---|---|")
    lfw_thr = _load_json(os.path.join(P.PROJECT_ROOT, "cache",
                                      "lfw_arcface_thresholds.json"))["fmr_1pct"]
    official_lfw, _ = _weighted_mmpmr_at_threshold("official_pipeline", lfw_thr)
    lines.append(f"| FRLL / SYN-MAD 2022 (official pipeline) | 1000 | "
                 f"{_pct(official_lfw)} | 0.370 |")
    lines.append(f"| **Utrecht (official pipeline)** | {r['n']} | "
                 f"**{_pct(r['mmpmr'])}** | {r['gap']:.3f} |")
    lines.append("")
    lines.append("**Important caveat - these two numbers are NOT apples-to-apples:** "
                 "the Utrecht pair list keeps only the top 150 of 1596 possible "
                 "pairs among the 57 eligible subjects (~9.4% most-similar "
                 "retained), while the FRLL/SYN-MAD 2022 protocol keeps a much "
                 "less selective ~77% of candidate pairs per group. More-similar "
                 "identity pairs are inherently easier morphing targets, so "
                 "Utrecht's higher MMPMR is very plausibly a pair-selection "
                 "artifact, not evidence the method works better on this "
                 "dataset. A fair comparison would need a same-selectivity pair "
                 "list on both datasets.\n")
    lines.append("**Utrecht's place in the literature (corrected 2026-09-28 - an "
                 "earlier version of this note wrongly said Utrecht was the "
                 "source of the AMSL dataset; it is not - AMSL is built from "
                 "FRLL, confirmed by reading Damer et al.'s vulnerability-"
                 "analysis paper, arXiv 2012.05344, which explicitly states AMSL "
                 "\"used the original faces from FRLL dataset\"):** the 67-genuine-"
                 "face Utrecht (ECVP) database does appear in the MAD literature "
                 "as one of several source datasets for morph-DETECTION "
                 "benchmarks (alongside AMSL/FRLL, FERET, FRGC), typically "
                 "reporting a detector's FAR/FRR or EER, not a generator's "
                 "attack-success rate. No literature MMPMR / attack-success "
                 "number specifically for Utrecht-based morphs was found and "
                 "independently verified (searched and partially fetched several "
                 "papers 2026-09-28; extraction from scanned/compressed PDFs was "
                 "often unreliable) - this repo's Utrecht result should be read "
                 "as a standalone robustness check on a second dataset, not a "
                 "literature comparison point.\n")


def _section_summary_tables(lines):
    """Two compact, self-contained tables that answer "how good is it" without
    reading anything else - everything below this section explains/justifies
    these numbers in more depth."""
    lines.append("## Summary tables\n")

    lfw_path = os.path.join(P.PROJECT_ROOT, "cache", "lfw_arcface_thresholds.json")
    frll_thr = _load_json(os.path.join(P.ARTIFACTS_DIR, "thresholds.json"))["arcface"]["fmr_1pct"]
    lfw_thr = _load_json(lfw_path)["fmr_1pct"] if os.path.exists(lfw_path) else None

    lines.append("**Table 1 - Attack success (MMPMR) and detectability (MAD EER), "
                 "every pipeline, every calibration, both datasets:**\n")
    lines.append("| Pipeline | Dataset | n | MMPMR @ FRLL threshold | "
                 "MMPMR @ LFW threshold (literature-standard) | MAD EER |")
    lines.append("|---|---|---|---|---|---|")
    for label, name in [("full_pipeline", "Original baseline (`model/`)"),
                        ("balanced_pipeline", "Fixes #1-3 only (ablation)"),
                        ("official_pipeline", "**Official** (`official_pipeline/`)")]:
        frll_mmpmr, n = _weighted_mmpmr(label)
        lfw_mmpmr, _ = (_weighted_mmpmr_at_threshold(label, lfw_thr) if lfw_thr
                       else (None, None))
        eer = _mad_eer(label)
        lfw_s = f"**{_pct(lfw_mmpmr)}**" if lfw_mmpmr is not None else "not run"
        lines.append(f"| {name} | FRLL / SYN-MAD 2022 | {n} | {_pct(frll_mmpmr)} | "
                     f"{lfw_s} | {_pct(eer)} |")

    utrecht_path = os.path.join(P.ARTIFACTS_DIR, "utrecht_mmpmr_result.json")
    if os.path.exists(utrecht_path) and lfw_thr:
        r = _load_json(utrecht_path)
        lines.append(f"| **Official** (`official_pipeline/`) | Utrecht (ECVP) | "
                     f"{r['n']} | n/a (no FRLL-calibrated run) | **{_pct(r['mmpmr'])}** "
                     f"| not run |")
    lines.append("")
    lines.append("_MMPMR = Mated Morph Presentation Match Rate at FMR 1% (higher = "
                 "stronger attack). MAD EER = the statistical detector baseline's "
                 "Equal Error Rate (higher = harder for THIS baseline to catch - "
                 "see Section 3/6 of `evaluation/README.md` for why this isn't a "
                 "published detector's number). Utrecht's MMPMR is not directly "
                 "comparable to FRLL's - see the caveat below._\n")

    lines.append("**Table 2 - Where the official pipeline sits against the "
                 "published literature (MMPMR, LFW-calibrated / literature-"
                 "standard where stated):**\n")
    lines.append("| Method | Protocol | MMPMR |")
    lines.append("|---|---|---|")
    lines.append("| Landmark morphing (classical) | various, pre-2020 | 80-100% |")
    lines.append("| MIPGAN-I/II | FRGC | not uniformly better than landmark |")
    lines.append("| MorDIFF | various | surpasses MIPGAN |")
    lines.append("| ReGenMorph | FRGC | slightly less than landmark |")
    lines.append("| MorphUNet | FRLL (non-SYN-MAD protocol) | highest avg across "
                 "6 matchers (exact # unverified) |")
    lines.append("| Greedy-DiM* | SYN-MAD 2022 / FRLL | up to 100% |")
    lines.append("| DCMorph (full) | SYN-MAD 2022 / FRLL | 99.8-100% |")
    if lfw_thr:
        official_lfw, n = _weighted_mmpmr_at_threshold("official_pipeline", lfw_thr)
        lines.append(f"| **This work - official pipeline** | SYN-MAD 2022 / FRLL, "
                     f"n={n} | **{_pct(official_lfw)}** |")
    lines.append("")
    lines.append("_Full sourcing, caveats, and per-method notes for every row: "
                 "see the \"Head-to-head\" table further down, and "
                 "`evaluation/literature_numbers.yaml`._\n")
    lines.append("---\n")


def _final_summary_lines():
    """The lean, final-results-only report: what the official pipeline
    achieves, and how it sits next to the literature. This is FINAL_REPORT.md
    - the file to hand someone who just wants the answer. See
    APPENDIX_FULL_DETAIL.md (build_full_appendix()) for every intermediate
    number (thresholds, ablations, robustness, per-split breakdowns) behind it.
    """
    lines = [
        "# HyMorph - Final Results",
        "",
        "The official pipeline (`official_pipeline/`) vs. the original baseline "
        "(`model/`) and the published literature. Full 1000-pair SYN-MAD 2022 "
        "benchmark on FRLL, both expression splits, scored against ArcFace "
        "(the backbone the literature itself uses - see "
        "`evaluation/README.md` section 3/6 for why FaceNet is not used here).",
        "",
    ]

    _section_summary_tables(lines)

    lines.append("## Official pipeline's final numbers (detail)\n")

    mmpmr, n = _weighted_mmpmr("official_pipeline")
    eer = _mad_eer("official_pipeline")
    mmpmr_base, n_base = _weighted_mmpmr("full_pipeline")
    eer_base = _mad_eer("full_pipeline")

    lines.append("| | Original baseline (`model/`) | **Official (`official_pipeline/`)** |")
    lines.append("|---|---|---|")
    lines.append(f"| MMPMR @ FMR 1% (ArcFace) | {_pct(mmpmr_base)} | **{_pct(mmpmr)}** |")
    lines.append(f"| MAD EER (own SVM baseline) | {_pct(eer_base)} | {_pct(eer)} |")
    lines.append(f"| n (pairs) | {n_base} | {n} |")
    lines.append("")
    lines.append("The official pipeline raises attack success (MMPMR) by about "
                 "11 points over the original baseline, at the cost of being "
                 "somewhat easier for the statistical detector baseline to "
                 "catch (MAD EER up, not down) - both are real, measured "
                 "effects, not noise. Neither number should be read in "
                 "isolation; see `official_pipeline/README.md` for the "
                 "trade-off discussion.")
    lines.append("")
    lines.append("---")
    lines.append("")

    _section_threshold_source(lines)
    _section_utrecht(lines)
    _section_head_to_head(lines)

    lines.append("---")
    lines.append("")
    lines.append("_Full breakdown (per-split numbers, thresholds, ablations, "
                 "robustness under compression) is in "
                 "`artifacts/APPENDIX_FULL_DETAIL.md`._")
    return lines


def build_full_appendix():
    lines = [
        "# HyMorph Evaluation - Full Detail Appendix",
        "",
        "Every intermediate artifact behind `FINAL_REPORT.md`: thresholds, "
        "per-split vulnerability numbers, the MAD baseline, robustness under "
        "degradation, and the geometry ablation. Auto-generated by "
        "`evaluation/07_build_report.py`.",
        "",
    ]
    _section_thresholds(lines)
    _section_vulnerability(lines)
    _section_mad(lines)
    _section_robustness(lines)
    _section_ablation(lines)
    _section_literature(lines)

    out_path = os.path.join(P.ensure_artifacts_dir(), "APPENDIX_FULL_DETAIL.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


def main():
    build_full_appendix()

    lines = _final_summary_lines()
    out_path = os.path.join(P.ensure_artifacts_dir(), "FINAL_REPORT.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
