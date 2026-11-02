"""Step 6 - Score the three ablation variants and tabulate the comparison.

Runs the same vulnerability (step 2) and MAD baseline (step 3) evaluations on
each of geo_only / network_only / full_pipeline (produced by
05_ablation_generate.py) and writes a single comparison table, so the
contribution of the neural refiner and of the Poisson/GFPGAN finishing pass can
each be read off directly.

Usage:
    python evaluation/06_ablation_eval.py
    python evaluation/06_ablation_eval.py --skip-mad   # faster, vulnerability only
"""
import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.common import paths as P

vuln_mod = importlib.import_module("evaluation.02_vulnerability_eval")
mad_mod = importlib.import_module("evaluation.03_mad_baseline_eval")

VARIANTS = ["geo_only", "network_only", "full_pipeline"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-mad", action="store_true")
    args = ap.parse_args()

    thr_path = os.path.join(P.ARTIFACTS_DIR, "thresholds.json")
    if not os.path.exists(thr_path):
        raise SystemExit("Run evaluation/01_calibrate_thresholds.py first.")
    with open(thr_path) as f:
        thresholds = json.load(f)

    comparison = {}
    for variant in VARIANTS:
        morph_root = os.path.join(P.ARTIFACTS_DIR, "ablation", variant)
        if not os.path.isdir(morph_root):
            print(f"[skip] {variant}: {morph_root} not found - run "
                  f"05_ablation_generate.py first.")
            continue
        label = f"ablation_{variant}"
        print(f"\n=== Ablation variant: {variant} ===")
        vuln_result = vuln_mod.evaluate_vulnerability(morph_root, thresholds, label=label)
        mad_result = None if args.skip_mad else mad_mod.run_mad_eval(morph_root, label=label)
        comparison[variant] = {"vulnerability": vuln_result, "mad_baseline": mad_result}

    out_path = os.path.join(P.ensure_artifacts_dir("ablation"), "comparison.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n=== Ablation comparison summary ===")
    header = f"{'variant':<15}{'MMPMR@FMR1%':>14}{'MMPMR@FMR0.1%':>16}{'MAD EER':>12}"
    print(header)
    for variant, res in comparison.items():
        # average MMPMR across expression/system combos found, for a one-line summary
        v = res["vulnerability"]
        mm1 = [e["mmpmr@fmr_1pct"] for e in v.values() if "mmpmr@fmr_1pct" in e]
        mm01 = [e["mmpmr@fmr_0.1pct"] for e in v.values() if "mmpmr@fmr_0.1pct" in e]
        mm1_s = f"{sum(mm1)/len(mm1):.3f}" if mm1 else "n/a"
        mm01_s = f"{sum(mm01)/len(mm01):.3f}" if mm01 else "n/a"
        eer_s = f"{res['mad_baseline']['eer']:.3f}" if res.get("mad_baseline") else "n/a"
        print(f"{variant:<15}{mm1_s:>14}{mm01_s:>16}{eer_s:>12}")

    print(f"\nFull detail: {out_path}")


if __name__ == "__main__":
    main()
