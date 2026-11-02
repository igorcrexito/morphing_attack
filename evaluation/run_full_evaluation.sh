#!/usr/bin/env bash
# Runs the entire evaluation suite end to end, in order, from the project root.
#
# Usage:
#   cd /home/igor/PycharmProjects/morphing_attack
#   ./evaluation/run_full_evaluation.sh                 # everything
#   SKIP_ROBUSTNESS=1 ./evaluation/run_full_evaluation.sh   # skip step 4 (slower)
#   ABLATION_LIMIT=15 ./evaluation/run_full_evaluation.sh   # smaller ablation subset
#
# Each step is idempotent and safe to re-run on its own (see README.md); this
# script exists purely as a convenience wrapper, not as new logic.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Activate the project's virtualenv if present and not already active.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "morphing_env/bin/activate" ]; then
    # shellcheck disable=SC1091
    source morphing_env/bin/activate
fi

ABLATION_LIMIT="${ABLATION_LIMIT:-30}"

echo "=== Step 0: SYN-MAD pairs + morphs (skipped if already present) ==="
if [ ! -f pairs_synmad.csv ]; then
    python3 main_syn_mad_pairs.py
else
    echo "pairs_synmad.csv already exists - skipping (delete it to rebuild)."
fi
if [ ! -d morphed_dataset/frll_synmad/neutral ] || [ -z "$(ls -A morphed_dataset/frll_synmad/neutral 2>/dev/null)" ]; then
    python3 main_generate_morphed_dataset.py --pairs pairs_synmad.csv
else
    echo "morphed_dataset/frll_synmad already populated - skipping (see README to force)."
fi

echo "=== Step 1: calibrate FR thresholds on bona fide FRLL data ==="
python3 evaluation/01_calibrate_thresholds.py

echo "=== Step 2: vulnerability (MMPMR) on the full pipeline's morphs ==="
python3 evaluation/02_vulnerability_eval.py --morph-root morphed_dataset/frll_synmad --label full_pipeline

echo "=== Step 3: statistical MAD baseline on the full pipeline's morphs ==="
python3 evaluation/03_mad_baseline_eval.py --morph-root morphed_dataset/frll_synmad --label full_pipeline

if [ "${SKIP_ROBUSTNESS:-0}" != "1" ]; then
    echo "=== Step 4: robustness (JPEG + print-scan simulation) ==="
    python3 evaluation/04_robustness_eval.py
else
    echo "=== Step 4: robustness SKIPPED (SKIP_ROBUSTNESS=1) ==="
fi

echo "=== Step 5: generate ablation variants (limit=${ABLATION_LIMIT}/group) ==="
python3 evaluation/05_ablation_generate.py --limit "${ABLATION_LIMIT}"

echo "=== Step 6: score ablation variants ==="
python3 evaluation/06_ablation_eval.py

echo "=== Step 7: build the final report ==="
python3 evaluation/07_build_report.py

echo
echo "Done. Report: evaluation/artifacts/FINAL_REPORT.md"
