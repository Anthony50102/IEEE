#!/bin/bash
# ===========================================================================
#  FNO Kuramoto-Sivashinsky Full Pipeline (Local)
# ===========================================================================
#
# Runs the complete FNO pipeline for the KS equation:
#   Step 1: Single-step training (+ quick test)
#   Step 2: Curriculum rollout training (+ quick test)
#   Step 3: Evaluation — forecasts, metrics, figures
#
# Estimated runtime:
#   ~12 h on CPU  |  ~1 h with GPU (CUDA)
#
# Usage (run from the IEEE/ project root):
#   ./scripts/run_fno_ks_local.sh            # Run all steps (1 → 2 → 3)
#   ./scripts/run_fno_ks_local.sh 1           # Step 1 only
#   ./scripts/run_fno_ks_local.sh 2 <run_dir> # Step 2 only (needs existing run)
#   ./scripts/run_fno_ks_local.sh 3 <run_dir> # Step 3 only (needs existing run)
#
# ===========================================================================

set -e

# ===  Configuration  =======================================================
CONFIG="fno/config/fno_ks_temporal_split_local.yaml"
STEP="${1:-all}"
RUN_DIR="${2:-}"

# ===  Banner  ===============================================================
echo "================================================"
echo "  FNO KS Pipeline  (local)"
echo "================================================"
echo "Config : $CONFIG"
echo "Step   : $STEP"
echo ""

# ===  Helpers  ==============================================================

find_latest_run() {
    # Extract output_base and run_name from config, then find the newest run dir
    local output_base run_name latest
    output_base=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
print(cfg.get('paths', {}).get('output_base', './output/'))
")
    run_name=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
print(cfg.get('run_name', 'fno_experiment'))
")
    latest=$(ls -td "${output_base}/${run_name}_"* 2>/dev/null | head -1)
    echo "$latest"
}

require_run_dir() {
    if [ -z "$RUN_DIR" ]; then
        echo "Error: Step $1 requires a run directory."
        echo "Usage: ./scripts/run_fno_ks_local.sh $1 <run_dir>"
        exit 1
    fi
    if [ ! -d "$RUN_DIR" ]; then
        echo "Error: Run directory does not exist: $RUN_DIR"
        exit 1
    fi
}

# ===  Step 1: Single-Step Training  ========================================

if [ "$STEP" = "1" ] || [ "$STEP" = "all" ]; then
    echo "================================================"
    echo "  Step 1: Single-Step Training"
    echo "================================================"
    python3 fno/step_1_train.py --config "$CONFIG" --test

    echo ""
    echo "Step 1 completed successfully."
    echo ""
fi

# ===  Auto-detect run directory (full pipeline)  ===========================

if [ "$STEP" = "all" ]; then
    RUN_DIR=$(find_latest_run)
    if [ -z "$RUN_DIR" ]; then
        echo "Error: No run directory found after Step 1."
        exit 1
    fi
    echo "Auto-detected run directory: $RUN_DIR"
    echo ""
fi

# ===  Step 2: Curriculum Rollout Training  ==================================

if [ "$STEP" = "2" ] || [ "$STEP" = "all" ]; then
    [ "$STEP" = "2" ] && require_run_dir 2

    echo "================================================"
    echo "  Step 2: Curriculum Rollout Training"
    echo "================================================"
    python3 fno/step_2_train.py --config "$CONFIG" --run-dir "$RUN_DIR" --test

    echo ""
    echo "Step 2 completed successfully."
    echo ""
fi

# ===  Step 3: Evaluation  ==================================================

if [ "$STEP" = "3" ] || [ "$STEP" = "all" ]; then
    [ "$STEP" = "3" ] && require_run_dir 3

    echo "================================================"
    echo "  Step 3: Evaluation"
    echo "================================================"
    python3 fno/step_3_evaluate.py --config "$CONFIG" --run-dir "$RUN_DIR"

    echo ""
    echo "Step 3 completed successfully."
    echo ""
fi

# ===  Summary  ==============================================================

echo "================================================"
echo "  FNO KS Pipeline Complete!"
echo "================================================"
echo "Run directory : $RUN_DIR"
echo ""

if [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ]; then
    FIGS=$(find "$RUN_DIR" -name "*.png" -o -name "*.pdf" 2>/dev/null)
    if [ -n "$FIGS" ]; then
        echo "Figures:"
        echo "$FIGS" | sed 's/^/  /'
    fi

    METRICS=$(find "$RUN_DIR" -name "*metrics*" 2>/dev/null)
    if [ -n "$METRICS" ]; then
        echo ""
        echo "Metrics files:"
        echo "$METRICS" | sed 's/^/  /'
    fi
fi
