#!/bin/bash
# Run FNO pipeline on Navier-Stokes data locally
#
# Usage:
#   ./run_fno_ns_local.sh [RE] [STEP] [RUN_DIR]
#   ./run_fno_ns_local.sh              # Re=100, all steps
#   ./run_fno_ns_local.sh 500          # Re=500, all steps
#   ./run_fno_ns_local.sh 100 1        # Re=100, Step 1 only
#   ./run_fno_ns_local.sh 100 2 <dir>  # Re=100, Step 2 only (needs run dir)
#   ./run_fno_ns_local.sh 100 3 <dir>  # Re=100, Step 3 only (needs run dir)

set -e
cd "$(dirname "$0")/../fno"

PYTHON="${PYTHON:-python}"
RE="${1:-100}"
if [ "$RE" = "100" ]; then
    CONFIG="config/fno_ns_temporal_split_local.yaml"
else
    CONFIG="config/fno_ns_re${RE}_temporal_split_local.yaml"
fi
STEP="${2:-all}"
RUN_DIR="${3:-}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    echo "Available configs:"; ls config/fno_ns_*.yaml 2>/dev/null
    exit 1
fi

echo "================================================"
echo "  FNO NS Pipeline (Re=$RE, 64x64)"
echo "================================================"
echo "Config: $CONFIG"
echo "Step: $STEP"
echo ""

if [ "$STEP" = "1" ] || [ "$STEP" = "all" ]; then
    echo "=== Step 1: Single-Step Training ==="
    if [ "$STEP" = "all" ]; then
        $PYTHON step_1_train.py --config "$CONFIG" --test
    else
        $PYTHON step_1_train.py --config "$CONFIG"
    fi
fi

if [ "$STEP" = "2" ]; then
    if [ -z "$RUN_DIR" ]; then
        echo "Error: Step 2 requires run dir argument"
        exit 1
    fi
    echo "=== Step 2: Rollout Training ==="
    $PYTHON step_2_train.py --config "$CONFIG" --run-dir "$RUN_DIR"
fi

if [ "$STEP" = "3" ]; then
    if [ -z "$RUN_DIR" ]; then
        echo "Error: Step 3 requires run dir argument"
        exit 1
    fi
    echo "=== Step 3: Evaluation ==="
    $PYTHON step_3_evaluate.py --config "$CONFIG" --run-dir "$RUN_DIR"
fi

if [ "$STEP" = "all" ]; then
    OUTPUT_BASE=$($PYTHON -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg.get('paths',{}).get('output_base','./output/'))")
    RUN_NAME=$($PYTHON -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg.get('run_name','fno_experiment'))")
    LATEST_RUN=$(ls -td ${OUTPUT_BASE}/${RUN_NAME}_* 2>/dev/null | head -1)
    if [ -n "$LATEST_RUN" ]; then
        echo ""
        echo "=== Step 2: Rollout Training ==="
        $PYTHON step_2_train.py --config "$CONFIG" --run-dir "$LATEST_RUN" --test

        echo ""
        echo "=== Step 3: Evaluation ==="
        $PYTHON step_3_evaluate.py --config "$CONFIG" --run-dir "$LATEST_RUN"
        
        echo ""
        echo "=== Pipeline complete ==="
        echo "Results in: $LATEST_RUN"
    else
        echo "Warning: No run directory found for Steps 2/3"
    fi
fi
