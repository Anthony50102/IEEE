#!/bin/bash
# Run FNO training locally
#
# Usage:
#   ./run_local.sh config/fno_temporal_split.yaml [step]
#
# Examples:
#   ./run_local.sh config/fno_temporal_split.yaml        # Run both steps
#   ./run_local.sh config/fno_temporal_split.yaml 1      # Run step 1 only
#   ./run_local.sh config/fno_temporal_split.yaml 2      # Run step 2 only (requires existing run dir)

set -e

CONFIG="${1:-config/fno_temporal_split.yaml}"
STEP="${2:-all}"
RUN_DIR="${3:-}"

echo "================================================"
echo "  FNO Training Pipeline"
echo "================================================"
echo "Config: $CONFIG"
echo "Step: $STEP"
echo ""

if [ "$STEP" = "1" ] || [ "$STEP" = "all" ]; then
    echo "Running Step 1: Single-Step Training..."
    python step_1_train.py --config "$CONFIG"
fi

if [ "$STEP" = "2" ]; then
    if [ -z "$RUN_DIR" ]; then
        echo "Error: Step 2 requires --run-dir argument"
        echo "Usage: ./run_local.sh config.yaml 2 <run_dir>"
        exit 1
    fi
    echo "Running Step 2: Rollout Training..."
    python step_2_train.py --config "$CONFIG" --run-dir "$RUN_DIR"
fi

if [ "$STEP" = "all" ]; then
    # Find the most recent run directory
    LATEST_RUN=$(ls -td output/fno_* 2>/dev/null | head -1)
    if [ -n "$LATEST_RUN" ]; then
        echo ""
        echo "Running Step 2: Rollout Training..."
        python step_2_train.py --config "$CONFIG" --run-dir "$LATEST_RUN"
    else
        echo "Warning: No run directory found for Step 2"
    fi
fi
