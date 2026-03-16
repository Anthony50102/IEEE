#!/bin/bash
# Run FNO training locally
#
# Usage:
#   ./run_local.sh config/fno_temporal_split.yaml [step]
#
# Examples:
#   ./run_local.sh config/fno_temporal_split.yaml        # Run all steps
#   ./run_local.sh config/fno_temporal_split.yaml 1      # Run step 1 only
#   ./run_local.sh config/fno_temporal_split.yaml 2      # Run step 2 only (requires existing run dir)
#   ./run_local.sh config/fno_temporal_split.yaml 3      # Run step 3 only (requires existing run dir)

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
    if [ "$STEP" = "all" ]; then
        python step_1_train.py --config "$CONFIG" --test
    else
        python step_1_train.py --config "$CONFIG"
    fi
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

if [ "$STEP" = "3" ]; then
    if [ -z "$RUN_DIR" ]; then
        echo "Error: Step 3 requires --run-dir argument"
        echo "Usage: ./run_local.sh config.yaml 3 <run_dir>"
        exit 1
    fi
    echo "Running Step 3: Evaluation..."
    python step_3_evaluate.py --config "$CONFIG" --run-dir "$RUN_DIR"
fi

if [ "$STEP" = "all" ]; then
    # Extract output_base and run_name from config YAML
    OUTPUT_BASE=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg.get('paths',{}).get('output_base','./output/'))")
    RUN_NAME=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print(cfg.get('run_name','fno_experiment'))")
    LATEST_RUN=$(ls -td ${OUTPUT_BASE}/${RUN_NAME}_* 2>/dev/null | head -1)
    if [ -n "$LATEST_RUN" ]; then
        echo ""
        echo "Running Step 2: Rollout Training..."
        python step_2_train.py --config "$CONFIG" --run-dir "$LATEST_RUN" --test
        
        echo ""
        echo "Running Step 3: Evaluation..."
        python step_3_evaluate.py --config "$CONFIG" --run-dir "$LATEST_RUN"
    else
        echo "Warning: No run directory found for Steps 2/3"
    fi
fi
