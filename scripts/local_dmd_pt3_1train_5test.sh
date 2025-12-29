#!/bin/bash
# Local DMD Pipeline - Step 3: Evaluate/Forecast
# Usage: ./local_dmd_pt3_1train_5test.sh <run_dir>

set -e

CONFIG_FILE="config/local_dmd_1train_5test.yaml"
RUN_DIR=$1

if [ -z "$RUN_DIR" ]; then
    echo "Usage: $0 <run_dir>"
    echo "  run_dir: Path to the run directory from Step 2"
    exit 1
fi

if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Run directory does not exist: $RUN_DIR"
    exit 1
fi

echo "=========================================="
echo "DMD Pipeline - Step 3: Evaluate/Forecast"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Run dir: $RUN_DIR"
echo ""

cd "$(dirname "$0")/.."

python dmd/step_3_evaluate_dmd.py \
    --config "$CONFIG_FILE" \
    --run-dir "$RUN_DIR"

echo ""
echo "Pipeline complete!"
echo "Results saved to: $RUN_DIR"
