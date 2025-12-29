#!/bin/bash
# Local DMD Pipeline - Full Run (Steps 1-3)
# Usage: ./local_dmd_full_1train_5test.sh

set -e

CONFIG_FILE="config/local_dmd_1train_5test.yaml"

echo "=========================================="
echo "DMD Pipeline - Full Run (Steps 1-3)"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo ""

cd "$(dirname "$0")/.."

# Step 1: Preprocessing (uses OpInf step 1)
echo "=========================================="
echo "Step 1: Preprocessing (POD computation)"
echo "=========================================="

# Run step 1 and capture output to get run directory
OUTPUT=$(python opinf/step_1_preprocess.py --config "$CONFIG_FILE" 2>&1 | tee /dev/tty)

# Extract run directory from output
RUN_DIR=$(echo "$OUTPUT" | grep -o "Run directory: .*" | head -1 | cut -d' ' -f3)

if [ -z "$RUN_DIR" ]; then
    echo "Error: Could not determine run directory from Step 1 output"
    exit 1
fi

echo ""
echo "Run directory: $RUN_DIR"

# Step 2: Fit DMD Model
echo ""
echo "=========================================="
echo "Step 2: Fit DMD Model"
echo "=========================================="

python dmd/step_2_fit_dmd.py \
    --config "$CONFIG_FILE" \
    --run-dir "$RUN_DIR"

# Step 3: Evaluate/Forecast
echo ""
echo "=========================================="
echo "Step 3: Evaluate/Forecast"
echo "=========================================="

python dmd/step_3_evaluate_dmd.py \
    --config "$CONFIG_FILE" \
    --run-dir "$RUN_DIR"

echo ""
echo "=========================================="
echo "DMD Pipeline Complete!"
echo "=========================================="
echo "Results saved to: $RUN_DIR"
