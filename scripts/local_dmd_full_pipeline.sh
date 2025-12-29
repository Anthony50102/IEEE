#!/bin/bash
# Local DMD Full Pipeline (reusing Step 1 from OpInf)
# 
# This script runs the complete DMD pipeline:
#   1. Step 1: Preprocessing (shared with OpInf)
#   2. Save POD basis (needed for full-state reconstruction)
#   3. Step 2: Fit DMD model
#   4. Step 3: Evaluate predictions
#
# Usage: ./local_dmd_full_pipeline.sh [run_name]

set -e

CONFIG_FILE="config/local_dmd_1train_5test.yaml"
RUN_NAME=${1:-"dmd_1train_5test_$(date +%Y%m%d_%H%M%S)"}

cd "$(dirname "$0")/.."

echo "=========================================="
echo "DMD Full Pipeline"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Run name: $RUN_NAME"
echo ""

# Step 1: Run preprocessing (shared with OpInf)
echo "=========================================="
echo "Step 1: Preprocessing (POD)"
echo "=========================================="

RUN_DIR=$(python -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    cfg = yaml.safe_load(f)
import os
run_dir = os.path.join(cfg['paths']['output_base'], '$RUN_NAME')
print(run_dir)
")

echo "Run directory: $RUN_DIR"
mkdir -p "$RUN_DIR"

# Use serial preprocessing for local runs
python opinf/step_1_preprocess.py \
    --config "$CONFIG_FILE" \
    --run-name "$RUN_NAME"

if [ $? -ne 0 ]; then
    echo "Step 1 failed!"
    exit 1
fi

echo ""
echo "Step 1 completed successfully"
echo ""

# Step 1.5: Save POD basis for full-state reconstruction
echo "=========================================="
echo "Step 1.5: Save POD Basis"
echo "=========================================="

python dmd/save_pod_basis.py \
    --config "$CONFIG_FILE" \
    --run-dir "$RUN_DIR"

if [ $? -ne 0 ]; then
    echo "POD basis saving failed!"
    exit 1
fi

echo ""
echo "POD basis saved successfully"
echo ""

# Step 2: Fit DMD model
echo "=========================================="
echo "Step 2: Fit DMD Model (BOPDMD)"
echo "=========================================="

python dmd/step_2_fit_dmd.py \
    --config "$CONFIG_FILE" \
    --run-dir "$RUN_DIR"

if [ $? -ne 0 ]; then
    echo "Step 2 failed!"
    exit 1
fi

echo ""
echo "Step 2 completed successfully"
echo ""

# Step 3: Evaluate predictions
echo "=========================================="
echo "Step 3: Evaluate Predictions"
echo "=========================================="

python dmd/step_3_evaluate_dmd.py \
    --config "$CONFIG_FILE" \
    --run-dir "$RUN_DIR"

if [ $? -ne 0 ]; then
    echo "Step 3 failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "DMD Pipeline Complete!"
echo "=========================================="
echo "Results saved to: $RUN_DIR"
echo ""

# Print summary of outputs
if [ -f "$RUN_DIR/dmd_evaluation_metrics.yaml" ]; then
    echo "Evaluation metrics:"
    cat "$RUN_DIR/dmd_evaluation_metrics.yaml" | head -30
fi
