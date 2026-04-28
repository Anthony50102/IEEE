#!/bin/bash
# Run OpInf pipeline on NS (Navier-Stokes) data locally
# Usage: ./run_opinf_ns_local.sh [RE]   (default: RE=100)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IEEE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$IEEE_DIR"

PYTHON="${PYTHON:-python}"
RE="${1:-100}"
CONFIG="opinf/config/opinf_ns_re${RE}.yaml"
RUN_DIR="local_output/opinf_ns_re${RE}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    echo "Available configs:"; ls opinf/config/opinf_ns_*.yaml 2>/dev/null
    exit 1
fi

# Clean previous run
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

echo "=== OpInf NS Pipeline (Re=$RE) ==="
echo "Config: $CONFIG"
echo "Output: $RUN_DIR"
echo ""

echo "--- Step 1: Preprocessing ---"
$PYTHON opinf/step_1_preprocess_serial.py --config "$CONFIG" --run-dir "$RUN_DIR"

echo ""
echo "--- Step 2: Training ---"
$PYTHON opinf/step_2_train_serial.py --config "$CONFIG" --run-dir "$RUN_DIR"

echo ""
echo "--- Step 3: Evaluation ---"
$PYTHON opinf/step_3_evaluate.py --config "$CONFIG" --run-dir "$RUN_DIR"

echo ""
echo "=== Pipeline complete ==="
echo "Results in: $RUN_DIR"
