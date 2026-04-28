#!/bin/bash
# Run DMD pipeline on NS (Navier-Stokes) data locally
# Usage: ./run_dmd_ns_local.sh [RE]   (default: RE=100)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IEEE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$IEEE_DIR"

PYTHON="${PYTHON:-python}"
RE="${1:-100}"
CONFIG="dmd/config/dmd_ns_re${RE}.yaml"
RUN_DIR="local_output/dmd_ns_re${RE}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    echo "Available configs:"; ls dmd/config/dmd_ns_*.yaml 2>/dev/null
    exit 1
fi

# Clean previous run
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

echo "=== DMD NS Pipeline (Re=$RE) ==="
echo "Config: $CONFIG"
echo "Output: $RUN_DIR"
echo ""

echo "--- Step 1: Preprocessing ---"
$PYTHON dmd/step_1_preprocess.py --config "$CONFIG" --run-dir "$RUN_DIR"

echo ""
echo "--- Step 2: Training ---"
$PYTHON dmd/step_2_train.py --config "$CONFIG" --run-dir "$RUN_DIR"

echo ""
echo "--- Step 3: Evaluation ---"
$PYTHON dmd/step_3_evaluate.py --config "$CONFIG" --run-dir "$RUN_DIR"

echo ""
echo "=== Pipeline complete ==="
echo "Results in: $RUN_DIR"
