#!/bin/bash
# Run all NS pipelines (DMD, OpInf, FNO) across all Reynolds numbers locally
#
# Usage:
#   ./run_all_ns_local.sh                    # All methods, all Re
#   ./run_all_ns_local.sh --methods dmd opinf # Specific methods only
#   ./run_all_ns_local.sh --re 100 500       # Specific Re values only
#   ./run_all_ns_local.sh --skip-fno         # Skip FNO (slowest)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
RE_VALUES=(100 500 1000)
METHODS=(dmd opinf fno)
SKIP_FNO=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --methods) shift; METHODS=(); while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do METHODS+=("$1"); shift; done ;;
        --re)      shift; RE_VALUES=(); while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do RE_VALUES+=("$1"); shift; done ;;
        --skip-fno) SKIP_FNO=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if $SKIP_FNO; then
    METHODS=( "${METHODS[@]/fno}" )
fi

echo "=============================================="
echo "  NS Pipeline Sweep"
echo "  Methods: ${METHODS[*]}"
echo "  Re values: ${RE_VALUES[*]}"
echo "=============================================="
echo ""

FAILED=()
for RE in "${RE_VALUES[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        [ -z "$METHOD" ] && continue
        echo ">>> Running $METHOD at Re=$RE ..."
        SCRIPT="${SCRIPT_DIR}/run_${METHOD}_ns_local.sh"
        if [ ! -f "$SCRIPT" ]; then
            echo "  SKIP: $SCRIPT not found"
            continue
        fi
        if bash "$SCRIPT" "$RE"; then
            echo "  ✓ $METHOD Re=$RE succeeded"
        else
            echo "  ✗ $METHOD Re=$RE FAILED"
            FAILED+=("${METHOD}_re${RE}")
        fi
        echo ""
    done
done

echo "=============================================="
echo "  Summary"
echo "=============================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All runs succeeded!"
else
    echo "Failed runs: ${FAILED[*]}"
    exit 1
fi
