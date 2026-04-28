#!/bin/bash
# Set up the Python environment on Frontera for HW DNS.
#
# Frontera ships system numpy / scipy / h5py with hardware-tuned BLAS.
# We must NOT clobber those. This script installs only the missing
# pure-Python deps to ~/.local via `pip install --user --no-deps`, and
# then installs hw2d itself (editable) from the local clone, also
# --no-deps so it cannot drag in a fresh numpy.
#
# Run from the repo root on a login node:
#   cd $WORK/IEEE
#   bash scripts/frontera/setup_python.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HW2D_DIR="$REPO_ROOT/hw/hw2d"

# Canonical environment: Python 3.9.2 + parallel HDF5 1.10.4 + system
# numpy/scipy/h5py/matplotlib. Strips the bashrc-injected PYTHONPATH that
# would otherwise force python3.7 site-packages into a 3.9 interpreter.
source "$REPO_ROOT/scripts/frontera/env.sh"

if [ ! -d "$HW2D_DIR/src/hw2d" ]; then
    echo "ERROR: $HW2D_DIR/src/hw2d not found. Did you forget to clone hw2d?"
    echo "       git clone https://github.com/the-rccg/hw2d.git $HW2D_DIR"
    exit 1
fi

echo "==> python: $(which python3)  $(python3 --version)"

# Sanity-check the system stack BEFORE we touch anything.
echo "==> verifying system BLAS-linked stack:"
python3 - <<'PY'
import importlib
for pkg in ("numpy", "scipy", "h5py", "matplotlib"):
    try:
        m = importlib.import_module(pkg)
        loc = getattr(m, "__file__", "?")
        is_user = ".local" in loc
        print(f"  {pkg:12s}  {m.__version__:10s}  {'USER' if is_user else 'SYSTEM'}  {loc}")
        if is_user:
            print(f"    WARNING: {pkg} is from ~/.local, not the TACC system tree.")
    except ImportError:
        print(f"  {pkg:12s}  MISSING (will install pure-Python only; do not "
              "force-install numpy/scipy/h5py here)")
PY

# Pure-Python deps we need that are not part of the standard TACC stack.
# `--no-deps` is critical to avoid pulling fresh numpy/scipy.
# numba+llvmlite are not pure-Python but are the canonical pip wheels and
# will install cleanly to ~/.local without touching system numpy.
# numba==0.56.4 / llvmlite==0.39.1 is the last numba line that supports
# numpy 1.20 (Frontera's system numpy). Newer numba demands numpy>=1.22.
PY_DEPS=(pyyaml fire tqdm perfplot matplotx "llvmlite==0.39.1" "numba==0.56.4")

echo "==> installing pure-Python deps (--user --no-deps):"
for pkg in "${PY_DEPS[@]}"; do
    python3 -m pip install --user --no-deps --upgrade "$pkg"
done

echo "==> installing hw2d (--user --no-deps) from $HW2D_DIR"
# Non-editable: Frontera's system setuptools predates PEP 660, so editable
# installs fail with "build backend missing build_editable hook". hw2d is a
# stable upstream we don't intend to modify, so a regular install is fine.
# Allow build isolation so pip uses a modern setuptools to build the wheel
# (system setuptools is too old to honour the [project] table and would
# otherwise produce an "UNKNOWN-0.0.0" wheel). --no-deps still applies at
# install time, so no fresh numpy/scipy lands in ~/.local.
python3 -m pip install --user --no-deps "$HW2D_DIR"

echo "==> post-install check:"
python3 - <<'PY'
import hw2d, numpy, scipy, h5py, numba, yaml
print(f"  hw2d:   {hw2d.__file__}")
print(f"  numpy:  {numpy.__version__:10s}  {numpy.__file__}")
print(f"  scipy:  {scipy.__version__:10s}  {scipy.__file__}")
print(f"  h5py:   {h5py.__version__:10s}  {h5py.__file__}")
print(f"  numba:  {numba.__version__:10s}  {numba.__file__}")
PY

echo "==> done. numpy/scipy/h5py should still point to a TACC system path."
