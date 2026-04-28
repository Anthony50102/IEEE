# Source this from any Frontera shell or SLURM script before running our code.
# Establishes the canonical environment: Python 3.9.2, parallel HDF5 1.10.4,
# system numpy/scipy/h5py/matplotlib (hardware-tuned BLAS).
#
#   source scripts/frontera/env.sh
#
# WHY: the user's bashrc hard-codes PYTHONPATH to python3.7's site-packages,
# which forces 3.7-built C extensions into 3.9 interpreters. We strip it.

unset PYTHONPATH
module reset >/dev/null 2>&1 || true
module load python3/3.9.2 phdf5/1.10.4
