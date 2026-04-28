"""Cross-check our hw.physics Gamma_n against hw2d's stored values.

Author: Anthony Poole

Loads a trajectory, recomputes Gamma_n with hw.physics.gamma_timeseries
on the saved fields, and compares to the per-frame `gamma_n` array that
hw2d stored at simulation resolution.

If the relative error is small (< ~1e-3) on every frame, our physics
port is consistent with hw2d's. Any larger discrepancy means either:
  - sign/transpose convention mismatch (most likely)
  - dx/grid units mismatch
  - hw2d is computing on omega rather than density-phi (check schema)

Usage:
    python -m hw.crosscheck <out_dir>/trajectory.h5
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

from hw.physics import gamma_timeseries, grid


def main(h5_path: str) -> int:
    p = Path(h5_path)
    with h5py.File(p, "r") as hf:
        if "gamma_n" not in hf:
            print("FAIL: no 'gamma_n' in HDF5 (hw2d may not have stored it).")
            return 2
        density = np.asarray(hf["density"], dtype=np.float64)
        phi = np.asarray(hf["phi"], dtype=np.float64)
        gn_native = np.asarray(hf["gamma_n"], dtype=np.float64)
        k0 = float(hf.attrs.get("k0", 0.15))
        c1 = float(hf.attrs.get("c1", 1.0))

    g = grid(k0=k0, nx=density.shape[-1])
    gn_ours, _ = gamma_timeseries(density, phi, g["dx"], c1=c1)

    n = min(len(gn_native), len(gn_ours))
    a = gn_native[:n]
    b = gn_ours[:n]
    abs_err = np.abs(a - b)
    rel_err = abs_err / np.maximum(np.abs(a), 1e-12)

    print(f"frames compared:         {n}")
    print(f"hw2d <gamma_n>:          {a.mean():+.6e} +- {a.std():.2e}")
    print(f"ours <gamma_n>:          {b.mean():+.6e} +- {b.std():.2e}")
    print(f"max abs error:           {abs_err.max():.3e}")
    print(f"max rel error:           {rel_err.max():.3e}")
    print(f"mean rel error:          {rel_err.mean():.3e}")

    if rel_err.max() < 1e-3:
        print("PASS: hw.physics matches hw2d to within 1e-3 relative.")
        return 0
    if rel_err.max() < 1e-1:
        print("WARN: agreement loose (1e-3 .. 1e-1). Check conventions.")
        return 1
    print("FAIL: large disagreement. Likely sign / axis / dx mismatch.")
    return 2


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m hw.crosscheck <trajectory.h5>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
