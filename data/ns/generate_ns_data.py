"""
Generate 2D Navier-Stokes training and test data.

Produces HDF5 files for decaying turbulence at Re=100 on a 256x256 periodic grid.
Uses the pseudospectral ω-ψ solver from ns_solver.py.

Usage:
    python generate_ns_data.py [--output-dir ./]

Author: Anthony Poole
"""

import os
import sys
import numpy as np

# Add repo root so we can import the solver
sys.path.insert(0, os.path.dirname(__file__))
from ns_solver import solve_ns2d, save_to_hdf5


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

PARAMS = {
    "Re": 100.0,
    "nx": 256,
    "ny": 256,
    "Lx": 2 * np.pi,
    "dt": 1e-3,
    "n_steps": 20000,     # 20 time units at dt=1e-3
    "save_every": 20,     # dt_save = 0.02, gives 1001 snapshots
    "k_peak": 4.0,
    "amplitude": 1.0,
}

# Training trajectory: long run from seed=42
TRAIN_SEEDS = [42]

# Test trajectories: different ICs
TEST_SEEDS = [123, 456]


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate 2D NS datasets")
    parser.add_argument("--output-dir", type=str, default=os.path.dirname(__file__),
                        help="Output directory for HDF5 files")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with reduced resolution (64x64, 2000 steps)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    params = PARAMS.copy()
    if args.quick:
        params["nx"] = 64
        params["ny"] = 64
        params["n_steps"] = 2000
        params["save_every"] = 10
        print("=== QUICK MODE: 64x64, 2000 steps ===\n")

    all_seeds = TRAIN_SEEDS + TEST_SEEDS
    for seed in all_seeds:
        role = "train" if seed in TRAIN_SEEDS else "test"
        print(f"\n{'='*60}")
        print(f"Generating {role} trajectory (seed={seed})")
        print(f"{'='*60}")

        result = solve_ns2d(seed=seed, **params)

        filename = (
            f"ns2d_re{int(params['Re'])}"
            f"_{params['nx']}x{params['ny']}"
            f"_steps{params['n_steps']}"
            f"_seed{seed}.h5"
        )
        filepath = os.path.join(args.output_dir, filename)
        save_to_hdf5(result, filepath)

        # Summary
        E = result["energy"]
        Z = result["enstrophy"]
        print(f"  Energy decay: {E[0]:.6e} → {E[-1]:.6e} (ratio: {E[-1]/E[0]:.4f})")
        print(f"  Enstrophy decay: {Z[0]:.6e} → {Z[-1]:.6e} (ratio: {Z[-1]/Z[0]:.4f})")

    print(f"\n{'='*60}")
    print(f"All datasets written to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
