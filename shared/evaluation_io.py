"""
Shared evaluation I/O utilities for standardized output across methods.

Provides helpers for:
- Loading continuous reference data for full trajectory plots
- Standardized figure directory structure
- QoI y-limit computation
"""
import os
import numpy as np


def get_figure_dirs(run_dir: str) -> dict:
    """Return standardized figure subdirectory paths.

    Creates: figures/train/, figures/test/, figures/full_trajectory/
    """
    figures = os.path.join(run_dir, "figures")
    dirs = {
        "figures": figures,
        "train": os.path.join(figures, "train"),
        "test": os.path.join(figures, "test"),
        "full_trajectory": os.path.join(figures, "full_trajectory"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def load_full_trajectory_reference(
    ref_file: str,
    train_ref_offset: int,
    test_ref_offset: int,
    n_test: int,
    pde: str = "ns",
):
    """Load continuous reference data spanning train+val+test.

    Loads a single continuous slice [train_ref_offset, test_ref_offset + n_test)
    from the HDF5 file, avoiding gaps from the validation window.

    Returns
    -------
    dict with keys:
        ref_state : np.ndarray, shape (n_total, ny, nx) or (n_total, N)
        ref_energy : np.ndarray, shape (n_total,)
        ref_enstrophy : np.ndarray, shape (n_total,)
        n_total : int
        train_n_steps : int  — index where test portion begins
    """
    import h5py

    n_total = test_ref_offset + n_test - train_ref_offset
    train_n_steps = test_ref_offset - train_ref_offset

    state_key = "omega" if pde == "ns" else "u"

    with h5py.File(ref_file, "r") as f:
        ref_state = np.array(f[state_key][train_ref_offset:train_ref_offset + n_total])
        ref_energy = np.array(f["energy"][train_ref_offset:train_ref_offset + n_total])
        ref_enstrophy = np.array(f["enstrophy"][train_ref_offset:train_ref_offset + n_total])

    return {
        "ref_state": ref_state,
        "ref_energy": ref_energy,
        "ref_enstrophy": ref_enstrophy,
        "n_total": n_total,
        "train_n_steps": train_n_steps,
    }


def compute_qoi_ylims(ref_energy: np.ndarray, ref_enstrophy: np.ndarray, pad_frac=0.1):
    """Compute y-axis limits from reference QoI with padding."""
    energy_range = ref_energy.max() - ref_energy.min()
    energy_pad = pad_frac * energy_range if energy_range > 0 else 1.0
    enstrophy_range = ref_enstrophy.max() - ref_enstrophy.min()
    enstrophy_pad = pad_frac * enstrophy_range if enstrophy_range > 0 else 1.0
    return (
        (float(ref_energy.min() - energy_pad), float(ref_energy.max() + energy_pad)),
        (float(ref_enstrophy.min() - enstrophy_pad), float(ref_enstrophy.max() + enstrophy_pad)),
    )
