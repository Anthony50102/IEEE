"""
KS Cross-Method Data Loaders
=============================
Shared loading / reconstruction helpers used by all KS analysis scripts.

Each method stores predictions differently; the loaders here reconstruct
full-state arrays ``u`` of shape ``(n_time, N)`` and compute QoI
(energy, enstrophy) so that downstream analysis code can treat all
methods identically.

Author: Anthony Poole
"""

import os
import logging
import numpy as np

from shared.physics import compute_ks_qoi_timeseries
from shared.data_io import load_ks_timeseries


# =============================================================================
# CONSTANTS
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

KS_DATA_FILE = os.path.join(
    REPO_ROOT,
    "data/ks/ks_sim_step0.1_L100_N200_steps2000_20260223093716_4479.h5",
)

DEFAULT_OPINF_DIR = os.path.join(
    REPO_ROOT, "local_output/20260316_090726_opinf_ks_temporal_split"
)
DEFAULT_DMD_DIR = os.path.join(
    REPO_ROOT, "local_output/20260316_090304_dmd_ks_temporal_split"
)
DEFAULT_FNO_DIR = os.path.join(
    REPO_ROOT, "local_output/fno_ks_temporal_split_20260309_192216"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(REPO_ROOT), "IEEE-CiSE-Special-Issue/results/comparison/"
)

# Physics parameters
DT = 0.1
L = 100.0
N_GRID = 200
TRAIN_START = 1250
TRAIN_END = 1800
TEST_END = 2000

# Method colours (consistent across all figures)
COLOR_REF = "black"
COLOR_OPINF = "#1f77b4"   # blue
COLOR_DMD = "#d62728"     # red
COLOR_FNO = "#ff7f0e"     # orange

CORR_THRESHOLD = 0.8


# =============================================================================
# REFERENCE DATA
# =============================================================================

def load_reference(ks_file: str, grid: dict) -> dict:
    """
    Load reference DNS data for the train+test window.

    Parameters
    ----------
    ks_file : str
        Path to KS HDF5 file.
    grid : dict
        Grid parameters from ``get_ks_grid_params``.

    Returns
    -------
    dict
        Keys: u_train, u_test, energy_train, energy_test,
        enstrophy_train, enstrophy_test.
    """
    n_train = TRAIN_END - TRAIN_START
    n_test = TEST_END - TRAIN_END

    u_train, energy_train, enstrophy_train = load_ks_timeseries(
        ks_file, max_timesteps=n_train, start_timestep=TRAIN_START
    )
    u_test, energy_test, enstrophy_test = load_ks_timeseries(
        ks_file, max_timesteps=n_test, start_timestep=TRAIN_END
    )

    return {
        "u_train": u_train,       # (n_train, N)
        "u_test": u_test,         # (n_test, N)
        "energy_train": energy_train,
        "energy_test": energy_test,
        "enstrophy_train": enstrophy_train,
        "enstrophy_test": enstrophy_test,
    }


# =============================================================================
# METHOD RECONSTRUCTIONS
# =============================================================================

def reconstruct_opinf(run_dir: str, grid: dict, ref: dict, logger) -> dict:
    """
    Load OpInf predictions and reconstruct full states.

    Returns dict with keys: name, color, u_train, u_test,
    energy_train, energy_test, enstrophy_train, enstrophy_test,
    has_ensemble (bool), and optionally ensemble arrays.
    """
    ens = np.load(os.path.join(run_dir, "ensemble_predictions.npz"))
    Ur = np.load(os.path.join(run_dir, "POD_basis_Ur.npy"))
    ic = np.load(os.path.join(run_dir, "initial_conditions.npz"))
    mean_train = ic["train_temporal_mean"]
    mean_test = ic["test_temporal_mean"]

    X_train_all = ens["train_traj_0_X_OpInf"]  # (n_models_train, n_train, r)
    X_test_all = ens["test_traj_0_X_OpInf"]    # (n_models_test, n_test, r)
    n_models_train = X_train_all.shape[0]
    n_models_test = X_test_all.shape[0]

    n_train = ref["u_train"].shape[0]
    n_test = ref["u_test"].shape[0]
    dx = grid["dx"]

    ens_energy_train = np.zeros((n_models_train, n_train))
    ens_enstrophy_train = np.zeros((n_models_train, n_train))
    ens_energy_test = np.zeros((n_models_test, n_test))
    ens_enstrophy_test = np.zeros((n_models_test, n_test))

    u_train_mean = np.zeros((n_train, N_GRID), dtype=np.float64)
    u_test_mean = np.zeros((n_test, N_GRID), dtype=np.float64)

    for m in range(n_models_train):
        Q_tr = Ur @ X_train_all[m].T + mean_train[:, None]
        u_tr = Q_tr.T
        e, p = compute_ks_qoi_timeseries(u_tr, dx)
        ens_energy_train[m] = e
        ens_enstrophy_train[m] = p
        u_train_mean += u_tr

    u_train_mean /= n_models_train

    for m in range(n_models_test):
        Q_te = Ur @ X_test_all[m].T + mean_test[:, None]
        u_te = Q_te.T
        e, p = compute_ks_qoi_timeseries(u_te, dx)
        ens_energy_test[m] = e
        ens_enstrophy_test[m] = p
        u_test_mean += u_te

    u_test_mean /= n_models_test

    logger.info(f"  OpInf: loaded {n_models_train} train / {n_models_test} test "
                f"ensemble models, train={n_train}, test={n_test}")

    return {
        "name": "OpInf",
        "color": COLOR_OPINF,
        "u_train": u_train_mean,
        "u_test": u_test_mean,
        "energy_train": np.mean(ens_energy_train, axis=0),
        "energy_test": np.mean(ens_energy_test, axis=0),
        "enstrophy_train": np.mean(ens_enstrophy_train, axis=0),
        "enstrophy_test": np.mean(ens_enstrophy_test, axis=0),
        "has_ensemble": True,
        "ens_energy_train": ens_energy_train,
        "ens_energy_test": ens_energy_test,
        "ens_enstrophy_train": ens_enstrophy_train,
        "ens_enstrophy_test": ens_enstrophy_test,
    }


def reconstruct_dmd(run_dir: str, grid: dict, ref: dict, logger) -> dict:
    """Load DMD predictions and reconstruct full states."""
    Xhat_train = np.load(os.path.join(run_dir, "Xhat_train.npy"))
    Xhat_test = np.load(os.path.join(run_dir, "Xhat_test.npy"))
    pod = np.load(os.path.join(run_dir, "pod_basis.npz"))
    Ur = pod["U_r"]
    mean = pod["mean"]

    Q_train = Ur @ Xhat_train.T + mean[:, None]
    Q_test = Ur @ Xhat_test.T + mean[:, None]

    u_train = Q_train.T
    u_test = Q_test.T
    dx = grid["dx"]

    e_tr, p_tr = compute_ks_qoi_timeseries(u_train, dx)
    e_te, p_te = compute_ks_qoi_timeseries(u_test, dx)

    logger.info(f"  DMD: train={u_train.shape[0]}, test={u_test.shape[0]}")

    return {
        "name": "DMD",
        "color": COLOR_DMD,
        "u_train": u_train,
        "u_test": u_test,
        "energy_train": e_tr,
        "energy_test": e_te,
        "enstrophy_train": p_tr,
        "enstrophy_test": p_te,
        "has_ensemble": False,
    }


def reconstruct_fno(run_dir: str, grid: dict, ref: dict, logger) -> dict:
    """Load FNO full-state predictions."""
    sp = np.load(os.path.join(run_dir, "state_predictions.npz"))
    u_train = sp["train_predictions"][:, 0, :].astype(np.float64)
    u_test = sp["test_predictions"][:, 0, :].astype(np.float64)
    dx = grid["dx"]

    e_tr, p_tr = compute_ks_qoi_timeseries(u_train, dx)
    e_te, p_te = compute_ks_qoi_timeseries(u_test, dx)

    logger.info(f"  FNO: train={u_train.shape[0]}, test={u_test.shape[0]}")

    return {
        "name": "FNO",
        "color": COLOR_FNO,
        "u_train": u_train,
        "u_test": u_test,
        "energy_train": e_tr,
        "energy_test": e_te,
        "enstrophy_train": p_tr,
        "enstrophy_test": p_te,
        "has_ensemble": False,
    }


def load_all_methods(args, grid: dict, ref: dict, logger) -> list:
    """
    Try to load each method; skip gracefully on failure.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments with opinf_dir, dmd_dir, fno_dir.
    grid : dict
        KS grid parameters.
    ref : dict
        Reference data dictionary.
    logger : logging.Logger

    Returns
    -------
    list[dict]
        Successfully loaded method dictionaries.
    """
    loaders = [
        ("OpInf", args.opinf_dir, reconstruct_opinf),
        ("DMD",   args.dmd_dir,   reconstruct_dmd),
        ("FNO",   args.fno_dir,   reconstruct_fno),
    ]
    methods = []
    for name, run_dir, loader in loaders:
        if not os.path.isdir(run_dir):
            logger.warning(f"  {name} dir not found: {run_dir} — skipping")
            continue
        try:
            m = loader(run_dir, grid, ref, logger)
            methods.append(m)
        except Exception as exc:
            logger.warning(f"  {name} loading failed: {exc} — skipping")
    return methods


def add_common_args(parser):
    """
    Add common CLI arguments shared by all KS analysis scripts.

    Parameters
    ----------
    parser : argparse.ArgumentParser
    """
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory for saved figures.")
    parser.add_argument("--opinf-dir", default=DEFAULT_OPINF_DIR,
                        help="OpInf run directory.")
    parser.add_argument("--dmd-dir", default=DEFAULT_DMD_DIR,
                        help="DMD run directory.")
    parser.add_argument("--fno-dir", default=DEFAULT_FNO_DIR,
                        help="FNO run directory.")
    parser.add_argument("--ks-data", default=KS_DATA_FILE,
                        help="Path to KS HDF5 data file.")
