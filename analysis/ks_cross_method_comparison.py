"""
KS Cross-Method Comparison
===========================
Compare OpInf, DMD, and FNO predictions on the Kuramoto-Sivashinsky equation.

Generates publication-quality figures for the IEEE paper:
1. Cross-method QoI comparison (energy, enstrophy vs time)
2. Cross-method relative L2 state error vs time
3. Time-averaged spatial power spectra comparison
4. Valid prediction horizon (correlation vs time)
5. Summary metrics table (printed to stdout)

Usage::

    python analysis/ks_cross_method_comparison.py
    python analysis/ks_cross_method_comparison.py --output-dir results/comparison/

Author: Anthony Poole
"""

import os
import sys
import argparse
import logging
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup — repo is not an installable package
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from shared.plotting import setup_publication_style
from shared.physics import (
    compute_ks_psd,
    get_ks_grid_params,
)
from analysis.ks_loaders import (
    load_reference,
    load_all_methods,
    add_common_args,
    DT, L, N_GRID, TRAIN_START, TRAIN_END, TEST_END,
    COLOR_REF, COLOR_OPINF, COLOR_DMD, COLOR_FNO,
    CORR_THRESHOLD,
)


# =============================================================================
# LOGGING
# =============================================================================

def _setup_logger() -> logging.Logger:
    """Create a console-only logger."""
    logger = logging.getLogger("ks_cross_method")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        logger.addHandler(ch)
    return logger


# =============================================================================
# FIGURE 1: CROSS-METHOD QoI COMPARISON
# =============================================================================

def plot_cross_method_qoi(
    methods: list,
    ref: dict,
    grid: dict,
    output_dir: str,
    logger,
):
    """
    Plot energy and enstrophy vs time for all methods + reference DNS.

    Two-row figure with train/test regions and vertical boundary line.
    OpInf shows ±2σ ensemble shading.

    Parameters
    ----------
    methods : list[dict]
        Loaded method dicts.
    ref : dict
        Reference data.
    grid : dict
        KS grid parameters.
    output_dir : str
        Directory for saved figure.
    logger : logging.Logger
    """
    n_train = ref["u_train"].shape[0]
    n_test = ref["u_test"].shape[0]
    n_total = n_train + n_test
    t_start = TRAIN_START * DT
    t = t_start + np.arange(n_total) * DT
    t_boundary = t_start + n_train * DT

    ref_energy = np.concatenate([ref["energy_train"], ref["energy_test"]])
    ref_enstrophy = np.concatenate([ref["enstrophy_train"], ref["enstrophy_test"]])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # --- Energy panel ---
    ax = axes[0]
    ax.plot(t, ref_energy, color=COLOR_REF, linewidth=1.5, label="Reference DNS")
    for m in methods:
        energy = np.concatenate([m["energy_train"], m["energy_test"]])
        # Pad if FNO has fewer points (autoregressive offset)
        if len(energy) < n_total:
            energy = np.pad(energy, (n_total - len(energy), 0),
                            constant_values=np.nan)
        ax.plot(t, energy, color=m["color"], linewidth=1.2, label=m["name"])
        if m.get("has_ensemble"):
            # Compute stats separately (train/test may have different model counts)
            mean_e_tr = np.mean(m["ens_energy_train"], axis=0)
            std_e_tr = np.std(m["ens_energy_train"], axis=0)
            mean_e_te = np.mean(m["ens_energy_test"], axis=0)
            std_e_te = np.std(m["ens_energy_test"], axis=0)
            mean_e = np.concatenate([mean_e_tr, mean_e_te])
            std_e = np.concatenate([std_e_tr, std_e_te])
            ax.fill_between(t, mean_e - 2 * std_e, mean_e + 2 * std_e,
                            color=m["color"], alpha=0.15, label=f"{m['name']} ±2σ")
    ax.axvline(t_boundary, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="Train / Test")
    ax.set_ylabel(r"Energy $\langle u^2 \rangle / 2$")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Cross-Method QoI Comparison — KS Equation")

    # --- Enstrophy panel ---
    ax = axes[1]
    ax.plot(t, ref_enstrophy, color=COLOR_REF, linewidth=1.5, label="Reference DNS")
    for m in methods:
        enstrophy = np.concatenate([m["enstrophy_train"], m["enstrophy_test"]])
        if len(enstrophy) < n_total:
            enstrophy = np.pad(enstrophy, (n_total - len(enstrophy), 0),
                               constant_values=np.nan)
        ax.plot(t, enstrophy, color=m["color"], linewidth=1.2, label=m["name"])
        if m.get("has_ensemble"):
            mean_p_tr = np.mean(m["ens_enstrophy_train"], axis=0)
            std_p_tr = np.std(m["ens_enstrophy_train"], axis=0)
            mean_p_te = np.mean(m["ens_enstrophy_test"], axis=0)
            std_p_te = np.std(m["ens_enstrophy_test"], axis=0)
            mean_p = np.concatenate([mean_p_tr, mean_p_te])
            std_p = np.concatenate([std_p_tr, std_p_te])
            ax.fill_between(t, mean_p - 2 * std_p, mean_p + 2 * std_p,
                            color=m["color"], alpha=0.15, label=f"{m['name']} ±2σ")
    ax.axvline(t_boundary, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Enstrophy $\langle u_x^2 \rangle$")
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "ks_cross_method_qoi.pdf")
    plt.savefig(path)
    plt.close(fig)
    logger.info(f"  Saved {path}")


# =============================================================================
# FIGURE 2: STATE ERROR COMPARISON
# =============================================================================

def plot_cross_method_state_error(
    methods: list,
    ref: dict,
    output_dir: str,
    logger,
):
    """
    Plot relative L2 state error vs time for all methods on same axes.

    Parameters
    ----------
    methods : list[dict]
        Loaded method dicts.
    ref : dict
        Reference data (u_train, u_test).
    output_dir : str
        Output directory.
    logger : logging.Logger
    """
    n_train = ref["u_train"].shape[0]
    n_test = ref["u_test"].shape[0]
    n_total = n_train + n_test
    t_start = TRAIN_START * DT
    t = t_start + np.arange(n_total) * DT
    t_boundary = t_start + n_train * DT

    ref_u = np.concatenate([ref["u_train"], ref["u_test"]], axis=0)  # (n_total, N)
    ref_norms = np.linalg.norm(ref_u, axis=1)
    ref_norms = np.where(ref_norms < 1e-12, 1e-12, ref_norms)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    for m in methods:
        pred_u = np.concatenate([m["u_train"], m["u_test"]], axis=0)
        # Handle length mismatch from FNO autoregressive offset
        min_len = min(pred_u.shape[0], ref_u.shape[0])
        offset = ref_u.shape[0] - min_len
        l2_err = np.linalg.norm(pred_u[:min_len] - ref_u[offset:], axis=1) / ref_norms[offset:]
        t_plot = t[offset:]
        ax.semilogy(t_plot, l2_err, color=m["color"], linewidth=1.2, label=m["name"])

    ax.axvline(t_boundary, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="Train / Test")
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("Cross-Method State Prediction Error — KS")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "ks_cross_method_state_error.pdf")
    plt.savefig(path)
    plt.close(fig)
    logger.info(f"  Saved {path}")


# =============================================================================
# FIGURE 3: POWER SPECTRA COMPARISON
# =============================================================================

def plot_cross_method_power_spectra(
    methods: list,
    ref: dict,
    grid: dict,
    output_dir: str,
    logger,
):
    """
    Compare time-averaged spatial power spectra (test region).

    Parameters
    ----------
    methods : list[dict]
        Loaded method dicts.
    ref : dict
        Reference data.
    grid : dict
        KS grid parameters.
    output_dir : str
        Output directory.
    logger : logging.Logger
    """
    dx = grid["dx"]
    u_ref = ref["u_test"]  # (n_test, N)

    k_ref, psd_ref = compute_ks_psd(u_ref, dx)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.loglog(k_ref[1:], psd_ref[1:], color=COLOR_REF, linewidth=1.5,
              label="Reference DNS")

    for m in methods:
        k, psd = compute_ks_psd(m["u_test"], dx)
        ax.loglog(k[1:], psd[1:], color=m["color"], linewidth=1.2,
                  label=m["name"], alpha=0.85)

    ax.set_xlabel(r"Wavenumber $k$")
    ax.set_ylabel(r"Power Spectral Density")
    ax.set_title("Time-Averaged Spatial Power Spectrum — KS (Test Region)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, "ks_cross_method_power_spectra.pdf")
    plt.savefig(path)
    plt.close(fig)
    logger.info(f"  Saved {path}")


# =============================================================================
# FIGURE 4: VALID PREDICTION HORIZON
# =============================================================================

def _rolling_correlation(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Compute pointwise Pearson correlation in space at each timestep.

    Parameters
    ----------
    pred : np.ndarray, shape (n_time, N)
    ref : np.ndarray, shape (n_time, N)

    Returns
    -------
    np.ndarray, shape (n_time,)
        Correlation coefficient per timestep.
    """
    pred_c = pred - pred.mean(axis=1, keepdims=True)
    ref_c = ref - ref.mean(axis=1, keepdims=True)
    num = np.sum(pred_c * ref_c, axis=1)
    denom = (np.linalg.norm(pred_c, axis=1) * np.linalg.norm(ref_c, axis=1))
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return num / denom


def plot_valid_prediction_horizon(
    methods: list,
    ref: dict,
    output_dir: str,
    logger,
):
    """
    Plot correlation between predicted and reference test trajectories.

    A horizontal line at the correlation threshold defines the valid
    prediction horizon for each method.

    Parameters
    ----------
    methods : list[dict]
        Loaded method dicts.
    ref : dict
        Reference data.
    output_dir : str
        Output directory.
    logger : logging.Logger
    """
    u_ref_test = ref["u_test"]
    n_test = u_ref_test.shape[0]
    t_test = np.arange(n_test) * DT

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for m in methods:
        u_pred = m["u_test"]
        min_len = min(u_pred.shape[0], n_test)
        corr = _rolling_correlation(u_pred[:min_len], u_ref_test[:min_len])
        ax.plot(t_test[:min_len], corr, color=m["color"], linewidth=1.2,
                label=m["name"])

        # Find horizon
        below = np.where(corr < CORR_THRESHOLD)[0]
        if len(below) > 0:
            horizon_t = t_test[below[0]]
            ax.axvline(horizon_t, color=m["color"], linestyle=":", linewidth=0.8,
                       alpha=0.7)
            logger.info(f"  {m['name']} valid horizon (ρ>{CORR_THRESHOLD}): "
                        f"t={horizon_t:.1f} ({below[0]} steps)")

    ax.axhline(CORR_THRESHOLD, color="gray", linestyle="--", linewidth=1,
               alpha=0.6, label=f"Threshold (ρ={CORR_THRESHOLD})")
    ax.set_xlabel("Test Time")
    ax.set_ylabel("Spatial Correlation ρ")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Valid Prediction Horizon — KS (Test Region)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "ks_cross_method_prediction_horizon.pdf")
    plt.savefig(path)
    plt.close(fig)
    logger.info(f"  Saved {path}")


# =============================================================================
# SUMMARY METRICS TABLE
# =============================================================================

def print_summary_table(methods: list, ref: dict, grid: dict, logger):
    """
    Print a formatted table of key metrics for each method.

    Metrics include: mean relative L2 error (train/test),
    QoI mean relative error (energy/enstrophy on test),
    and valid prediction horizon.

    Parameters
    ----------
    methods : list[dict]
        Loaded method dicts.
    ref : dict
        Reference data.
    grid : dict
        KS grid parameters.
    logger : logging.Logger
    """
    u_ref_train = ref["u_train"]
    u_ref_test = ref["u_test"]
    ref_energy_test = ref["energy_test"]
    ref_enstrophy_test = ref["enstrophy_test"]

    header = (
        f"{'Method':<8} | {'L2 Train':>10} | {'L2 Test':>10} | "
        f"{'E err (test)':>12} | {'P err (test)':>12} | {'Horizon':>10}"
    )
    sep = "-" * len(header)

    logger.info("")
    logger.info("=" * len(header))
    logger.info("  KS Cross-Method Summary Metrics")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info(sep)

    for m in methods:
        # Relative L2 state error
        def _mean_rel_l2(pred, ref_u):
            min_len = min(pred.shape[0], ref_u.shape[0])
            offset = ref_u.shape[0] - min_len
            diff = pred[:min_len] - ref_u[offset:]
            norms = np.linalg.norm(ref_u[offset:], axis=1)
            norms = np.where(norms < 1e-12, 1e-12, norms)
            return np.mean(np.linalg.norm(diff, axis=1) / norms)

        l2_train = _mean_rel_l2(m["u_train"], u_ref_train)
        l2_test = _mean_rel_l2(m["u_test"], u_ref_test)

        # QoI relative errors (test)
        e_err = np.abs(
            np.mean(m["energy_test"]) - np.mean(ref_energy_test)
        ) / max(np.abs(np.mean(ref_energy_test)), 1e-12)
        p_err = np.abs(
            np.mean(m["enstrophy_test"]) - np.mean(ref_enstrophy_test)
        ) / max(np.abs(np.mean(ref_enstrophy_test)), 1e-12)

        # Prediction horizon
        min_len = min(m["u_test"].shape[0], u_ref_test.shape[0])
        corr = _rolling_correlation(m["u_test"][:min_len], u_ref_test[:min_len])
        below = np.where(corr < CORR_THRESHOLD)[0]
        horizon = f"{below[0] * DT:.1f}" if len(below) > 0 else f">{min_len * DT:.0f}"

        logger.info(
            f"{m['name']:<8} | {l2_train:>10.4f} | {l2_test:>10.4f} | "
            f"{e_err:>12.4e} | {p_err:>12.4e} | {horizon:>10}"
        )

    logger.info(sep)
    logger.info("")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="KS cross-method comparison for IEEE paper."
    )
    add_common_args(p)
    return p.parse_args()


def main():
    """Entry point for KS cross-method comparison analysis."""
    args = parse_args()
    logger = _setup_logger()

    logger.info("KS Cross-Method Comparison")
    logger.info(f"  Output dir: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Publication styling
    setup_publication_style()

    # Grid parameters
    grid = get_ks_grid_params(L=L, N=N_GRID)

    # ------------------------------------------------------------------ #
    # 1. Load reference DNS
    # ------------------------------------------------------------------ #
    logger.info("Loading reference DNS data ...")
    ref = load_reference(args.ks_data, grid)
    logger.info(f"  Train: {ref['u_train'].shape}, Test: {ref['u_test'].shape}")

    # ------------------------------------------------------------------ #
    # 2. Load method predictions
    # ------------------------------------------------------------------ #
    logger.info("Loading method predictions ...")
    methods = load_all_methods(args, grid, ref, logger)
    if not methods:
        logger.error("No methods loaded — nothing to compare.")
        sys.exit(1)
    logger.info(f"  Loaded {len(methods)} method(s): "
                f"{[m['name'] for m in methods]}")

    # ------------------------------------------------------------------ #
    # 3. Generate figures
    # ------------------------------------------------------------------ #
    logger.info("Generating figures ...")

    plot_cross_method_qoi(methods, ref, grid, args.output_dir, logger)
    plot_cross_method_state_error(methods, ref, args.output_dir, logger)
    plot_cross_method_power_spectra(methods, ref, grid, args.output_dir, logger)
    plot_valid_prediction_horizon(methods, ref, args.output_dir, logger)

    # ------------------------------------------------------------------ #
    # 4. Summary metrics
    # ------------------------------------------------------------------ #
    print_summary_table(methods, ref, grid, logger)

    logger.info("Done.")


if __name__ == "__main__":
    main()
