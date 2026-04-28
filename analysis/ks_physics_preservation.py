"""
KS Physics Preservation Analysis
==================================
Assess how faithfully each surrogate method (OpInf, DMD, FNO) preserves
the physical properties of the Kuramoto-Sivashinsky equation.

Metrics computed:
1. Power spectral density (spatial wavenumber content)
2. Field value PDF (invariant measure of the attractor)
3. Spatial autocorrelation C(Δx)
4. Energy rate balance (dE/dt from PDE terms vs finite differences)
5. Statistical moments (mean, variance, skewness, kurtosis)

All metrics are computed on the **test region** predictions.

Usage::

    python analysis/ks_physics_preservation.py
    python analysis/ks_physics_preservation.py --output-dir results/physics/

Author: Anthony Poole
"""

import os
import sys
import argparse
import logging
import numpy as np

import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Path setup — repo is not an installable package
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from shared.plotting import (
    setup_publication_style,
    plot_ks_cross_method_psd,
    plot_ks_cross_method_pdf,
    plot_ks_cross_method_autocorrelation,
    plot_ks_cross_method_energy_rate,
    plot_ks_cross_method_moments,
)
from shared.physics import (
    get_ks_grid_params,
    compute_ks_psd,
    compute_ks_field_pdf,
    compute_ks_spatial_autocorrelation,
    compute_ks_energy_rate,
    compute_ks_statistical_moments,
)
from analysis.ks_loaders import (
    load_reference,
    load_all_methods,
    add_common_args,
    DT, L, N_GRID,
    COLOR_REF,
)


# =============================================================================
# LOGGING
# =============================================================================

def _setup_logger() -> logging.Logger:
    """Create a console-only logger."""
    logger = logging.getLogger("ks_physics_preservation")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        logger.addHandler(ch)
    return logger


# =============================================================================
# SUMMARY METRICS
# =============================================================================

def _psd_relative_error(u_pred: np.ndarray, u_ref: np.ndarray, dx: float) -> float:
    """Relative L2 error of the time-averaged PSD (excluding DC)."""
    _, psd_ref = compute_ks_psd(u_ref, dx)
    _, psd_pred = compute_ks_psd(u_pred, dx)
    # Skip DC component (index 0)
    return float(
        np.linalg.norm(psd_pred[1:] - psd_ref[1:])
        / max(np.linalg.norm(psd_ref[1:]), 1e-12)
    )


def _pdf_kl_divergence(
    u_pred: np.ndarray, u_ref: np.ndarray, n_bins: int = 100
) -> float:
    """Symmetric KL divergence between field-value PDFs."""
    _, p = compute_ks_field_pdf(u_pred, n_bins)
    _, q = compute_ks_field_pdf(u_ref, n_bins)
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    return 0.5 * (kl_pq + kl_qp)


def _autocorrelation_error(
    u_pred: np.ndarray, u_ref: np.ndarray, dx: float
) -> float:
    """Relative L2 error of the spatial autocorrelation (first half)."""
    _, C_ref = compute_ks_spatial_autocorrelation(u_ref, dx)
    _, C_pred = compute_ks_spatial_autocorrelation(u_pred, dx)
    half = len(C_ref) // 2
    return float(
        np.linalg.norm(C_pred[:half] - C_ref[:half])
        / max(np.linalg.norm(C_ref[:half]), 1e-12)
    )


def _energy_budget_rmse(u: np.ndarray, dx: float, dt: float) -> float:
    """RMS energy budget residual (should be ~0 for perfect physics)."""
    rates = compute_ks_energy_rate(u, dx, dt)
    return float(np.sqrt(np.mean(rates["residual"] ** 2)))


def _moment_errors(
    u_pred: np.ndarray, u_ref: np.ndarray
) -> dict:
    """Relative error in time-averaged statistical moments."""
    mom_ref = compute_ks_statistical_moments(u_ref)
    mom_pred = compute_ks_statistical_moments(u_pred)
    errors = {}
    for key in ("mean", "variance", "skewness", "kurtosis"):
        ref_val = np.mean(mom_ref[key])
        pred_val = np.mean(mom_pred[key])
        denom = max(abs(ref_val), 1e-12)
        errors[key] = float(abs(pred_val - ref_val) / denom)
    return errors


def print_physics_summary(methods: list, ref: dict, grid: dict, logger):
    """
    Print physics-preservation metrics for both train (in-domain) and test
    (out-of-domain) regions, plus a degradation ratio.

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
    dx = grid["dx"]

    # ---- Compact per-metric table: Train | Test | Ratio ----
    metrics = [
        ("PSD err",   _psd_relative_error),
        ("KL div",    _pdf_kl_divergence),
        ("ACF err",   _autocorrelation_error),
        ("E-budget",  None),          # special case
        ("Var err",   None),          # from moments
        ("Skew err",  None),
        ("Kurt err",  None),
    ]

    def _compute_all(u_pred, u_ref, dx_):
        """Compute all scalar metrics for a (pred, ref) pair."""
        psd = _psd_relative_error(u_pred, u_ref, dx_)
        kl = _pdf_kl_divergence(u_pred, u_ref)
        acf = _autocorrelation_error(u_pred, u_ref, dx_)
        eb = _energy_budget_rmse(u_pred, dx_, DT)
        merr = _moment_errors(u_pred, u_ref)
        return {
            "PSD err": psd,
            "KL div": kl,
            "ACF err": acf,
            "E-budget": eb,
            "Var err": merr["variance"],
            "Skew err": merr["skewness"],
            "Kurt err": merr["kurtosis"],
        }

    metric_names = ["PSD err", "KL div", "ACF err", "E-budget",
                    "Var err", "Skew err", "Kurt err"]

    # Header
    header = (
        f"{'Method':<8} | {'Metric':<10} | "
        f"{'Train':>10} | {'Test':>10} | {'Ratio':>8}"
    )
    sep = "-" * len(header)

    logger.info("")
    logger.info("=" * len(header))
    logger.info("  KS Physics Preservation — Train vs Test")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info(sep)

    for m in methods:
        # Handle FNO length mismatch: trim to shorter of pred/ref
        u_train_pred = m["u_train"]
        u_test_pred = m["u_test"]
        u_train_ref = ref["u_train"]
        u_test_ref = ref["u_test"]

        min_tr = min(u_train_pred.shape[0], u_train_ref.shape[0])
        min_te = min(u_test_pred.shape[0], u_test_ref.shape[0])

        vals_train = _compute_all(
            u_train_pred[:min_tr], u_train_ref[:min_tr], dx
        )
        vals_test = _compute_all(
            u_test_pred[:min_te], u_test_ref[:min_te], dx
        )

        for i, name in enumerate(metric_names):
            tr = vals_train[name]
            te = vals_test[name]
            ratio = te / tr if tr > 1e-12 else float("inf")
            label = m["name"] if i == 0 else ""
            logger.info(
                f"{label:<8} | {name:<10} | "
                f"{tr:>10.4e} | {te:>10.4e} | {ratio:>8.1f}x"
            )
        logger.info(sep)

    # Reference self-consistency (energy budget only)
    ref_tr_budget = _energy_budget_rmse(ref["u_train"], dx, DT)
    ref_te_budget = _energy_budget_rmse(ref["u_test"], dx, DT)
    logger.info(
        f"{'Ref DNS':<8} | {'E-budget':<10} | "
        f"{ref_tr_budget:>10.4e} | {ref_te_budget:>10.4e} | {'—':>8}"
    )
    logger.info(sep)
    logger.info("")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="KS physics preservation analysis for IEEE paper."
    )
    add_common_args(p)
    return p.parse_args()


def main():
    """Entry point for KS physics preservation analysis."""
    args = parse_args()
    logger = _setup_logger()

    logger.info("KS Physics Preservation Analysis")
    logger.info(f"  Output dir: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    setup_publication_style()

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
    # 3. Generate physics-preservation figures
    # ------------------------------------------------------------------ #
    logger.info("Generating physics-preservation figures ...")
    u_ref_test = ref["u_test"]
    out = args.output_dir

    plot_ks_cross_method_psd(
        methods, u_ref_test, grid,
        os.path.join(out, "ks_physics_psd.pdf"), logger,
        color_ref=COLOR_REF,
    )
    plot_ks_cross_method_pdf(
        methods, u_ref_test,
        os.path.join(out, "ks_physics_pdf.pdf"), logger,
        color_ref=COLOR_REF,
    )
    plot_ks_cross_method_autocorrelation(
        methods, u_ref_test, grid,
        os.path.join(out, "ks_physics_autocorrelation.pdf"), logger,
        color_ref=COLOR_REF,
    )
    plot_ks_cross_method_energy_rate(
        methods, u_ref_test, grid, DT,
        os.path.join(out, "ks_physics_energy_rate.pdf"), logger,
        color_ref=COLOR_REF,
    )
    plot_ks_cross_method_moments(
        methods, u_ref_test, DT,
        os.path.join(out, "ks_physics_moments.pdf"), logger,
        color_ref=COLOR_REF,
    )

    # ------------------------------------------------------------------ #
    # 4. Summary metrics
    # ------------------------------------------------------------------ #
    print_physics_summary(methods, ref, grid, logger)

    logger.info("Done.")


if __name__ == "__main__":
    main()
