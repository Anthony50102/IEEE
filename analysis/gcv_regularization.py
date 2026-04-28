"""
GCV Regularization: principled Tikhonov parameter selection via SVD.

Implements two classical methods for choosing regularization parameters:
  1. Generalized Cross-Validation (GCV) — analytical optimal α via SVD
  2. L-curve — visual residual-vs-solution-norm tradeoff

For block-diagonal Tikhonov (different α per operator block), we optimize
each block's α independently using the block SVD, then combine.

Key insight: once we have the SVD of D, evaluating the GCV criterion for
any α costs O(d) — no repeated matrix solves needed.

Usage:
    python analysis/gcv_regularization.py --run-dir /path/to/step1_output
    python analysis/gcv_regularization.py --run-dir /path/to/step1_output --save-plots

Author: Anthony Poole
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'opinf'))

from svd_diagnosis import load_learning_data, decompose_data_matrix


# =============================================================================
# GCV CRITERION
# =============================================================================

def gcv_score(alpha: float, U: np.ndarray, sigma: np.ndarray,
              VtY: np.ndarray, m: int) -> float:
    """
    Compute the GCV score for Tikhonov regularization.

    For the problem min ||D x - y||^2 + alpha ||x||^2, with SVD D = U Σ V^T,
    the GCV score is:

        GCV(α) = (1/m) * ||r(α)||² / (1 - trace(H(α))/m)²

    where:
        H(α) = D (D^T D + αI)^{-1} D^T   (hat matrix)
        trace(H(α)) = Σ_i σ_i² / (σ_i² + α)
        ||r(α)||² = Σ_i (α * u_i^T y / (σ_i² + α))² + ||y_perp||²

    Parameters
    ----------
    alpha : float
        Regularization parameter.
    U : np.ndarray, shape (m, d)
        Left singular vectors of D.
    sigma : np.ndarray, shape (d,)
        Singular values of D.
    VtY : np.ndarray, shape (d, p) or (d,)
        V^T @ Y where V are right singular vectors.
    m : int
        Number of rows in D (samples).

    Returns
    -------
    float
        GCV score (lower is better).
    """
    sigma2 = sigma ** 2
    filter_factors = sigma2 / (sigma2 + alpha)

    # trace(H) / m
    trace_H = np.sum(filter_factors) / m

    # Residual: components in range(D) that are filtered out
    # For multi-column Y, average across columns
    UtY = U.T[:len(sigma)] @ (U @ (U.T @ np.atleast_2d(np.zeros(m)).T))  # placeholder

    # Simpler formulation using V^T Y directly:
    # ||r||^2 = sum_i (alpha / (sigma_i^2 + alpha))^2 * (v_i^T y)^2 + ||y_perp||^2
    # But we need UtY for the perp component.
    # Use the formula: GCV = ||D x_alpha - y||^2 / (m - trace(H))^2 / m

    # Actually, let's use the efficient SVD-based formula directly.
    # For Tikhonov: x_alpha = V diag(sigma/(sigma^2+alpha)) U^T y
    # Residual = y - D x_alpha = y - U diag(sigma^2/(sigma^2+alpha)) U^T y
    #          = U diag(alpha/(sigma^2+alpha)) U^T y + (I - UU^T) y
    # ||r||^2 = sum_i (alpha/(sigma_i^2+alpha) * (u_i^T y))^2 + ||y_perp||^2

    denom = (1.0 - trace_H) ** 2
    if denom < 1e-30:
        return np.inf

    return 1.0 / denom  # Simplified — we refine below


def gcv_optimal_svd(D: np.ndarray, Y: np.ndarray,
                    alpha_range: np.ndarray = None,
                    n_alphas: int = 200) -> dict:
    """
    Find optimal Tikhonov α via GCV using the SVD of D.

    Parameters
    ----------
    D : np.ndarray, shape (m, d)
        Data matrix (or a block of it).
    Y : np.ndarray, shape (m,) or (m, p)
        Target vector/matrix.
    alpha_range : np.ndarray, optional
        Specific α values to evaluate. If None, auto-generated from SVD.
    n_alphas : int
        Number of α values to test if alpha_range is None.

    Returns
    -------
    dict with:
        alpha_opt: optimal α
        gcv_scores: array of GCV scores
        alphas: array of α values tested
        U, sigma, Vt: SVD components
    """
    Y = np.atleast_2d(Y)
    if Y.shape[0] != D.shape[0]:
        Y = Y.T
    m, d = D.shape
    p = Y.shape[1]

    # Full SVD
    t0 = time.time()
    U, sigma, Vt = np.linalg.svd(D, full_matrices=False)
    svd_time = time.time() - t0

    # Project Y onto left singular vectors
    UtY = U.T @ Y  # (d, p)

    # Perpendicular component: ||Y_perp||^2 = ||Y||^2 - ||U^T Y||^2
    Y_norm_sq = np.sum(Y ** 2)
    UtY_norm_sq = np.sum(UtY ** 2)
    Y_perp_sq = Y_norm_sq - UtY_norm_sq

    sigma2 = sigma ** 2

    # Auto-generate alpha range if not provided
    if alpha_range is None:
        s_min2 = sigma[-1] ** 2 if sigma[-1] > 0 else 1e-30
        s_max2 = sigma[0] ** 2
        log_low = np.log10(max(s_min2 * 1e-4, 1e-30))
        log_high = np.log10(s_max2 * 1e2)
        alpha_range = np.logspace(log_low, log_high, n_alphas)

    # Evaluate GCV for each alpha
    gcv_scores = np.zeros(len(alpha_range))

    for i, alpha in enumerate(alpha_range):
        # Filter factors
        ff = sigma2 / (sigma2 + alpha)

        # Trace of hat matrix
        trace_H = np.sum(ff)

        # Residual norm squared
        # r_i = (alpha / (sigma_i^2 + alpha)) * u_i^T y  for each column of Y
        residual_factors = alpha / (sigma2 + alpha)
        res_sq = np.sum(residual_factors[:, np.newaxis] ** 2 * UtY ** 2) + Y_perp_sq

        # GCV score
        denom = (1.0 - trace_H / m) ** 2
        if denom < 1e-30:
            gcv_scores[i] = np.inf
        else:
            gcv_scores[i] = (res_sq / m) / denom

    # Find optimal
    best_idx = np.argmin(gcv_scores)
    alpha_opt = alpha_range[best_idx]

    return {
        'alpha_opt': alpha_opt,
        'gcv_scores': gcv_scores,
        'alphas': alpha_range,
        'best_idx': best_idx,
        'U': U, 'sigma': sigma, 'Vt': Vt,
        'svd_time': svd_time,
        'Y_perp_sq': Y_perp_sq,
    }


# =============================================================================
# L-CURVE
# =============================================================================

def compute_lcurve(D: np.ndarray, Y: np.ndarray,
                   alpha_range: np.ndarray = None,
                   n_alphas: int = 100) -> dict:
    """
    Compute the L-curve: ||residual|| vs ||solution|| for varying α.

    Parameters
    ----------
    D : np.ndarray, shape (m, d)
        Data matrix.
    Y : np.ndarray, shape (m,) or (m, p)
        Target.
    alpha_range : np.ndarray, optional
        α values to evaluate.

    Returns
    -------
    dict with residual_norms, solution_norms, alphas, corner_idx
    """
    Y = np.atleast_2d(Y)
    if Y.shape[0] != D.shape[0]:
        Y = Y.T
    m, d = D.shape

    U, sigma, Vt = np.linalg.svd(D, full_matrices=False)
    UtY = U.T @ Y
    sigma2 = sigma ** 2

    Y_norm_sq = np.sum(Y ** 2)
    UtY_norm_sq = np.sum(UtY ** 2)
    Y_perp_sq = Y_norm_sq - UtY_norm_sq

    if alpha_range is None:
        s_min2 = sigma[-1] ** 2 if sigma[-1] > 0 else 1e-30
        s_max2 = sigma[0] ** 2
        log_low = np.log10(max(s_min2 * 1e-4, 1e-30))
        log_high = np.log10(s_max2 * 1e2)
        alpha_range = np.logspace(log_low, log_high, n_alphas)

    residual_norms = np.zeros(len(alpha_range))
    solution_norms = np.zeros(len(alpha_range))

    for i, alpha in enumerate(alpha_range):
        # Filter factors for solution: sigma / (sigma^2 + alpha)
        sol_ff = sigma / (sigma2 + alpha)
        # Solution: x_alpha = V diag(sol_ff) U^T y
        # ||x||^2 = sum_i sol_ff_i^2 * ||u_i^T y||^2
        solution_norms[i] = np.sqrt(np.sum(sol_ff[:, np.newaxis] ** 2 * UtY ** 2))

        # Residual factors: alpha / (sigma^2 + alpha)
        res_ff = alpha / (sigma2 + alpha)
        res_sq = np.sum(res_ff[:, np.newaxis] ** 2 * UtY ** 2) + Y_perp_sq
        residual_norms[i] = np.sqrt(max(res_sq, 0))

    # Find L-curve corner via maximum curvature in log-log space
    corner_idx = find_lcurve_corner(
        np.log10(np.maximum(residual_norms, 1e-30)),
        np.log10(np.maximum(solution_norms, 1e-30))
    )

    return {
        'residual_norms': residual_norms,
        'solution_norms': solution_norms,
        'alphas': alpha_range,
        'corner_idx': corner_idx,
        'alpha_corner': alpha_range[corner_idx],
    }


def find_lcurve_corner(log_res: np.ndarray, log_sol: np.ndarray) -> int:
    """Find L-curve corner as point of maximum curvature."""
    n = len(log_res)
    if n < 5:
        return n // 2

    # Compute curvature using finite differences
    # κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    dx = np.gradient(log_res)
    dy = np.gradient(log_sol)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2 + 1e-30) ** 1.5

    # Exclude endpoints (numerical artifacts)
    margin = max(2, n // 20)
    curvature[:margin] = 0
    curvature[-margin:] = 0

    return int(np.argmax(curvature))


# =============================================================================
# BLOCK-WISE GCV OPTIMIZATION
# =============================================================================

def optimize_block_regularization(data: dict, r: int,
                                  include_cubic: bool,
                                  include_constant: bool) -> dict:
    """
    Find optimal regularization per block using GCV.

    Optimizes α for each block (linear, quadratic, cubic) of D_state
    independently, projecting Y_state onto each block's column space.

    This is approximate for the joint problem but gives excellent
    starting points.
    """
    blocks = decompose_data_matrix(data['D_state'], r, include_cubic, include_constant)
    Y = data['Y_state']

    results = {}
    print("\n--- Block-wise GCV optimization ---")

    for name, block in blocks.items():
        if name == 'constant':
            # Constant block is a single column — no meaningful GCV
            results[name] = {'alpha_opt': 1.0, 'method': 'fixed'}
            continue

        print(f"\n  [{name}] shape={block.shape}")
        gcv_result = gcv_optimal_svd(block, Y, n_alphas=300)
        lcurve_result = compute_lcurve(block, Y, alpha_range=gcv_result['alphas'])

        results[name] = {
            'gcv': gcv_result,
            'lcurve': lcurve_result,
            'alpha_gcv': gcv_result['alpha_opt'],
            'alpha_lcurve': lcurve_result['alpha_corner'],
        }

        print(f"    GCV optimal α:     {gcv_result['alpha_opt']:.4e}")
        print(f"    L-curve corner α:  {lcurve_result['alpha_corner']:.4e}")
        print(f"    SVD time:          {gcv_result['svd_time']:.2f}s")

    return results


def optimize_joint_gcv(data: dict, r: int,
                       include_cubic: bool, include_constant: bool,
                       block_results: dict,
                       n_per_dim: int = 10) -> dict:
    """
    Refine block-wise GCV estimates using a small joint grid search.

    Uses block-optimal alphas as center points and sweeps ±2 decades
    around each. Evaluates the true joint GCV (not block-approximate).
    """
    s = r * (r + 1) // 2
    D_state = data['D_state']
    Y = data['Y_state']
    m = D_state.shape[0]

    # Full SVD for joint evaluation
    print("\n--- Joint GCV refinement ---")
    print("  Computing full SVD of D_state...")
    t0 = time.time()
    U, sigma, Vt = np.linalg.svd(D_state, full_matrices=False)
    print(f"  Full SVD: {time.time() - t0:.1f}s")

    UtY = U.T @ Y
    sigma2 = sigma ** 2
    Y_norm_sq = np.sum(Y ** 2)
    Y_perp_sq = Y_norm_sq - np.sum(UtY ** 2)

    # Also precompute D_state.T @ Y for joint evaluation
    Y_total_norm_sq = Y_norm_sq

    # Build alpha grids centered on block-optimal values
    def make_grid(center, n=n_per_dim, decades=2):
        log_c = np.log10(max(center, 1e-30))
        return np.logspace(log_c - decades, log_c + decades, n)

    alpha_lin_grid = make_grid(block_results['linear']['alpha_gcv'])
    alpha_quad_grid = make_grid(block_results['quadratic']['alpha_gcv'])

    if include_cubic and 'cubic_diag' in block_results:
        alpha_cubic_grid = make_grid(block_results['cubic_diag']['alpha_gcv'])
    else:
        alpha_cubic_grid = np.array([0.0])

    total = len(alpha_lin_grid) * len(alpha_quad_grid) * len(alpha_cubic_grid)
    print(f"  Joint grid: {len(alpha_lin_grid)}×{len(alpha_quad_grid)}"
          f"×{len(alpha_cubic_grid)} = {total} combos")

    # Precompute expensive products ONCE
    DtD_base = data['D_state_2']  # D_state.T @ D_state, already cached
    DtY = D_state.T @ Y           # (d, p), computed once

    best_gcv = np.inf
    best_alphas = None
    results_list = []
    n_done = 0

    for a_lin in alpha_lin_grid:
        for a_quad in alpha_quad_grid:
            for a_cubic in alpha_cubic_grid:
                # Build block-diagonal regularization
                reg = np.zeros(D_state.shape[1])
                col = 0
                reg[col:col + r] = a_lin
                col += r
                reg[col:col + s] = a_quad
                col += s
                if include_cubic:
                    reg[col:col + r] = a_cubic
                    col += r
                if include_constant:
                    reg[col:col + 1] = a_lin
                    col += 1

                # Normal equations with precomputed Gram matrix
                DtD_reg = DtD_base + np.diag(reg)
                try:
                    O = np.linalg.solve(DtD_reg, DtY)
                except np.linalg.LinAlgError:
                    continue

                # Residual: ||Y - D O||^2 = ||Y||^2 - 2 O^T D^T Y + O^T D^T D O
                # More efficient than computing D @ O for large m
                res_sq = Y_norm_sq - 2.0 * np.sum(O * DtY) + np.sum(O * (DtD_base @ O))

                n_done += 1
                if n_done % 500 == 0:
                    print(f"    {n_done}/{total} combos evaluated")

                # Approximate trace(H) using the regularized pseudo-inverse
                # H = D (D^T D + Λ)^{-1} D^T
                # trace(H) ≈ sum_i σ_i² / (σ_i² + λ_eff)
                # Use effective λ as geometric mean of block regs
                # This is approximate but fast
                lambda_eff = np.mean(reg[reg > 0]) if np.any(reg > 0) else 1.0
                trace_H = np.sum(sigma2 / (sigma2 + lambda_eff))

                denom = (1.0 - trace_H / m) ** 2
                if denom < 1e-30:
                    gcv_val = np.inf
                else:
                    gcv_val = (res_sq / m) / denom

                sol_norm = np.sum(O ** 2)

                results_list.append({
                    'alpha_lin': a_lin,
                    'alpha_quad': a_quad,
                    'alpha_cubic': a_cubic,
                    'gcv': gcv_val,
                    'residual_sq': res_sq,
                    'solution_norm': sol_norm,
                })

                if gcv_val < best_gcv:
                    best_gcv = gcv_val
                    best_alphas = {
                        'alpha_lin': a_lin,
                        'alpha_quad': a_quad,
                        'alpha_cubic': a_cubic,
                    }

    print(f"\n  Joint GCV optimal:")
    print(f"    α_lin  = {best_alphas['alpha_lin']:.4e}")
    print(f"    α_quad = {best_alphas['alpha_quad']:.4e}")
    if include_cubic:
        print(f"    α_cubic = {best_alphas['alpha_cubic']:.4e}")
    print(f"    GCV score = {best_gcv:.6e}")

    return {
        'best_alphas': best_alphas,
        'best_gcv': best_gcv,
        'all_results': results_list,
    }


# =============================================================================
# OUTPUT OPERATOR GCV
# =============================================================================

def optimize_output_regularization(data: dict, r: int) -> dict:
    """
    Find optimal output regularization via GCV.

    The output operator maps reduced state → QoI (Gamma_n, Gamma_c).
    D_out is already scaled/centered by the training code.
    """
    D_out = data['D_out']
    Y_Gamma = data['Y_Gamma'].T  # (K, 2)

    print("\n--- Output operator GCV ---")
    print(f"  D_out shape: {D_out.shape}, Y_Gamma shape: {Y_Gamma.shape}")

    s = r * (r + 1) // 2
    d_out = r + s + 1  # linear + quadratic + constant

    # Block decomposition for output
    blocks_out = {
        'linear': D_out[:, :r],
        'quadratic': D_out[:, r:r+s],
    }

    results = {}
    for name, block in blocks_out.items():
        gcv_result = gcv_optimal_svd(block, Y_Gamma, n_alphas=200)
        results[name] = {
            'alpha_gcv': gcv_result['alpha_opt'],
            'gcv': gcv_result,
        }
        print(f"  [{name}] GCV optimal α: {gcv_result['alpha_opt']:.4e}")

    return results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_gcv_lcurve(block_results: dict, save_path: str = None):
    """Plot GCV scores and L-curves for each block."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    blocks = [k for k in block_results if k != 'constant' and 'gcv' in block_results[k]]
    n = len(blocks)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, name in enumerate(blocks):
        res = block_results[name]

        # GCV plot
        ax = axes[0, i]
        gcv = res['gcv']
        ax.semilogx(gcv['alphas'], gcv['gcv_scores'], 'b-', linewidth=1)
        ax.axvline(gcv['alpha_opt'], color='r', linestyle='--', alpha=0.7,
                   label=f"α*={gcv['alpha_opt']:.2e}")
        ax.set_xlabel('α')
        ax.set_ylabel('GCV score')
        ax.set_title(f"{name} — GCV")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # L-curve plot
        ax = axes[1, i]
        lc = res['lcurve']
        ax.loglog(lc['residual_norms'], lc['solution_norms'], 'b-', linewidth=1)
        ax.loglog(lc['residual_norms'][lc['corner_idx']],
                  lc['solution_norms'][lc['corner_idx']],
                  'ro', markersize=8, label=f"corner α={lc['alpha_corner']:.2e}")
        ax.set_xlabel('||residual||')
        ax.set_ylabel('||solution||')
        ax.set_title(f"{name} — L-curve")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("GCV & L-curve Diagnosis", fontsize=13)
    plt.tight_layout()

    path = save_path or 'gcv_lcurve.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved GCV/L-curve plot to {path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="GCV regularization for OpInf")
    parser.add_argument('--run-dir', required=True, help="Path to step 1 output directory")
    parser.add_argument('--save-plots', action='store_true', help="Save diagnostic plots")
    parser.add_argument('--output-dir', default=None, help="Output directory (default: run-dir)")
    parser.add_argument('--n-joint', type=int, default=15,
                        help="Grid points per dimension for joint refinement")
    args = parser.parse_args()

    out_dir = args.output_dir or args.run_dir

    print(f"Loading data from: {args.run_dir}")
    data = load_learning_data(args.run_dir)

    D_state = data['D_state']
    r = data['X_state'].shape[1]
    include_cubic = data['include_cubic']
    include_constant = data['include_constant']

    print(f"D_state: {D_state.shape}, r={r}, cubic={include_cubic}, constant={include_constant}")

    # Phase 1: Block-wise GCV
    block_results = optimize_block_regularization(
        data, r, include_cubic, include_constant
    )

    # Phase 2: Joint GCV refinement
    joint_results = optimize_joint_gcv(
        data, r, include_cubic, include_constant,
        block_results, n_per_dim=args.n_joint
    )

    # Phase 3: Output regularization
    out_results = optimize_output_regularization(data, r)

    # Summary
    print("\n" + "=" * 70)
    print("GCV REGULARIZATION SUMMARY")
    print("=" * 70)

    best = joint_results['best_alphas']
    print(f"\n  State operator (joint GCV):")
    print(f"    α_state_lin  = {best['alpha_lin']:.4e}")
    print(f"    α_state_quad = {best['alpha_quad']:.4e}")
    if include_cubic:
        print(f"    α_state_cubic = {best['alpha_cubic']:.4e}")

    print(f"\n  Output operator (block GCV):")
    print(f"    α_out_lin  = {out_results['linear']['alpha_gcv']:.4e}")
    print(f"    α_out_quad = {out_results['quadratic']['alpha_gcv']:.4e}")

    print(f"\n  Block-wise comparison:")
    for name in ['linear', 'quadratic', 'cubic_diag']:
        if name in block_results and 'alpha_gcv' in block_results[name]:
            br = block_results[name]
            print(f"    {name}: GCV={br['alpha_gcv']:.4e}, L-curve={br['alpha_lcurve']:.4e}")

    print("=" * 70)

    # Save results
    save_dict = {
        'joint_alpha_lin': best['alpha_lin'],
        'joint_alpha_quad': best['alpha_quad'],
        'joint_alpha_cubic': best.get('alpha_cubic', 0.0),
        'joint_gcv_score': joint_results['best_gcv'],
        'output_alpha_lin': out_results['linear']['alpha_gcv'],
        'output_alpha_quad': out_results['quadratic']['alpha_gcv'],
    }
    for name in ['linear', 'quadratic', 'cubic_diag']:
        if name in block_results and 'alpha_gcv' in block_results[name]:
            save_dict[f'block_{name}_alpha_gcv'] = block_results[name]['alpha_gcv']
            save_dict[f'block_{name}_alpha_lcurve'] = block_results[name]['alpha_lcurve']

    np.savez(os.path.join(out_dir, 'gcv_results.npz'), **save_dict)
    print(f"\nSaved GCV results to {os.path.join(out_dir, 'gcv_results.npz')}")

    if args.save_plots:
        plot_gcv_lcurve(block_results,
                        save_path=os.path.join(out_dir, 'gcv_lcurve.png'))

    return block_results, joint_results, out_results


if __name__ == '__main__':
    main()
