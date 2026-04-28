"""
Sequential Regularization Optimization: GCV-guided rollout refinement.

Combines GCV-optimal regularization with actual rollout validation:
  1. Load GCV results (from gcv_regularization.py) as center points
  2. Build a small grid around GCV-optimal (±1-2 decades)
  3. Solve state operators, run actual rollouts to check stability
  4. For stable models, solve output operators and compute Γ metrics
  5. Report best models and recommended final regularization

This bridges the gap between GCV (optimizes one-step fit) and what we
actually need (stable 20k-step rollout with good Γ statistics).

Usage:
    python analysis/sequential_optimize.py --run-dir /path/to/step1_output
    python analysis/sequential_optimize.py --run-dir /path/to/step1_output \
        --gcv-results /path/to/gcv_results.npz --full-steps 20000

Author: Anthony Poole
"""

import argparse
import gc
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'opinf'))

from core import get_quadratic_terms, get_cubic_diagonal_terms, solve_difference_model
from svd_diagnosis import load_learning_data


# =============================================================================
# STATE OPERATOR EVALUATION
# =============================================================================

def solve_state_operator(data: dict, r: int, alpha_lin: float, alpha_quad: float,
                         alpha_cubic: float = 0.0) -> dict:
    """Solve for state operators with given regularization."""
    include_cubic = data.get('include_cubic', False)
    include_constant = data.get('include_constant', False)

    s = r * (r + 1) // 2
    d_state = r + s
    if include_cubic:
        d_state += r
    if include_constant:
        d_state += 1

    reg = np.zeros(d_state)
    col = 0
    reg[col:col + r] = alpha_lin
    col += r
    reg[col:col + s] = alpha_quad
    col += s
    if include_cubic:
        reg[col:col + r] = alpha_cubic
        col += r
    if include_constant:
        reg[col:col + 1] = alpha_lin

    DtD_reg = data['D_state_2'] + np.diag(reg)
    O = np.linalg.solve(DtD_reg, data['D_state'].T @ data['Y_state']).T

    col = 0
    A = O[:, col:col + r]
    col += r
    F = O[:, col:col + s]
    col += s
    H = O[:, col:col + r] if include_cubic else None
    if include_cubic:
        col += r
    c_state = O[:, col] if include_constant else None

    return {'A': A, 'F': F, 'H': H, 'c_state': c_state}


def build_transition_fn(ops: dict):
    """Build state transition function from operators."""
    A, F = ops['A'], ops['F']
    H, c_state = ops.get('H'), ops.get('c_state')

    def f(x):
        result = A @ x + F @ get_quadratic_terms(x)
        if H is not None:
            result += H @ get_cubic_diagonal_terms(x)
        if c_state is not None:
            result += c_state
        return result

    return f


def test_stability(data: dict, ops: dict, n_steps: int) -> tuple:
    """Run rollout and check stability. Returns (stable, trajectory)."""
    f = build_transition_fn(ops)
    u0 = data['X_state'][0, :]
    is_nan, X_pred = solve_difference_model(u0, n_steps, f)
    return not is_nan, X_pred


def compute_gamma_error(data: dict, X_pred: np.ndarray, r: int,
                        alpha_out_lin: float, alpha_out_quad: float,
                        training_end: int) -> dict:
    """Compute Γ errors for a given state trajectory and output regularization."""
    s = r * (r + 1) // 2
    d_out = r + s + 1

    X_OpInf = X_pred.T  # (n_steps, r)

    # Scale for output operator
    Xhat_scaled = (X_OpInf - data['mean_Xhat']) / data['scaling_Xhat']
    Xhat_2 = get_quadratic_terms(Xhat_scaled)

    # Output regularization
    reg_out = np.zeros(d_out)
    reg_out[:r] = alpha_out_lin
    reg_out[r:r+s] = alpha_out_quad
    reg_out[r+s:] = alpha_out_lin

    DtD_out = data['D_out_2'] + np.diag(reg_out)
    O_out = np.linalg.solve(DtD_out, data['D_out'].T @ data['Y_Gamma'].T).T
    C, G, c = O_out[:, :r], O_out[:, r:r+s], O_out[:, r+s]

    Y_OpInf = C @ Xhat_scaled.T + G @ Xhat_2.T + c[:, np.newaxis]

    ts_Gamma_n = Y_OpInf[0, :training_end]
    ts_Gamma_c = Y_OpInf[1, :training_end]

    mean_err_n = abs(data['mean_Gamma_n'] - np.mean(ts_Gamma_n)) / abs(data['mean_Gamma_n'])
    std_err_n = abs(data['std_Gamma_n'] - np.std(ts_Gamma_n, ddof=1)) / data['std_Gamma_n']
    mean_err_c = abs(data['mean_Gamma_c'] - np.mean(ts_Gamma_c)) / abs(data['mean_Gamma_c'])
    std_err_c = abs(data['std_Gamma_c'] - np.std(ts_Gamma_c, ddof=1)) / data['std_Gamma_c']

    total = mean_err_n + std_err_n + mean_err_c + std_err_c

    return {
        'total_error': total,
        'mean_err_Gamma_n': mean_err_n,
        'std_err_Gamma_n': std_err_n,
        'mean_err_Gamma_c': mean_err_c,
        'std_err_Gamma_c': std_err_c,
    }


# =============================================================================
# SEQUENTIAL OPTIMIZATION
# =============================================================================

def phase1_state_stability(data: dict, r: int,
                           center_lin: float, center_quad: float,
                           center_cubic: float = 0.0,
                           n_per_dim: int = 7,
                           decades: float = 2.0,
                           short_steps: int = 500) -> list:
    """
    Phase 1: Coarse stability scan around GCV center.
    
    Uses short rollouts to quickly identify stable region.
    """
    include_cubic = data.get('include_cubic', False)

    def make_grid(center, n, dec):
        log_c = np.log10(max(center, 1e-30))
        return np.logspace(log_c - dec, log_c + dec, n)

    grid_lin = make_grid(center_lin, n_per_dim, decades)
    grid_quad = make_grid(center_quad, n_per_dim, decades)
    grid_cubic = make_grid(center_cubic, n_per_dim, decades) if include_cubic else [0.0]

    total = len(grid_lin) * len(grid_quad) * len(grid_cubic)
    print(f"\n  Phase 1: Stability scan ({total} combos, {short_steps} steps)")

    stable_configs = []
    n_tested = 0

    for a_lin in grid_lin:
        for a_quad in grid_quad:
            for a_cubic in grid_cubic:
                ops = solve_state_operator(data, r, a_lin, a_quad, a_cubic)
                stable, _ = test_stability(data, ops, short_steps)
                n_tested += 1

                if stable:
                    stable_configs.append({
                        'alpha_lin': a_lin,
                        'alpha_quad': a_quad,
                        'alpha_cubic': a_cubic,
                    })

                if n_tested % 50 == 0:
                    print(f"    {n_tested}/{total}: {len(stable_configs)} stable")
                    gc.collect()

    print(f"  Phase 1 complete: {len(stable_configs)}/{total} stable "
          f"({100*len(stable_configs)/max(total,1):.0f}%)")

    return stable_configs


def phase2_full_rollout(data: dict, r: int,
                        stable_configs: list,
                        full_steps: int,
                        max_configs: int = 50) -> list:
    """
    Phase 2: Full-length rollout validation on stable configs.
    """
    # Subsample if too many
    if len(stable_configs) > max_configs:
        indices = np.linspace(0, len(stable_configs) - 1, max_configs, dtype=int)
        configs = [stable_configs[i] for i in indices]
    else:
        configs = stable_configs

    print(f"\n  Phase 2: Full rollout ({len(configs)} configs, {full_steps} steps)")

    survivors = []
    for i, cfg in enumerate(configs):
        ops = solve_state_operator(data, r, cfg['alpha_lin'],
                                   cfg['alpha_quad'], cfg['alpha_cubic'])
        stable, X_pred = test_stability(data, ops, full_steps)

        if stable:
            cfg['X_pred'] = X_pred
            survivors.append(cfg)

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(configs)}: {len(survivors)} survive")
            gc.collect()

    print(f"  Phase 2 complete: {len(survivors)}/{len(configs)} survive full rollout")
    return survivors


def phase3_output_sweep(data: dict, r: int,
                        survivors: list,
                        training_end: int,
                        out_center_lin: float = 1.0,
                        out_center_quad: float = 1.0,
                        n_out: int = 7,
                        decades_out: float = 2.0) -> list:
    """
    Phase 3: Sweep output regularization for each surviving state config.
    """
    def make_grid(center, n, dec):
        log_c = np.log10(max(center, 1e-30))
        return np.logspace(log_c - dec, log_c + dec, n)

    grid_out_lin = make_grid(out_center_lin, n_out, decades_out)
    grid_out_quad = make_grid(out_center_quad, n_out, decades_out)

    n_out_combos = len(grid_out_lin) * len(grid_out_quad)
    total = len(survivors) * n_out_combos
    print(f"\n  Phase 3: Output sweep ({len(survivors)} state × "
          f"{n_out_combos} output = {total} evaluations)")

    all_results = []
    for i, cfg in enumerate(survivors):
        X_pred = cfg['X_pred']
        best_for_this = None

        for a_out_lin in grid_out_lin:
            for a_out_quad in grid_out_quad:
                metrics = compute_gamma_error(
                    data, X_pred, r, a_out_lin, a_out_quad, training_end
                )

                result = {
                    'alpha_state_lin': cfg['alpha_lin'],
                    'alpha_state_quad': cfg['alpha_quad'],
                    'alpha_state_cubic': cfg['alpha_cubic'],
                    'alpha_out_lin': a_out_lin,
                    'alpha_out_quad': a_out_quad,
                    **metrics,
                }
                all_results.append(result)

                if best_for_this is None or metrics['total_error'] < best_for_this['total_error']:
                    best_for_this = result

        if (i + 1) % 5 == 0:
            print(f"    State config {i+1}/{len(survivors)}: "
                  f"best total_error={best_for_this['total_error']:.4f}")
            gc.collect()

    # Sort by total error
    all_results.sort(key=lambda x: x['total_error'])

    print(f"\n  Phase 3 complete: {len(all_results)} total evaluations")
    if all_results:
        best = all_results[0]
        print(f"  Best model:")
        print(f"    total_error = {best['total_error']:.4f}")
        print(f"    mean_err_Γn = {best['mean_err_Gamma_n']:.4f}")
        print(f"    std_err_Γn  = {best['std_err_Gamma_n']:.4f}")
        print(f"    mean_err_Γc = {best['mean_err_Gamma_c']:.4f}")
        print(f"    std_err_Γc  = {best['std_err_Gamma_c']:.4f}")
        print(f"    α_state_lin  = {best['alpha_state_lin']:.4e}")
        print(f"    α_state_quad = {best['alpha_state_quad']:.4e}")
        print(f"    α_state_cubic = {best['alpha_state_cubic']:.4e}")
        print(f"    α_out_lin    = {best['alpha_out_lin']:.4e}")
        print(f"    α_out_quad   = {best['alpha_out_quad']:.4e}")

    return all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sequential regularization optimization")
    parser.add_argument('--run-dir', required=True, help="Path to step 1 output directory")
    parser.add_argument('--gcv-results', default=None,
                        help="Path to gcv_results.npz (default: run-dir/gcv_results.npz)")
    parser.add_argument('--short-steps', type=int, default=500,
                        help="Steps for Phase 1 stability scan")
    parser.add_argument('--full-steps', type=int, default=0,
                        help="Steps for Phase 2 rollout (0=auto from data)")
    parser.add_argument('--training-end', type=int, default=0,
                        help="Training window end for Γ evaluation (0=auto)")
    parser.add_argument('--n-state', type=int, default=7,
                        help="Grid points per state reg dimension")
    parser.add_argument('--n-output', type=int, default=7,
                        help="Grid points per output reg dimension")
    parser.add_argument('--decades', type=float, default=2.0,
                        help="Decades around GCV center for state sweep")
    parser.add_argument('--output-dir', default=None,
                        help="Output directory (default: run-dir)")
    parser.add_argument('--only-center', default=None,
                        help="Run only this center (gcv, lcurve, svd)")
    args = parser.parse_args()

    out_dir = args.output_dir or args.run_dir

    # Load data
    print(f"Loading data from: {args.run_dir}")
    data = load_learning_data(args.run_dir)

    r = data['X_state'].shape[1]
    include_cubic = data['include_cubic']
    include_constant = data['include_constant']

    # Determine full steps and training_end
    n_samples = data['D_state'].shape[0]
    full_steps = args.full_steps if args.full_steps > 0 else n_samples + 1
    training_end = args.training_end if args.training_end > 0 else n_samples + 1

    print(f"r={r}, cubic={include_cubic}, constant={include_constant}")
    print(f"full_steps={full_steps}, training_end={training_end}")

    # Load GCV results and SVD diagnosis for center points
    gcv_path = args.gcv_results or os.path.join(args.run_dir, 'gcv_results.npz')
    svd_path = os.path.join(args.run_dir, 'svd_diagnosis.npz')

    # Build a hierarchy of center candidates: GCV → L-curve → SVD σ_max²
    centers = []

    if os.path.exists(gcv_path):
        gcv = np.load(gcv_path)
        # GCV centers (best one-step fit, may be too weak for stability)
        centers.append({
            'name': 'GCV',
            'lin': float(gcv['joint_alpha_lin']),
            'quad': float(gcv['joint_alpha_quad']),
            'cubic': float(gcv.get('joint_alpha_cubic', 0.0)),
        })
        # L-curve centers (often stronger, better for stability)
        if 'block_linear_alpha_lcurve' in gcv:
            centers.append({
                'name': 'L-curve',
                'lin': float(gcv['block_linear_alpha_lcurve']),
                'quad': float(gcv['block_quadratic_alpha_lcurve']),
                'cubic': float(gcv.get('block_cubic_diag_alpha_lcurve', 0.0)),
            })
        out_center_lin = float(gcv['output_alpha_lin'])
        out_center_quad = float(gcv['output_alpha_quad'])
        # Use wide output search: center at 1.0, ±N decades to cover
        # both very weak (1e-4) and strong (1e5) output regularization
        out_center_lin = 1.0
        out_center_quad = 1.0
    else:
        out_center_lin = 1e-1
        out_center_quad = 1e0

    if os.path.exists(svd_path):
        svd = np.load(svd_path)
        # SVD σ_max² as strong-regularization fallback
        centers.append({
            'name': 'SVD σ_max²',
            'lin': float(svd.get('rec_linear_sigma_max_sq', 1e4)),
            'quad': float(svd.get('rec_quadratic_sigma_max_sq', 1e6)),
            'cubic': float(svd.get('rec_cubic_diag_sigma_max_sq', 1e6)),
        })

    if not centers:
        centers.append({
            'name': 'default',
            'lin': 1e2, 'quad': 1e8, 'cubic': 1e8,
        })

    print(f"\nCenter candidates:")
    for c in centers:
        print(f"  [{c['name']}] lin={c['lin']:.4e}, quad={c['quad']:.4e}, cubic={c['cubic']:.4e}")

    # Filter to a single center if requested
    if args.only_center:
        name_map = {'gcv': 'GCV', 'lcurve': 'L-curve', 'svd': 'SVD σ_max²'}
        target = name_map.get(args.only_center.lower(), args.only_center)
        centers = [c for c in centers if c['name'] == target]
        if not centers:
            print(f"ERROR: center '{args.only_center}' not found")
            sys.exit(1)
        print(f"\n  → Running only center: {target}")

    # Try ALL center candidates and compare results
    t0 = time.time()
    all_center_results = {}

    for center in centers:
        print(f"\n{'='*60}")
        print(f"CENTER: {center['name']}")
        print(f"  lin={center['lin']:.4e}, quad={center['quad']:.4e}, "
              f"cubic={center['cubic']:.4e}")
        print(f"{'='*60}")

        stable = phase1_state_stability(
            data, r, center['lin'], center['quad'], center['cubic'],
            n_per_dim=args.n_state, decades=args.decades,
            short_steps=args.short_steps
        )

        if not stable:
            print(f"  {center['name']}: 0% stable, skipping")
            continue

        survivors = phase2_full_rollout(data, r, stable, full_steps, max_configs=50)
        if not survivors:
            print(f"  {center['name']}: no survivors at full rollout, skipping")
            continue

        results = phase3_output_sweep(
            data, r, survivors, training_end,
            out_center_lin=out_center_lin,
            out_center_quad=out_center_quad,
            n_out=args.n_output,
            decades_out=4.0
        )

        if results:
            all_center_results[center['name']] = results

    if not all_center_results:
        print("\nERROR: No center produced valid results!")
        return

    # Find overall best across all centers
    print(f"\n{'='*70}")
    print("CROSS-CENTER COMPARISON")
    print(f"{'='*70}")
    overall_best = None
    overall_best_center = None
    for cname, results in all_center_results.items():
        best = results[0]
        print(f"\n  [{cname}] best total_error={best['total_error']:.4f}")
        print(f"    Γn: mean={best['mean_err_Gamma_n']:.4f}, std={best['std_err_Gamma_n']:.4f}")
        print(f"    Γc: mean={best['mean_err_Gamma_c']:.4f}, std={best['std_err_Gamma_c']:.4f}")
        print(f"    α: lin={best['alpha_state_lin']:.2e}, quad={best['alpha_state_quad']:.2e}, "
              f"cubic={best['alpha_state_cubic']:.2e}")
        if overall_best is None or best['total_error'] < overall_best['total_error']:
            overall_best = best
            overall_best_center = cname

    print(f"\n  WINNER: {overall_best_center} (total_error={overall_best['total_error']:.4f})")

    # Use the best center's results going forward
    results = all_center_results[overall_best_center]
    stable = []  # not needed below

    total_time = time.time() - t0
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save results
    if results:
        # Save top 100 results
        top_n = min(100, len(results))
        save_dict = {
            'n_results': len(results),
            'total_time': total_time,
        }
        for key in ['alpha_state_lin', 'alpha_state_quad', 'alpha_state_cubic',
                     'alpha_out_lin', 'alpha_out_quad', 'total_error',
                     'mean_err_Gamma_n', 'std_err_Gamma_n',
                     'mean_err_Gamma_c', 'std_err_Gamma_c']:
            save_dict[key] = np.array([r[key] for r in results[:top_n]])

        np.savez(os.path.join(out_dir, 'sequential_results.npz'), **save_dict)
        print(f"\nSaved results to {os.path.join(out_dir, 'sequential_results.npz')}")

        # Print top 10
        print("\n" + "=" * 70)
        print("TOP 10 MODELS")
        print("=" * 70)
        for i, res in enumerate(results[:10]):
            print(f"\n  #{i+1}: total_error={res['total_error']:.4f}")
            print(f"    Γn: mean={res['mean_err_Gamma_n']:.4f}, std={res['std_err_Gamma_n']:.4f}")
            print(f"    Γc: mean={res['mean_err_Gamma_c']:.4f}, std={res['std_err_Gamma_c']:.4f}")
            print(f"    α: lin={res['alpha_state_lin']:.2e}, quad={res['alpha_state_quad']:.2e}, "
                  f"cubic={res['alpha_state_cubic']:.2e}")
            print(f"    out: lin={res['alpha_out_lin']:.2e}, quad={res['alpha_out_quad']:.2e}")

    return results


if __name__ == '__main__':
    main()
