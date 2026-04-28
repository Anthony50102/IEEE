"""
Regularization Probe: quickly identify viable regularization windows.

Strategy:
  1. Load pre-computed learning matrices from a step 1 output
  2. Phase 1 — Coarse stability scan: test a wide log-spaced grid of
     (alpha_state_lin, alpha_state_quad) with fixed output reg, using
     SHORT rollouts (500 steps) to quickly detect NaN
  3. Phase 2 — Refine around stable region: zoom into the viable window
     with finer spacing and FULL rollouts
  4. Phase 3 — Sweep output reg within the stable state-reg window

Produces a summary of the viable regularization window and recommended
config ranges for the full sweep.

Usage:
    python analysis/probe_regularization.py --run-dir /path/to/step1_output
    python analysis/probe_regularization.py --run-dir /path/to/step1_output --full-steps 20000

Author: Generated for IEEE paper - OpInf closure analysis
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'opinf'))

from core import get_quadratic_terms, get_cubic_diagonal_terms, solve_difference_model


def load_learning_data(run_dir: str) -> dict:
    """Load pre-computed learning matrices from step 1 output."""
    lm_path = os.path.join(run_dir, 'learning_matrices.npz')
    gamma_path = os.path.join(run_dir, 'gamma_reference.npz')

    learning = np.load(lm_path, allow_pickle=True)
    gamma_ref = np.load(gamma_path, allow_pickle=True)

    data = {
        'D_state': learning['D_state'],
        'Y_state': learning['Y_state'],
        'D_state_2': learning['D_state_2'],
        'D_out': learning['D_out'],
        'D_out_2': learning['D_out_2'],
        'X_state': learning['X_state'],
        'mean_Xhat': learning['mean_Xhat'],
        'scaling_Xhat': learning['scaling_Xhat'],
        'Y_Gamma': gamma_ref['Y_Gamma'],
        'mean_Gamma_n': float(gamma_ref['mean_Gamma_n']),
        'std_Gamma_n': float(gamma_ref['std_Gamma_n']),
        'mean_Gamma_c': float(gamma_ref['mean_Gamma_c']),
        'std_Gamma_c': float(gamma_ref['std_Gamma_c']),
        'include_cubic': bool(learning['include_cubic']) if 'include_cubic' in learning else False,
        'include_constant': bool(learning['include_constant']) if 'include_constant' in learning else False,
        'closure_enabled': bool(learning['closure_enabled']) if 'closure_enabled' in learning else False,
    }
    learning.close()
    gamma_ref.close()
    return data


def quick_stability_test(alpha_sl, alpha_sq, data, r, n_steps,
                         alpha_sc=0.0, alpha_ol=1e-2, alpha_oq=1e-2):
    """Test if a (state_lin, state_quad, [state_cubic]) combo produces a stable rollout.

    Returns (stable: bool, blowup_step: int or None)
    """
    include_cubic = data.get('include_cubic', False)
    include_constant = data.get('include_constant', False)

    s = r * (r + 1) // 2
    d_state = r + s
    if include_cubic:
        d_state += r
    if include_constant:
        d_state += 1

    reg_state = np.zeros(d_state)
    col = 0
    reg_state[col:col + r] = alpha_sl
    col += r
    reg_state[col:col + s] = alpha_sq
    col += s
    if include_cubic:
        reg_state[col:col + r] = alpha_sc
        col += r
    if include_constant:
        reg_state[col:col + 1] = alpha_sl
        col += 1

    try:
        DtD = data['D_state_2'] + np.diag(reg_state)
        O = np.linalg.solve(DtD, data['D_state'].T @ data['Y_state']).T
    except np.linalg.LinAlgError:
        return False, 0

    col = 0
    A = O[:, col:col + r]; col += r
    F = O[:, col:col + s]; col += s
    H = O[:, col:col + r] if include_cubic else None
    if include_cubic:
        col += r
    c_state = O[:, col] if include_constant else None

    def f(x):
        result = A @ x + F @ get_quadratic_terms(x)
        if H is not None:
            result += H @ get_cubic_diagonal_terms(x)
        if c_state is not None:
            result += c_state
        return result

    u0 = data['X_state'][0, :]
    is_nan, Xhat = solve_difference_model(u0, n_steps, f)

    if is_nan:
        # Find where it blew up
        nans = np.any(np.isnan(Xhat), axis=0) | np.any(np.isinf(Xhat), axis=0)
        blowup = int(np.argmax(nans)) if np.any(nans) else n_steps
        return False, blowup

    return True, n_steps


def full_evaluate(alpha_sl, alpha_sq, alpha_ol, alpha_oq, data, r,
                  n_steps, training_end, alpha_sc=0.0):
    """Full evaluation including output operator and error metrics."""
    include_cubic = data.get('include_cubic', False)
    include_constant = data.get('include_constant', False)

    s = r * (r + 1) // 2
    d_state = r + s
    if include_cubic:
        d_state += r
    if include_constant:
        d_state += 1
    d_out = r + s + 1

    reg_state = np.zeros(d_state)
    col = 0
    reg_state[col:col + r] = alpha_sl
    col += r
    reg_state[col:col + s] = alpha_sq
    col += s
    if include_cubic:
        reg_state[col:col + r] = alpha_sc
        col += r
    if include_constant:
        reg_state[col:col + 1] = alpha_sl
        col += 1

    DtD = data['D_state_2'] + np.diag(reg_state)
    O = np.linalg.solve(DtD, data['D_state'].T @ data['Y_state']).T

    col = 0
    A = O[:, col:col + r]; col += r
    F = O[:, col:col + s]; col += s
    H = O[:, col:col + r] if include_cubic else None
    if include_cubic:
        col += r
    c_state = O[:, col] if include_constant else None

    def f(x):
        result = A @ x + F @ get_quadratic_terms(x)
        if H is not None:
            result += H @ get_cubic_diagonal_terms(x)
        if c_state is not None:
            result += c_state
        return result

    u0 = data['X_state'][0, :]
    is_nan, Xhat = solve_difference_model(u0, n_steps, f)
    if is_nan:
        return None

    X_OpInf = Xhat.T
    Xhat_scaled = (X_OpInf - data['mean_Xhat']) / data['scaling_Xhat']
    Xhat_2 = get_quadratic_terms(Xhat_scaled)

    reg_out = np.zeros(d_out)
    reg_out[:r] = alpha_ol
    reg_out[r:r+s] = alpha_oq
    reg_out[r+s:] = alpha_ol

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

    return {
        'total_error': mean_err_n + std_err_n + mean_err_c + std_err_c,
        'mean_err_Gamma_n': mean_err_n, 'std_err_Gamma_n': std_err_n,
        'mean_err_Gamma_c': mean_err_c, 'std_err_Gamma_c': std_err_c,
    }


def main():
    parser = argparse.ArgumentParser(description='Probe regularization stability')
    parser.add_argument('--run-dir', required=True, help='Step 1 output directory')
    parser.add_argument('--r', type=int, default=75, help='POD rank')
    parser.add_argument('--probe-steps', type=int, default=500,
                        help='Short rollout steps for stability probing')
    parser.add_argument('--full-steps', type=int, default=20000,
                        help='Full rollout steps for refinement')
    parser.add_argument('--training-end', type=int, default=8000,
                        help='Training end index for metrics')
    args = parser.parse_args()

    print("=" * 70)
    print("REGULARIZATION PROBE")
    print("=" * 70)
    print(f"  Run dir: {args.run_dir}")
    print(f"  r={args.r}, probe_steps={args.probe_steps}, full_steps={args.full_steps}")

    data = load_learning_data(args.run_dir)
    r = args.r
    print(f"  D_state shape: {data['D_state'].shape}")
    print(f"  Closure: cubic={data['include_cubic']}, constant={data['include_constant']}")

    # =========================================================================
    # PHASE 1: Coarse stability scan over (alpha_state_lin, alpha_state_quad)
    # Wide log-spaced grid, short rollout
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Coarse stability scan (short rollout)")
    print("=" * 70)

    # Very wide ranges
    alpha_sl_grid = np.logspace(-1, 8, 20)
    alpha_sq_grid = np.logspace(6, 20, 20)

    # If cubic closure, also sweep cubic reg
    if data['include_cubic']:
        alpha_sc_grid = np.logspace(6, 22, 10)
    else:
        alpha_sc_grid = [0.0]

    n_total = len(alpha_sl_grid) * len(alpha_sq_grid) * len(alpha_sc_grid)
    print(f"  Grid: {len(alpha_sl_grid)} x {len(alpha_sq_grid)} x {len(alpha_sc_grid)} = {n_total} combos")
    print(f"  state_lin: [{alpha_sl_grid[0]:.0e}, {alpha_sl_grid[-1]:.0e}]")
    print(f"  state_quad: [{alpha_sq_grid[0]:.0e}, {alpha_sq_grid[-1]:.0e}]")
    if data['include_cubic']:
        print(f"  state_cubic: [{alpha_sc_grid[0]:.0e}, {alpha_sc_grid[-1]:.0e}]")

    stable_combos = []
    t0 = time.time()

    for i, asl in enumerate(alpha_sl_grid):
        for j, asq in enumerate(alpha_sq_grid):
            for asc in alpha_sc_grid:
                stable, blowup = quick_stability_test(
                    asl, asq, data, r, args.probe_steps, alpha_sc=asc
                )
                if stable:
                    stable_combos.append((asl, asq, asc))

        n_done = (i + 1) * len(alpha_sq_grid) * len(alpha_sc_grid)
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        print(f"  [{n_done}/{n_total}] {len(stable_combos)} stable "
              f"({elapsed:.1f}s, {rate:.1f} combos/s)")

    print(f"\nPhase 1 complete: {len(stable_combos)}/{n_total} stable")

    if len(stable_combos) == 0:
        print("\n*** NO STABLE COMBOS FOUND ***")
        print("Trying even stronger regularization...")

        # Emergency scan with extreme reg
        alpha_sl_grid = np.logspace(4, 12, 15)
        alpha_sq_grid = np.logspace(14, 25, 15)
        if data['include_cubic']:
            alpha_sc_grid = np.logspace(14, 28, 8)
        else:
            alpha_sc_grid = [0.0]

        for asl in alpha_sl_grid:
            for asq in alpha_sq_grid:
                for asc in alpha_sc_grid:
                    stable, blowup = quick_stability_test(
                        asl, asq, data, r, args.probe_steps, alpha_sc=asc
                    )
                    if stable:
                        stable_combos.append((asl, asq, asc))

        print(f"Emergency scan: {len(stable_combos)} stable")
        if len(stable_combos) == 0:
            print("Still no stable combos. The closure formulation may not work for this problem.")
            return

    # Identify the viable window
    sl_vals = sorted(set(c[0] for c in stable_combos))
    sq_vals = sorted(set(c[1] for c in stable_combos))
    print(f"\n  Stable state_lin range:  [{min(sl_vals):.2e}, {max(sl_vals):.2e}]")
    print(f"  Stable state_quad range: [{min(sq_vals):.2e}, {max(sq_vals):.2e}]")
    if data['include_cubic']:
        sc_vals = sorted(set(c[2] for c in stable_combos))
        print(f"  Stable state_cubic range: [{min(sc_vals):.2e}, {max(sc_vals):.2e}]")

    # =========================================================================
    # PHASE 2: Full rollout validation on stable combos
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Full rollout validation")
    print("=" * 70)

    # Take a subset of stable combos for full validation
    n_validate = min(50, len(stable_combos))
    if len(stable_combos) > n_validate:
        indices = np.linspace(0, len(stable_combos) - 1, n_validate, dtype=int)
        validate_combos = [stable_combos[i] for i in indices]
    else:
        validate_combos = stable_combos

    print(f"  Validating {len(validate_combos)} combos with {args.full_steps} steps...")

    full_stable = []
    t0 = time.time()
    for i, (asl, asq, asc) in enumerate(validate_combos):
        stable, blowup = quick_stability_test(
            asl, asq, data, r, args.full_steps, alpha_sc=asc
        )
        if stable:
            full_stable.append((asl, asq, asc))
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(validate_combos)}] {len(full_stable)} stable "
                  f"({time.time()-t0:.1f}s)")

    print(f"\nPhase 2: {len(full_stable)}/{len(validate_combos)} survive full rollout")

    if len(full_stable) == 0:
        print("No combos survive full rollout. Using short-rollout stable window.")
        full_stable = stable_combos[:20]

    # =========================================================================
    # PHASE 3: Output reg sweep on best state reg combos
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Output regularization sweep + error evaluation")
    print("=" * 70)

    alpha_ol_grid = np.logspace(-4, 2, 8)
    alpha_oq_grid = np.logspace(-3, 3, 8)

    # Pick top 5 state-reg combos (spread across the stable range)
    n_state = min(5, len(full_stable))
    indices = np.linspace(0, len(full_stable) - 1, n_state, dtype=int)
    state_combos = [full_stable[i] for i in indices]

    best_result = None
    best_params = None
    all_results = []

    n_total = n_state * len(alpha_ol_grid) * len(alpha_oq_grid)
    print(f"  {n_state} state combos x {len(alpha_ol_grid)} x {len(alpha_oq_grid)} = {n_total} evaluations")

    t0 = time.time()
    count = 0
    for asl, asq, asc in state_combos:
        for aol in alpha_ol_grid:
            for aoq in alpha_oq_grid:
                result = full_evaluate(
                    asl, asq, aol, aoq, data, r,
                    args.full_steps, args.training_end, alpha_sc=asc
                )
                count += 1
                if result is not None:
                    result['alpha_sl'] = asl
                    result['alpha_sq'] = asq
                    result['alpha_sc'] = asc
                    result['alpha_ol'] = aol
                    result['alpha_oq'] = aoq
                    all_results.append(result)

                    if best_result is None or result['total_error'] < best_result['total_error']:
                        best_result = result
                        best_params = (asl, asq, asc, aol, aoq)

        print(f"  State combo ({asl:.1e}, {asq:.1e}, {asc:.1e}): "
              f"{len(all_results)} valid so far, "
              f"best_err={best_result['total_error']:.4f}" if best_result else "no valid")

    elapsed = time.time() - t0
    print(f"\nPhase 3 complete: {len(all_results)}/{count} valid ({elapsed:.1f}s)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if best_result is None:
        print("No valid results found!")
        return

    asl, asq, asc, aol, aoq = best_params
    print(f"  Best total error: {best_result['total_error']:.4f}")
    print(f"    mean_err_Gamma_n: {best_result['mean_err_Gamma_n']:.4f}")
    print(f"    std_err_Gamma_n:  {best_result['std_err_Gamma_n']:.4f}")
    print(f"    mean_err_Gamma_c: {best_result['mean_err_Gamma_c']:.4f}")
    print(f"    std_err_Gamma_c:  {best_result['std_err_Gamma_c']:.4f}")
    print(f"  Best params:")
    print(f"    alpha_state_lin:   {asl:.2e}")
    print(f"    alpha_state_quad:  {asq:.2e}")
    if data['include_cubic']:
        print(f"    alpha_state_cubic: {asc:.2e}")
    print(f"    alpha_out_lin:     {aol:.2e}")
    print(f"    alpha_out_quad:    {aoq:.2e}")

    # Recommended config ranges (centered on best, ±1 decade)
    print(f"\n  RECOMMENDED CONFIG RANGES (for full sweep):")
    sl_stable = sorted(set(c[0] for c in full_stable))
    sq_stable = sorted(set(c[1] for c in full_stable))
    print(f"    state_lin:  {{min: {min(sl_stable):.1e}, max: {max(sl_stable):.1e}, num: 10, scale: log}}")
    print(f"    state_quad: {{min: {min(sq_stable):.1e}, max: {max(sq_stable):.1e}, num: 10, scale: log}}")
    if data['include_cubic']:
        sc_stable = sorted(set(c[2] for c in full_stable))
        print(f"    state_cubic: {{min: {min(sc_stable):.1e}, max: {max(sc_stable):.1e}, num: 5, scale: log}}")
    print(f"    output_lin:  {{min: {aol/10:.1e}, max: {aol*10:.1e}, num: 10, scale: log}}")
    print(f"    output_quad: {{min: {aoq/10:.1e}, max: {aoq*10:.1e}, num: 10, scale: log}}")

    # Top 10 results
    all_results.sort(key=lambda x: x['total_error'])
    print(f"\n  TOP 10 RESULTS:")
    print(f"  {'total':>8s} {'mn_Gn':>8s} {'sd_Gn':>8s} {'mn_Gc':>8s} {'sd_Gc':>8s} "
          f"{'a_sl':>10s} {'a_sq':>10s} {'a_sc':>10s} {'a_ol':>10s} {'a_oq':>10s}")
    for res in all_results[:10]:
        print(f"  {res['total_error']:8.4f} {res['mean_err_Gamma_n']:8.4f} "
              f"{res['std_err_Gamma_n']:8.4f} {res['mean_err_Gamma_c']:8.4f} "
              f"{res['std_err_Gamma_c']:8.4f} {res['alpha_sl']:10.2e} "
              f"{res['alpha_sq']:10.2e} {res['alpha_sc']:10.2e} "
              f"{res['alpha_ol']:10.2e} {res['alpha_oq']:10.2e}")

    # Save results
    out_path = os.path.join(args.run_dir, 'probe_results.npz')
    np.savez(out_path,
             all_results=all_results,
             stable_combos=stable_combos,
             full_stable=full_stable,
             best_params=best_params,
             best_result=best_result)
    print(f"\n  Results saved to: {out_path}")


if __name__ == '__main__':
    main()
