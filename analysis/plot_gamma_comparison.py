"""
Plot Γ time series comparison: reference vs OpInf (old probe vs new pipeline).

Generates:
  1. Γn time series (reference vs old vs new)
  2. Γc time series (reference vs old vs new)
  3. Running statistics (mean, std) convergence

Usage:
    python analysis/plot_gamma_comparison.py --run-dir /path/to/step1_output
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'opinf'))
from core import get_quadratic_terms, get_cubic_diagonal_terms, solve_difference_model


def compute_gamma_trajectory(data, gamma_ref, alpha_state_lin, alpha_state_quad,
                              alpha_state_cubic, alpha_out_lin, alpha_out_quad,
                              n_steps=8000):
    """Compute full Γ trajectory for given regularization."""
    r = data['X_state'].shape[1]
    include_cubic = data.get('include_cubic', False)
    include_constant = data.get('include_constant', False)
    s = r * (r + 1) // 2

    # State operator
    d_state = r + s
    if include_cubic:
        d_state += r
    if include_constant:
        d_state += 1

    reg = np.zeros(d_state)
    col = 0
    reg[col:col + r] = alpha_state_lin; col += r
    reg[col:col + s] = alpha_state_quad; col += s
    if include_cubic:
        reg[col:col + r] = alpha_state_cubic; col += r
    if include_constant:
        reg[col:col + 1] = alpha_state_lin

    DtD_reg = data['D_state_2'] + np.diag(reg)
    O = np.linalg.solve(DtD_reg, data['D_state'].T @ data['Y_state']).T

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
    is_nan, Xhat_pred = solve_difference_model(u0, n_steps, f)
    if is_nan:
        return None

    X_OpInf = Xhat_pred.T
    Xhat_scaled = (X_OpInf - data['mean_Xhat']) / data['scaling_Xhat']
    Xhat_2 = get_quadratic_terms(Xhat_scaled)

    d_out = r + s + 1
    reg_out = np.zeros(d_out)
    reg_out[:r] = alpha_out_lin
    reg_out[r:r+s] = alpha_out_quad
    reg_out[r+s:] = alpha_out_lin

    DtD_out = data['D_out_2'] + np.diag(reg_out)
    O_out = np.linalg.solve(DtD_out, data['D_out'].T @ gamma_ref['Y_Gamma'].T).T
    C, G, c = O_out[:, :r], O_out[:, r:r+s], O_out[:, r+s]

    Y_OpInf = C @ Xhat_scaled.T + G @ Xhat_2.T + c[:, np.newaxis]

    return Y_OpInf  # (2, n_steps)


def running_stat(x, window=200):
    """Compute running mean and std with given window."""
    n = len(x)
    means = np.array([np.mean(x[max(0, i-window):i+1]) for i in range(n)])
    stds = np.array([np.std(x[max(0, i-window):i+1], ddof=1) if i > 0 else 0.0 for i in range(n)])
    return means, stds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--n-steps', type=int, default=8000)
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.run_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    lm = np.load(os.path.join(args.run_dir, 'learning_matrices.npz'))
    data = dict(lm)
    data['include_cubic'] = True
    data['include_constant'] = True

    gamma_ref = np.load(os.path.join(args.run_dir, 'gamma_reference.npz'))
    ref_Gn = gamma_ref['Y_Gamma'][0, :]
    ref_Gc = gamma_ref['Y_Gamma'][1, :]

    # Probe best (previous best regularization)
    print("Computing probe trajectory...")
    Y_probe = compute_gamma_trajectory(
        data, gamma_ref,
        alpha_state_lin=1.62e4, alpha_state_quad=6.16e17,
        alpha_state_cubic=1.67e20,
        alpha_out_lin=1e-4, alpha_out_quad=1e3,
        n_steps=args.n_steps
    )

    # New pipeline best
    print("Computing pipeline trajectory...")
    Y_pipeline = compute_gamma_trajectory(
        data, gamma_ref,
        alpha_state_lin=7.89e3, alpha_state_quad=2.64e17,
        alpha_state_cubic=1.27e24,
        alpha_out_lin=1e-4, alpha_out_quad=1e-2,
        n_steps=args.n_steps
    )

    if Y_probe is None:
        print("WARNING: Probe trajectory diverged!")
    if Y_pipeline is None:
        print("ERROR: Pipeline trajectory diverged!")
        return

    # Import matplotlib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = np.arange(len(ref_Gn))
    n = min(args.n_steps, len(ref_Gn))
    t = t[:n]

    # =========================================================================
    # Figure 1: Γn time series
    # =========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    ax.plot(t, ref_Gn[:n], 'k-', alpha=0.6, linewidth=0.5, label='Reference (DNS)')
    if Y_probe is not None:
        ax.plot(t, Y_probe[0, :n], 'r-', alpha=0.6, linewidth=0.5, label='Probe (prev best)')
    ax.plot(t, Y_pipeline[0, :n], 'b-', alpha=0.6, linewidth=0.5, label='Pipeline (new best)')
    ax.set_ylabel('Γn')
    ax.set_title('Particle Flux Γn — Time Series')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, ref_Gc[:n], 'k-', alpha=0.6, linewidth=0.5, label='Reference (DNS)')
    if Y_probe is not None:
        ax.plot(t, Y_probe[1, :n], 'r-', alpha=0.6, linewidth=0.5, label='Probe (prev best)')
    ax.plot(t, Y_pipeline[1, :n], 'b-', alpha=0.6, linewidth=0.5, label='Pipeline (new best)')
    ax.set_ylabel('Γc')
    ax.set_xlabel('Time step')
    ax.set_title('Enstrophy Transfer Γc — Time Series')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(out_dir, 'gamma_timeseries_comparison.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"Saved: {path1}")
    plt.close()

    # =========================================================================
    # Figure 2: Running statistics convergence
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    window = 500
    ref_mean_n, ref_std_n = running_stat(ref_Gn[:n], window)
    ref_mean_c, ref_std_c = running_stat(ref_Gc[:n], window)

    pipe_mean_n, pipe_std_n = running_stat(Y_pipeline[0, :n], window)
    pipe_mean_c, pipe_std_c = running_stat(Y_pipeline[1, :n], window)

    if Y_probe is not None:
        probe_mean_n, probe_std_n = running_stat(Y_probe[0, :n], window)
        probe_mean_c, probe_std_c = running_stat(Y_probe[1, :n], window)

    # Γn mean
    ax = axes[0, 0]
    ax.plot(t, ref_mean_n, 'k-', linewidth=1.5, label='Reference')
    if Y_probe is not None:
        ax.plot(t, probe_mean_n, 'r--', linewidth=1.2, label='Probe')
    ax.plot(t, pipe_mean_n, 'b--', linewidth=1.2, label='Pipeline')
    ax.axhline(float(gamma_ref['mean_Gamma_n']), color='gray', linestyle=':', alpha=0.5, label='True mean')
    ax.set_title(f'Γn Running Mean (window={window})')
    ax.set_ylabel('Mean Γn')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Γn std
    ax = axes[0, 1]
    ax.plot(t, ref_std_n, 'k-', linewidth=1.5, label='Reference')
    if Y_probe is not None:
        ax.plot(t, probe_std_n, 'r--', linewidth=1.2, label='Probe')
    ax.plot(t, pipe_std_n, 'b--', linewidth=1.2, label='Pipeline')
    ax.axhline(float(gamma_ref['std_Gamma_n']), color='gray', linestyle=':', alpha=0.5, label='True std')
    ax.set_title(f'Γn Running Std (window={window})')
    ax.set_ylabel('Std Γn')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Γc mean
    ax = axes[1, 0]
    ax.plot(t, ref_mean_c, 'k-', linewidth=1.5, label='Reference')
    if Y_probe is not None:
        ax.plot(t, probe_mean_c, 'r--', linewidth=1.2, label='Probe')
    ax.plot(t, pipe_mean_c, 'b--', linewidth=1.2, label='Pipeline')
    ax.axhline(float(gamma_ref['mean_Gamma_c']), color='gray', linestyle=':', alpha=0.5, label='True mean')
    ax.set_title(f'Γc Running Mean (window={window})')
    ax.set_ylabel('Mean Γc')
    ax.set_xlabel('Time step')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Γc std
    ax = axes[1, 1]
    ax.plot(t, ref_std_c, 'k-', linewidth=1.5, label='Reference')
    if Y_probe is not None:
        ax.plot(t, probe_std_c, 'r--', linewidth=1.2, label='Probe')
    ax.plot(t, pipe_std_c, 'b--', linewidth=1.2, label='Pipeline')
    ax.axhline(float(gamma_ref['std_Gamma_c']), color='gray', linestyle=':', alpha=0.5, label='True std')
    ax.set_title(f'Γc Running Std (window={window})')
    ax.set_ylabel('Std Γc')
    ax.set_xlabel('Time step')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Running Statistics Convergence — HW2D Closure OpInf', fontsize=14, y=1.01)
    plt.tight_layout()
    path2 = os.path.join(out_dir, 'gamma_running_stats_comparison.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {path2}")
    plt.close()

    # =========================================================================
    # Figure 3: Error summary bar chart
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5))

    metrics = ['mean_err_Γn', 'std_err_Γn', 'mean_err_Γc', 'std_err_Γc', 'total_error']

    # Compute errors
    def compute_errors(Y, ref_data):
        gn = Y[0, :n]
        gc = Y[1, :n]
        me_n = abs(float(ref_data['mean_Gamma_n']) - np.mean(gn)) / abs(float(ref_data['mean_Gamma_n']))
        se_n = abs(float(ref_data['std_Gamma_n']) - np.std(gn, ddof=1)) / float(ref_data['std_Gamma_n'])
        me_c = abs(float(ref_data['mean_Gamma_c']) - np.mean(gc)) / abs(float(ref_data['mean_Gamma_c']))
        se_c = abs(float(ref_data['std_Gamma_c']) - np.std(gc, ddof=1)) / float(ref_data['std_Gamma_c'])
        return [me_n, se_n, me_c, se_c, me_n + se_n + me_c + se_c]

    pipe_errs = compute_errors(Y_pipeline, gamma_ref)
    labels = ['mean Γn', 'std Γn', 'mean Γc', 'std Γc', 'Total']

    x = np.arange(len(metrics))
    width = 0.3

    if Y_probe is not None:
        probe_errs = compute_errors(Y_probe, gamma_ref)
        ax.bar(x - width/2, probe_errs, width, label='Probe (prev best)', color='#d62728', alpha=0.8)
        ax.bar(x + width/2, pipe_errs, width, label='Pipeline (new best)', color='#1f77b4', alpha=0.8)
    else:
        ax.bar(x, pipe_errs, width, label='Pipeline (new best)', color='#1f77b4', alpha=0.8)

    ax.set_ylabel('Relative Error')
    ax.set_title('HW2D Closure OpInf — Error Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(pipe_errs):
        ax.text(x[i] + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8, color='#1f77b4')
    if Y_probe is not None:
        for i, v in enumerate(probe_errs):
            ax.text(x[i] - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8, color='#d62728')

    plt.tight_layout()
    path3 = os.path.join(out_dir, 'gamma_error_comparison.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"Saved: {path3}")
    plt.close()

    # Print error summary
    print("\n" + "="*60)
    print("ERROR SUMMARY")
    print("="*60)
    if Y_probe is not None:
        print(f"  {'Metric':<15} {'Probe':>10} {'Pipeline':>10} {'Δ':>10}")
        print(f"  {'-'*45}")
        for l, p, q in zip(labels, probe_errs, pipe_errs):
            delta = (q - p) / p * 100 if p > 0 else 0
            print(f"  {l:<15} {p:>10.4f} {q:>10.4f} {delta:>+9.1f}%")
    else:
        print(f"  {'Metric':<15} {'Pipeline':>10}")
        for l, q in zip(labels, pipe_errs):
            print(f"  {l:<15} {q:>10.4f}")


if __name__ == '__main__':
    main()
