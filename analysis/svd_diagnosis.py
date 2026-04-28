"""
SVD Diagnosis: understand the regularization landscape from data.

Computes the singular value decomposition of the OpInf data matrix,
both as a whole and per block (linear, quadratic, cubic diagonal,
constant), to reveal the natural regularization scales.

Key outputs:
  - Singular value spectra per block
  - Condition numbers
  - Recommended alpha ranges (alpha ~ sigma_min^2 of each block)
  - Diagnostic plots

Usage:
    python analysis/svd_diagnosis.py --run-dir /path/to/step1_output
    python analysis/svd_diagnosis.py --run-dir /path/to/step1_output --save-plots

Author: Anthony Poole
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'opinf'))


def load_learning_data(run_dir: str) -> dict:
    """Load pre-computed learning matrices from step 1 output."""
    lm_path = os.path.join(run_dir, 'learning_matrices.npz')
    learning = np.load(lm_path, allow_pickle=True)

    data = {
        'D_state': learning['D_state'],
        'Y_state': learning['Y_state'],
        'D_state_2': learning['D_state_2'],
        'D_out': learning['D_out'],
        'D_out_2': learning['D_out_2'],
        'X_state': learning['X_state'],
        'mean_Xhat': learning['mean_Xhat'],
        'scaling_Xhat': float(learning['scaling_Xhat']),
        'include_cubic': bool(learning.get('include_cubic', False)),
        'include_constant': bool(learning.get('include_constant', False)),
    }

    # Load gamma reference
    gamma_path = os.path.join(run_dir, 'gamma_reference.npz')
    if os.path.exists(gamma_path):
        gamma_ref = np.load(gamma_path, allow_pickle=True)
        data['Y_Gamma'] = gamma_ref['Y_Gamma']
        data['mean_Gamma_n'] = float(gamma_ref['mean_Gamma_n'])
        data['std_Gamma_n'] = float(gamma_ref['std_Gamma_n'])
        data['mean_Gamma_c'] = float(gamma_ref['mean_Gamma_c'])
        data['std_Gamma_c'] = float(gamma_ref['std_Gamma_c'])
        gamma_ref.close()

    learning.close()
    return data


def decompose_data_matrix(D_state: np.ndarray, r: int,
                          include_cubic: bool, include_constant: bool) -> dict:
    """Split D_state into its constituent blocks."""
    s = r * (r + 1) // 2
    col = 0

    blocks = {}
    blocks['linear'] = D_state[:, col:col + r]
    col += r
    blocks['quadratic'] = D_state[:, col:col + s]
    col += s
    if include_cubic:
        blocks['cubic_diag'] = D_state[:, col:col + r]
        col += r
    if include_constant:
        blocks['constant'] = D_state[:, col:col + 1]
        col += 1

    return blocks


def compute_block_svd(blocks: dict) -> dict:
    """Compute SVD for each block of the data matrix."""
    results = {}
    for name, block in blocks.items():
        t0 = time.time()
        # Compute singular values only (faster than full SVD)
        sigma = np.linalg.svd(block, compute_uv=False)
        elapsed = time.time() - t0

        results[name] = {
            'sigma': sigma,
            'shape': block.shape,
            'cond': sigma[0] / sigma[-1] if sigma[-1] > 0 else np.inf,
            'sigma_max': sigma[0],
            'sigma_min': sigma[-1],
            'rank': np.sum(sigma > sigma[0] * 1e-14),
            'time': elapsed,
        }

    return results


def compute_full_svd(D_state: np.ndarray) -> dict:
    """Compute SVD of the full data matrix."""
    t0 = time.time()
    sigma = np.linalg.svd(D_state, compute_uv=False)
    elapsed = time.time() - t0

    return {
        'sigma': sigma,
        'shape': D_state.shape,
        'cond': sigma[0] / sigma[-1] if sigma[-1] > 0 else np.inf,
        'sigma_max': sigma[0],
        'sigma_min': sigma[-1],
        'rank': np.sum(sigma > sigma[0] * 1e-14),
        'time': elapsed,
    }


def recommend_alpha_ranges(block_svds: dict) -> dict:
    """
    Recommend regularization ranges based on singular value analysis.

    For Tikhonov regularization (D^T D + alpha I), the natural scale is:
      - alpha << sigma_min^2  -> under-regularized (ill-conditioned)
      - alpha ~  sigma_min^2  -> balanced (GCV-optimal neighborhood)
      - alpha >> sigma_max^2  -> over-regularized (suppresses all signal)

    We recommend sweeping from 0.01 * sigma_min^2 to 10 * sigma_max^2,
    with the "center" at sigma_min^2.
    """
    recommendations = {}
    for name, svd in block_svds.items():
        s_min2 = svd['sigma_min'] ** 2
        s_max2 = svd['sigma_max'] ** 2

        recommendations[name] = {
            'alpha_low': 1e-2 * s_min2,
            'alpha_center': s_min2,
            'alpha_high': 10 * s_max2,
            'sigma_min_sq': s_min2,
            'sigma_max_sq': s_max2,
        }

    return recommendations


def print_diagnosis(full_svd: dict, block_svds: dict, recommendations: dict):
    """Print a formatted diagnosis report."""
    print("\n" + "=" * 70)
    print("SVD DIAGNOSIS REPORT")
    print("=" * 70)

    print(f"\nFull D_state: {full_svd['shape']}")
    print(f"  Condition number: {full_svd['cond']:.2e}")
    print(f"  Numerical rank:   {full_svd['rank']} / {full_svd['shape'][1]}")
    print(f"  σ_max: {full_svd['sigma_max']:.4e}  σ_min: {full_svd['sigma_min']:.4e}")
    print(f"  SVD time: {full_svd['time']:.1f}s")

    print("\n--- Block-wise analysis ---")
    for name, svd in block_svds.items():
        rec = recommendations[name]
        print(f"\n  [{name}] shape={svd['shape']}")
        print(f"    σ_max={svd['sigma_max']:.4e}  σ_min={svd['sigma_min']:.4e}  "
              f"cond={svd['cond']:.2e}  rank={svd['rank']}/{svd['shape'][1]}")
        print(f"    Recommended α range: [{rec['alpha_low']:.2e}, {rec['alpha_high']:.2e}]")
        print(f"    Center (σ_min²):     {rec['alpha_center']:.2e}")
        print(f"    σ_max²:              {rec['sigma_max_sq']:.2e}")

    # Map block names to config parameter names
    param_map = {
        'linear': 'state_lin',
        'quadratic': 'state_quad',
        'cubic_diag': 'state_cubic',
    }

    print("\n--- Suggested config ranges ---")
    for name, rec in recommendations.items():
        param_name = param_map.get(name, name)
        if name == 'constant':
            continue
        log_low = np.log10(max(rec['alpha_low'], 1e-30))
        log_high = np.log10(max(rec['alpha_high'], 1e-30))
        print(f"  {param_name}: min={10**log_low:.1e}, max={10**log_high:.1e}, "
              f"num=15, scale=log")

    print("\n" + "=" * 70)


def plot_singular_values(full_svd: dict, block_svds: dict, recommendations: dict,
                         save_path: str = None):
    """Plot singular value spectra."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_blocks = len(block_svds)
    fig, axes = plt.subplots(1, n_blocks + 1, figsize=(4 * (n_blocks + 1), 4))

    if n_blocks + 1 == 1:
        axes = [axes]

    # Full matrix
    ax = axes[0]
    ax.semilogy(full_svd['sigma'], 'b-', linewidth=1)
    ax.set_title(f"Full D_state\ncond={full_svd['cond']:.1e}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value")
    ax.grid(True, alpha=0.3)

    # Per block
    colors = {'linear': 'C0', 'quadratic': 'C1', 'cubic_diag': 'C2', 'constant': 'C3'}
    for i, (name, svd) in enumerate(block_svds.items()):
        ax = axes[i + 1]
        color = colors.get(name, f'C{i}')
        ax.semilogy(svd['sigma'], color=color, linewidth=1)
        rec = recommendations[name]
        ax.axhline(np.sqrt(rec['alpha_center']), color='r', linestyle='--',
                    alpha=0.7, label=f'σ_min={svd["sigma_min"]:.1e}')
        ax.set_title(f"{name} ({svd['shape'][1]} cols)\ncond={svd['cond']:.1e}")
        ax.set_xlabel("Index")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("SVD Diagnosis — Singular Value Spectra", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved SVD plot to {save_path}")
    else:
        plt.savefig('svd_diagnosis.png', dpi=150, bbox_inches='tight')
        print("Saved SVD plot to svd_diagnosis.png")

    plt.close()


def save_results(full_svd: dict, block_svds: dict, recommendations: dict,
                 save_path: str):
    """Save SVD diagnosis results to npz."""
    save_dict = {
        'full_sigma': full_svd['sigma'],
        'full_cond': full_svd['cond'],
        'full_shape': full_svd['shape'],
    }
    for name, svd in block_svds.items():
        save_dict[f'{name}_sigma'] = svd['sigma']
        save_dict[f'{name}_cond'] = svd['cond']
        save_dict[f'{name}_shape'] = svd['shape']
    for name, rec in recommendations.items():
        for key, val in rec.items():
            save_dict[f'rec_{name}_{key}'] = val

    np.savez(save_path, **save_dict)
    print(f"Saved SVD diagnosis to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="SVD diagnosis for OpInf regularization")
    parser.add_argument('--run-dir', required=True, help="Path to step 1 output directory")
    parser.add_argument('--save-plots', action='store_true', help="Save diagnostic plots")
    parser.add_argument('--output-dir', default=None, help="Output directory (default: run-dir)")
    args = parser.parse_args()

    out_dir = args.output_dir or args.run_dir

    print(f"Loading data from: {args.run_dir}")
    data = load_learning_data(args.run_dir)

    D_state = data['D_state']
    K, d = D_state.shape
    r = data['X_state'].shape[1]
    s = r * (r + 1) // 2
    include_cubic = data['include_cubic']
    include_constant = data['include_constant']

    print(f"D_state shape: ({K}, {d}), r={r}, s={s}")
    print(f"Closure: cubic={include_cubic}, constant={include_constant}")

    # Block decomposition
    blocks = decompose_data_matrix(D_state, r, include_cubic, include_constant)
    print(f"Blocks: {list(blocks.keys())}")

    # Compute SVDs
    print("\nComputing SVDs...")
    full_svd = compute_full_svd(D_state)
    block_svds = compute_block_svd(blocks)

    # Recommendations
    recommendations = recommend_alpha_ranges(block_svds)

    # Report
    print_diagnosis(full_svd, block_svds, recommendations)

    # Save results
    save_results(full_svd, block_svds, recommendations,
                 os.path.join(out_dir, 'svd_diagnosis.npz'))

    if args.save_plots:
        plot_singular_values(full_svd, block_svds, recommendations,
                             os.path.join(out_dir, 'svd_diagnosis.png'))

    return full_svd, block_svds, recommendations


if __name__ == '__main__':
    main()
