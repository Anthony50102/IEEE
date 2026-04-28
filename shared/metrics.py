"""
Machine-readable run summary and metrics for automated comparison.

This module provides a standardized run_summary.yaml output that Copilot
(or any tool) can parse to determine whether one run is better than another
without reading plots.

The summary includes:
- Run metadata (method, PDE, config, timestamp)
- Scalar metrics in a flat, comparable structure
- Per-trajectory breakdowns
- A quality verdict with interpretive thresholds

Usage in step_3_evaluate.py:
    from shared.metrics import RunSummary
    summary = RunSummary(method="opinf", pde="ks", run_dir=args.run_dir,
                         config_path=args.config)
    summary.add_qoi_metrics(pred_gamma_n, ref_gamma_n, pred_gamma_c, ref_gamma_c,
                            split="test", trajectory=0)
    summary.add_state_metrics(pred_states, ref_states, split="test", trajectory=0)
    summary.save()

Author: Anthony Poole
"""

import os
import hashlib
import datetime
import numpy as np
import yaml


# =============================================================================
# METRIC COMPUTATION FUNCTIONS
# =============================================================================

def relative_error(pred: float, ref: float) -> float:
    """Relative error |pred - ref| / |ref|, returns nan if ref is zero."""
    if abs(ref) < 1e-15:
        return float('nan')
    return abs(pred - ref) / abs(ref)


def nrmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Normalized RMSE: RMSE / std(ref). Scale-free error measure."""
    rmse = np.sqrt(np.mean((pred - ref) ** 2))
    ref_std = np.std(ref, ddof=1)
    if ref_std < 1e-15:
        return float('nan')
    return float(rmse / ref_std)


def pearson_correlation(pred: np.ndarray, ref: np.ndarray) -> float:
    """Pearson correlation coefficient between two time series."""
    if len(pred) < 2 or np.std(pred) < 1e-15 or np.std(ref) < 1e-15:
        return float('nan')
    return float(np.corrcoef(pred, ref)[0, 1])


def valid_prediction_time(
    pred: np.ndarray,
    ref: np.ndarray,
    dt: float,
    correlation_threshold: float = 0.8,
    window: int = 1,
) -> float:
    """
    Time at which rolling correlation drops below threshold.

    A longer valid prediction time means the model tracks the reference
    dynamics for more of the forecast horizon. Returns the full forecast
    duration if correlation never drops below threshold.

    Parameters
    ----------
    pred, ref : np.ndarray, shape (n_time,)
    dt : float
        Time step between snapshots.
    correlation_threshold : float
        Threshold below which prediction is considered invalid.
    window : int
        Half-width of rolling window for smoothing. 1 = no smoothing.

    Returns
    -------
    float
        Time in simulation units at which prediction becomes invalid.
    """
    n = len(pred)
    if n < 2 * window + 1:
        return 0.0

    for i in range(window, n - window):
        seg_pred = pred[i - window:i + window + 1]
        seg_ref = ref[i - window:i + window + 1]
        if np.std(seg_pred) < 1e-15 or np.std(seg_ref) < 1e-15:
            return float(i * dt)
        corr = np.corrcoef(seg_pred, seg_ref)[0, 1]
        if corr < correlation_threshold:
            return float(i * dt)

    return float(n * dt)


def compute_qoi_scalar_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    dt: float = 1.0,
    label: str = "qoi",
) -> dict:
    """
    Compute a full set of scalar metrics for a single QoI time series.

    Returns a flat dict with keys prefixed by `label`.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    ref_mean = float(np.mean(ref))
    ref_std = float(np.std(ref, ddof=1))
    pred_mean = float(np.mean(pred))
    pred_std = float(np.std(pred, ddof=1))

    rmse_val = float(np.sqrt(np.mean((pred - ref) ** 2)))
    mse_val = float(np.mean((pred - ref) ** 2))
    nrmse_val = nrmse(pred, ref)
    corr = pearson_correlation(pred, ref)
    vpt = valid_prediction_time(pred, ref, dt)

    return {
        f'{label}_ref_mean': ref_mean,
        f'{label}_ref_std': ref_std,
        f'{label}_pred_mean': pred_mean,
        f'{label}_pred_std': pred_std,
        f'{label}_err_mean': relative_error(pred_mean, ref_mean),
        f'{label}_err_std': relative_error(pred_std, ref_std),
        f'{label}_rmse': rmse_val,
        f'{label}_mse': mse_val,
        f'{label}_nrmse': nrmse_val,
        f'{label}_correlation': corr,
        f'{label}_valid_time': vpt,
    }


def compute_state_scalar_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
) -> dict:
    """
    Compute state-level metrics from full-field predictions.

    Parameters
    ----------
    pred, ref : np.ndarray, shape (n_time, ...) or (n_time, n_spatial)
        Predicted and reference state arrays. Flattened per timestep.

    Returns
    -------
    dict with flat scalar metrics.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    n_time = pred.shape[0]
    pred_flat = pred.reshape(n_time, -1)
    ref_flat = ref.reshape(n_time, -1)

    # Relative L2 error per timestep
    ref_norms = np.linalg.norm(ref_flat, axis=1)
    safe_norms = np.where(ref_norms < 1e-15, 1.0, ref_norms)
    l2_errors = np.linalg.norm(pred_flat - ref_flat, axis=1) / safe_norms

    mse_total = float(np.mean((pred_flat - ref_flat) ** 2))

    return {
        'state_mse': mse_total,
        'state_rel_l2_mean': float(np.mean(l2_errors)),
        'state_rel_l2_max': float(np.max(l2_errors)),
        'state_rel_l2_final': float(l2_errors[-1]),
        'state_rel_l2_median': float(np.median(l2_errors)),
    }


def _config_hash(config_path: str) -> str:
    """SHA256 of the config file for reproducibility tracking."""
    try:
        with open(config_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except (FileNotFoundError, TypeError):
        return "unknown"


# =============================================================================
# QUALITY VERDICTS
# =============================================================================

# Thresholds for interpreting metric quality (for the primary test QoI)
_THRESHOLDS = {
    'excellent': {'nrmse': 0.10, 'correlation': 0.95, 'err_mean': 0.05},
    'good':      {'nrmse': 0.30, 'correlation': 0.80, 'err_mean': 0.15},
    'fair':      {'nrmse': 0.60, 'correlation': 0.50, 'err_mean': 0.30},
    # worse than 'fair' is 'poor'
}


def _grade(nrmse_val: float, corr: float, err_mean: float) -> str:
    """Assign a quality grade based on primary test metrics."""
    for grade in ['excellent', 'good', 'fair']:
        t = _THRESHOLDS[grade]
        if nrmse_val <= t['nrmse'] and corr >= t['correlation'] and err_mean <= t['err_mean']:
            return grade
    return 'poor'


# =============================================================================
# RUN SUMMARY CLASS
# =============================================================================

class RunSummary:
    """
    Accumulates metrics during evaluation and writes a standardized
    run_summary.yaml at the end.

    The file is designed to be parsed by Copilot to compare runs.
    """

    def __init__(
        self,
        method: str,
        pde: str,
        run_dir: str,
        config_path: str = None,
        dt: float = 1.0,
    ):
        self.method = method
        self.pde = pde
        self.run_dir = run_dir
        self.config_path = config_path
        self.dt = dt

        self._data = {
            'meta': {
                'method': method,
                'pde': pde,
                'run_dir': run_dir,
                'config_hash': _config_hash(config_path),
                'timestamp': datetime.datetime.now().isoformat(),
                'schema_version': 1,
            },
            'train': {'trajectories': [], 'aggregate': {}},
            'test': {'trajectories': [], 'aggregate': {}},
            'verdict': {},
        }

    # -----------------------------------------------------------------
    # Adding metrics
    # -----------------------------------------------------------------

    def add_qoi_metrics(
        self,
        pred_qoi1: np.ndarray,
        ref_qoi1: np.ndarray,
        pred_qoi2: np.ndarray,
        ref_qoi2: np.ndarray,
        split: str = "test",
        trajectory: int = 0,
        label_1: str = None,
        label_2: str = None,
    ):
        """
        Add QoI metrics for one trajectory.

        Parameters
        ----------
        pred_qoi1, ref_qoi1 : np.ndarray
            Predicted and reference time series for primary QoI
            (Gamma_n for HW2D, Energy for KS).
        pred_qoi2, ref_qoi2 : np.ndarray
            Secondary QoI (Gamma_c / Enstrophy).
        split : str
            "train" or "test".
        trajectory : int
            Trajectory index.
        label_1, label_2 : str, optional
            Override default QoI labels.
        """
        if label_1 is None:
            label_1 = "energy" if self.pde == "ks" else "Gamma_n"
        if label_2 is None:
            label_2 = "enstrophy" if self.pde == "ks" else "Gamma_c"

        m1 = compute_qoi_scalar_metrics(pred_qoi1, ref_qoi1, self.dt, label_1)
        m2 = compute_qoi_scalar_metrics(pred_qoi2, ref_qoi2, self.dt, label_2)

        traj_entry = {
            'trajectory': trajectory,
            'n_steps': len(pred_qoi1),
            **m1,
            **m2,
        }
        self._data[split]['trajectories'].append(traj_entry)

    def add_state_metrics(
        self,
        pred_states: np.ndarray,
        ref_states: np.ndarray,
        split: str = "test",
        trajectory: int = 0,
    ):
        """Add state-level metrics for one trajectory."""
        state_m = compute_state_scalar_metrics(pred_states, ref_states)

        # Find the matching trajectory entry and merge
        for entry in self._data[split]['trajectories']:
            if entry['trajectory'] == trajectory:
                entry.update(state_m)
                return

        # If no QoI entry yet, create a new one
        self._data[split]['trajectories'].append({
            'trajectory': trajectory,
            **state_m,
        })

    # -----------------------------------------------------------------
    # Finalize and save
    # -----------------------------------------------------------------

    def _compute_aggregates(self, split: str):
        """Average per-trajectory metrics into aggregate scalars."""
        trajs = self._data[split]['trajectories']
        if not trajs:
            return

        # Collect all numeric keys
        numeric_keys = [
            k for k in trajs[0]
            if k not in ('trajectory', 'n_steps') and isinstance(trajs[0][k], (int, float))
        ]

        agg = {}
        for key in numeric_keys:
            vals = [t[key] for t in trajs if key in t and not _is_nan(t[key])]
            if vals:
                agg[key] = float(np.mean(vals))

        self._data[split]['aggregate'] = agg

    def _compute_verdict(self):
        """Assign quality grades for the test split."""
        agg = self._data['test'].get('aggregate', {})
        if not agg:
            self._data['verdict'] = {'grade': 'unknown', 'reason': 'no test metrics'}
            return

        # Determine primary QoI label
        qoi1_label = "energy" if self.pde == "ks" else "Gamma_n"

        nrmse_val = agg.get(f'{qoi1_label}_nrmse', float('inf'))
        corr = agg.get(f'{qoi1_label}_correlation', 0.0)
        err_mean = agg.get(f'{qoi1_label}_err_mean', float('inf'))

        grade = _grade(nrmse_val, corr, err_mean)

        self._data['verdict'] = {
            'grade': grade,
            f'{qoi1_label}_nrmse': _safe_float(nrmse_val),
            f'{qoi1_label}_correlation': _safe_float(corr),
            f'{qoi1_label}_err_mean': _safe_float(err_mean),
            'thresholds': _THRESHOLDS,
        }

    def save(self, filename: str = "run_summary.yaml"):
        """Write run_summary.yaml to the run directory."""
        self._compute_aggregates('train')
        self._compute_aggregates('test')
        self._compute_verdict()

        path = os.path.join(self.run_dir, filename)
        with open(path, 'w') as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False,
                      allow_unicode=True)

        return path


# =============================================================================
# COMPARISON UTILITY
# =============================================================================

def compare_runs(summary_a_path: str, summary_b_path: str) -> dict:
    """
    Compare two run_summary.yaml files and return a structured diff.

    Parameters
    ----------
    summary_a_path, summary_b_path : str
        Paths to run_summary.yaml files.

    Returns
    -------
    dict with keys:
        winner : str — "A", "B", or "tie"
        reason : str — human-readable explanation
        metrics : dict — side-by-side metric comparison
    """
    with open(summary_a_path) as f:
        a = yaml.safe_load(f)
    with open(summary_b_path) as f:
        b = yaml.safe_load(f)

    agg_a = a.get('test', {}).get('aggregate', {})
    agg_b = b.get('test', {}).get('aggregate', {})

    # Determine primary QoI label
    pde = a.get('meta', {}).get('pde', 'hw2d')
    qoi1 = "energy" if pde == "ks" else "Gamma_n"
    qoi2 = "enstrophy" if pde == "ks" else "Gamma_c"

    # Compare key metrics (lower is better for errors, higher for correlation)
    comparison = {}
    score_a, score_b = 0, 0

    lower_better = [
        f'{qoi1}_nrmse', f'{qoi1}_err_mean', f'{qoi1}_rmse',
        f'{qoi2}_nrmse', f'{qoi2}_err_mean', f'{qoi2}_rmse',
        'state_mse', 'state_rel_l2_mean',
    ]
    higher_better = [
        f'{qoi1}_correlation', f'{qoi2}_correlation',
        f'{qoi1}_valid_time', f'{qoi2}_valid_time',
    ]

    for key in lower_better:
        va = agg_a.get(key)
        vb = agg_b.get(key)
        if va is not None and vb is not None and not (_is_nan(va) or _is_nan(vb)):
            comparison[key] = {'A': va, 'B': vb, 'better': 'A' if va < vb else 'B'}
            if va < vb:
                score_a += 1
            elif vb < va:
                score_b += 1

    for key in higher_better:
        va = agg_a.get(key)
        vb = agg_b.get(key)
        if va is not None and vb is not None and not (_is_nan(va) or _is_nan(vb)):
            comparison[key] = {'A': va, 'B': vb, 'better': 'A' if va > vb else 'B'}
            if va > vb:
                score_a += 1
            elif vb > va:
                score_b += 1

    if score_a > score_b:
        winner = 'A'
    elif score_b > score_a:
        winner = 'B'
    else:
        winner = 'tie'

    return {
        'winner': winner,
        'score': {'A': score_a, 'B': score_b},
        'run_A': a.get('meta', {}),
        'run_B': b.get('meta', {}),
        'verdict_A': a.get('verdict', {}),
        'verdict_B': b.get('verdict', {}),
        'metrics': comparison,
    }


# =============================================================================
# HELPERS
# =============================================================================

def _is_nan(val) -> bool:
    """Check if a value is NaN (works for float and np types)."""
    try:
        return np.isnan(val)
    except (TypeError, ValueError):
        return False


def _safe_float(val) -> float:
    """Convert to float, handling nan/inf gracefully."""
    try:
        v = float(val)
        return v if np.isfinite(v) else float('nan')
    except (TypeError, ValueError):
        return float('nan')
