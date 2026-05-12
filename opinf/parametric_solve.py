"""
Parametric OpInf — solver and hyperparameter sweep for affine-mu pOpInf (B2).

This module implements:
- the grouped two-hyperparameter Tikhonov solve (lambda1, lambda2);
- discrete-time rollout at a queried alpha;
- the 2D hyperparameter sweep with stability-constrained selection
  (a sentinel alpha disqualifies any candidate that blows up).

McQuarrie-Khodabakhshi-Willcox 2023 §4 establishes that good selection
strategies (a) score by mean-squared rollout error across training
parameters and (b) disqualify candidates that produce divergent
integration. We adopt both; the sentinel alpha lets us require finite
rollout at a held-out parameter without using its data for training.

This module is pure numpy. No MPI, no I/O.

Author: Anthony Poole
"""

from __future__ import annotations

import numpy as np
from itertools import product
from typing import Sequence, Mapping, Callable

from core import get_quadratic_terms, solve_difference_model
from parametric_data import (
    split_operator_block,
    assemble_operators_at_alpha,
    parametric_regularizer_diagonal,
)


def solve_parametric_opinf(
    D: np.ndarray,
    Y: np.ndarray,
    layout: Mapping,
    lambda1_sq: float,
    lambda2_sq: float,
) -> dict:
    """
    Solve the grouped-Tikhonov parametric least squares problem:

        min_{O}  || D O^T - Y ||_F^2  +  || Lambda(lambda1, lambda2) O^T ||_F^2

    via the augmented normal equations.

    Parameters
    ----------
    D : ndarray (K_total, total_cols)
        Parametric data matrix from `build_parametric_data_matrix`.
    Y : ndarray (K_total, r)
        Target matrix.
    layout : dict
        Output of `parametric_data_matrix_row_widths(r)`.
    lambda1_sq, lambda2_sq : float
        Regularization weights (already squared, per opinf convention
        where `alpha_lin` etc. are squared lambdas).

    Returns
    -------
    dict with the operator blocks {A0, A1, F0, F1, c0, c1}.
    """
    r = Y.shape[1]
    reg = parametric_regularizer_diagonal(layout, lambda1_sq, lambda2_sq)
    DtD = D.T @ D + np.diag(reg)
    DtY = D.T @ Y
    O = np.linalg.solve(DtD, DtY).T   # (r, total_cols)
    return split_operator_block(O, r)


def make_alpha_stepper(blocks: Mapping[str, np.ndarray],
                       alpha: float) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a one-step transition function f_alpha(x) for use with
    `core.solve_difference_model`.

        f_alpha(x) = A(alpha) x + F(alpha) quad(x) + c(alpha)
    """
    ops = assemble_operators_at_alpha(blocks, alpha)
    A, F, c = ops["A"], ops["F"], ops["c"]

    def f(x: np.ndarray) -> np.ndarray:
        return A @ x + F @ get_quadratic_terms(x) + c

    return f


def rollout_at_alpha(blocks: Mapping[str, np.ndarray],
                     alpha: float,
                     x0: np.ndarray,
                     n_steps: int) -> tuple:
    """
    Roll out the parametric ROM forward in time at a chosen alpha.

    Returns
    -------
    is_nan : bool
    Xhat   : ndarray (r, n_steps)   the trajectory
    """
    f = make_alpha_stepper(blocks, alpha)
    return solve_difference_model(x0, n_steps, f)


def rollout_is_bounded(Xhat: np.ndarray,
                       is_nan: bool,
                       max_state_norm: float = 1e3) -> bool:
    """
    Cheap stability check used by the sentinel-alpha disqualifier.

    A rollout is "bounded" iff:
      - the integrator did not return NaN/Inf, AND
      - the per-step L2 norm stays below `max_state_norm`.

    The norm threshold is conservative; we are not asking the rollout
    to be *accurate* at the sentinel alpha, only to be *finite*. If the
    centered reduced state ever exceeds the threshold, the operators
    are almost certainly unphysical at that alpha.
    """
    if is_nan:
        return False
    if not np.isfinite(Xhat).all():
        return False
    step_norms = np.linalg.norm(Xhat, axis=0)
    return bool(step_norms.max() < max_state_norm)


def evaluate_parametric_candidate(
    D: np.ndarray,
    Y: np.ndarray,
    layout: Mapping,
    lambda1_sq: float,
    lambda2_sq: float,
    training_ics: Mapping[float, np.ndarray],
    training_targets: Mapping[float, np.ndarray],
    n_steps: int,
    sentinel_alpha: float = None,
    sentinel_ic: np.ndarray = None,
    sentinel_steps: int = None,
    sentinel_max_norm: float = 1e3,
) -> dict:
    """
    Train a candidate (lambda1, lambda2) on the pooled data, then score it.

    Score:
      - For each training alpha, roll out from the provided IC for n_steps
        steps, compare to the training target trajectory (in reduced coords).
        Score is the mean per-alpha mean-squared coordinate error,
        averaged across training alphas.
      - If `sentinel_alpha` is provided, also roll out at the sentinel
        from `sentinel_ic` for `sentinel_steps` steps. If that rollout is
        unbounded, mark the candidate as disqualified.

    Returns
    -------
    dict with keys:
        is_disqualified : bool
        is_nan_train : bool (any training rollout went NaN)
        sentinel_bounded : bool or None
        per_alpha_mse : dict {alpha -> float}
        score : float (mean over training alphas; np.inf if disqualified)
        lambda1_sq, lambda2_sq : the candidate
    """
    blocks = solve_parametric_opinf(D, Y, layout, lambda1_sq, lambda2_sq)

    # Score on training alphas
    per_alpha_mse = {}
    any_nan = False
    for alpha, x0 in training_ics.items():
        is_nan, Xhat_pred = rollout_at_alpha(blocks, alpha, x0, n_steps)
        if is_nan or not np.isfinite(Xhat_pred).all():
            any_nan = True
            per_alpha_mse[alpha] = float("inf")
            continue
        target = training_targets[alpha]
        # target shape: (n_steps, r); Xhat_pred shape: (r, n_steps)
        diff = Xhat_pred.T - target
        per_alpha_mse[alpha] = float(np.mean(diff ** 2))

    score = (
        float("inf")
        if any_nan
        else float(np.mean(list(per_alpha_mse.values())))
    )

    # Sentinel check
    sentinel_bounded = None
    is_disqualified = any_nan
    if sentinel_alpha is not None and sentinel_ic is not None:
        s_steps = sentinel_steps if sentinel_steps is not None else n_steps
        is_nan_s, Xhat_s = rollout_at_alpha(
            blocks, sentinel_alpha, sentinel_ic, s_steps
        )
        sentinel_bounded = rollout_is_bounded(
            Xhat_s, is_nan_s, max_state_norm=sentinel_max_norm
        )
        if not sentinel_bounded:
            is_disqualified = True
            score = float("inf")

    return {
        "is_disqualified": is_disqualified,
        "is_nan_train": any_nan,
        "sentinel_bounded": sentinel_bounded,
        "per_alpha_mse": per_alpha_mse,
        "score": score,
        "lambda1_sq": float(lambda1_sq),
        "lambda2_sq": float(lambda2_sq),
    }


def parametric_hyperparam_sweep(
    D: np.ndarray,
    Y: np.ndarray,
    layout: Mapping,
    lambda1_grid: Sequence[float],
    lambda2_grid: Sequence[float],
    training_ics: Mapping[float, np.ndarray],
    training_targets: Mapping[float, np.ndarray],
    n_steps: int,
    sentinel_alpha: float = None,
    sentinel_ic: np.ndarray = None,
    sentinel_steps: int = None,
    sentinel_max_norm: float = 1e3,
    logger=None,
) -> list:
    """
    2D grid search over (lambda1_sq, lambda2_sq) with the stability-
    constrained scoring of `evaluate_parametric_candidate`.

    Returns a list of result dicts, sorted by ascending `score`. The
    first non-disqualified entry is the best candidate.

    This is the serial reference implementation; the MPI version in
    `step_2_train_parametric.py` slices the grid across ranks.
    """
    results = []
    n_total = len(lambda1_grid) * len(lambda2_grid)
    for idx, (l1, l2) in enumerate(product(lambda1_grid, lambda2_grid)):
        res = evaluate_parametric_candidate(
            D=D, Y=Y, layout=layout,
            lambda1_sq=l1, lambda2_sq=l2,
            training_ics=training_ics,
            training_targets=training_targets,
            n_steps=n_steps,
            sentinel_alpha=sentinel_alpha,
            sentinel_ic=sentinel_ic,
            sentinel_steps=sentinel_steps,
            sentinel_max_norm=sentinel_max_norm,
        )
        results.append(res)
        if logger is not None and (idx + 1) % max(1, n_total // 10) == 0:
            logger.info(
                f"  sweep {idx + 1}/{n_total}: lambda1_sq={l1:.2e} "
                f"lambda2_sq={l2:.2e} score={res['score']:.4e} "
                f"disq={res['is_disqualified']} "
                f"sentinel_ok={res['sentinel_bounded']}"
            )
    results.sort(key=lambda r: r["score"])
    return results
