"""
Parametric OpInf — data-matrix assembly for affine-mu pOpInf (B2).

Implements the affine-parametric data matrix construction following
McQuarrie, Khodabakhshi & Willcox 2023 (arXiv:2110.07653), adapted to
the discrete-time form used by the rest of opinf/.

Ansatz (discrete time, q=2 linear-in-mu for each operator):

    x_{k+1} = A(mu) x_k + F(mu) quad(x_k) + c(mu)
    A(mu) = A0 + mu * A1
    F(mu) = F0 + mu * F1
    c(mu) = c0 + mu * c1

Per-row data layout (for a snapshot x_k taken at parameter mu_i):

    d_k(mu_i) = [ x_k | mu_i * x_k | quad(x_k) | mu_i * quad(x_k) | 1 | mu_i ]

with widths (r, r, s, s, 1, 1) where s = r(r+1)/2.

The full data matrix D stacks rows from all training mu values; the
target matrix Y stacks the corresponding x_{k+1}. A single regularized
least squares solves for the operator block-stack
[A0 A1 F0 F1 c0 c1] of shape (r, 2(r+s+1)).

This module contains only stateless numpy helpers — no MPI, no I/O.
Smoke-testable in isolation.

Author: Anthony Poole
"""

from __future__ import annotations

import numpy as np
from typing import Sequence, Mapping

from core import get_quadratic_terms


def check_affine_feature_rank(mus: Sequence[float],
                              q: int = 2,
                              tol: float = 1e-12) -> dict:
    """
    Theorem 2.3 (McQuarrie-Khodabakhshi-Willcox 2023) sanity check.

    For affine library theta(mu) = [1, mu, mu^2, ..., mu^{q-1}], assemble
    Theta in R^{s x q} where s = len(mus) and check it has full column rank.

    Parameters
    ----------
    mus : sequence of float
        Training parameter values.
    q : int
        Number of affine features (q=2 for linear-in-mu).
    tol : float
        Threshold below which a singular value is considered zero.

    Returns
    -------
    dict with keys:
        s, q                : counts
        Theta               : the s x q feature matrix
        singular_values     : sigma(Theta)
        condition_number    : sigma_max / sigma_min
        full_column_rank    : bool
        rank_deficient_blocks : list of operator names that would be
                                rank-deficient (for diagnostic logging).
    """
    mus_arr = np.asarray(mus, dtype=float)
    s = mus_arr.size
    Theta = np.vander(mus_arr, q, increasing=True)  # columns [1, mu, mu^2, ...]
    sv = np.linalg.svd(Theta, compute_uv=False)
    full_rank = bool(s >= q and sv[-1] > tol * sv[0])
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    return {
        "s": int(s),
        "q": int(q),
        "Theta": Theta,
        "singular_values": sv,
        "condition_number": cond,
        "full_column_rank": full_rank,
        "rank_deficient_blocks": [] if full_rank else ["c", "A", "F"],
    }


def parametric_data_matrix_row_widths(r: int) -> dict:
    """
    Column block sizes for the parametric data matrix at reduced dim r.

    Returns the per-block widths and the cumulative column offsets.
    Block order matches `build_parametric_data_matrix`.
    """
    s = r * (r + 1) // 2
    widths = {
        "A0": r,
        "A1": r,
        "F0": s,
        "F1": s,
        "c0": 1,
        "c1": 1,
    }
    total = sum(widths.values())  # = 2*(r + s + 1)
    offsets = {}
    col = 0
    for name, w in widths.items():
        offsets[name] = (col, col + w)
        col += w
    return {"widths": widths, "offsets": offsets, "total": total, "s": s}


def build_parametric_data_matrix(
    Xhat_per_alpha: Mapping[float, np.ndarray],
    Yhat_per_alpha: Mapping[float, np.ndarray],
) -> dict:
    """
    Assemble the affine-parametric data matrix D and target Y by stacking
    rows across training alphas with alpha-blocked column structure.

    For each training alpha and each pair (x_k, x_{k+1}) projected to the
    pooled POD basis, the row contribution to D is:

        [ x_k | alpha * x_k | quad(x_k) | alpha * quad(x_k) | 1 | alpha ]

    and the row contribution to Y is x_{k+1}.

    Parameters
    ----------
    Xhat_per_alpha : dict {alpha -> ndarray (K_alpha, r)}
        Projected state snapshots at each training alpha (the "predictor"
        side: x_k for k = 0 .. K_alpha-1).
    Yhat_per_alpha : dict {alpha -> ndarray (K_alpha, r)}
        Projected target snapshots (x_{k+1}, same K_alpha as predictor).

    Returns
    -------
    dict with keys:
        D       : (K_total, 2*(r+s+1)) data matrix
        Y       : (K_total, r) target matrix
        alphas  : (K_total,) per-row alpha labels (for diagnostics)
        layout  : output of parametric_data_matrix_row_widths(r)
        K_per_alpha : dict {alpha -> K_alpha}
    """
    alphas = sorted(Xhat_per_alpha.keys())
    if not alphas:
        raise ValueError("Xhat_per_alpha is empty")
    r = Xhat_per_alpha[alphas[0]].shape[1]
    for a in alphas:
        if Xhat_per_alpha[a].shape[1] != r:
            raise ValueError(
                f"Inconsistent r across alphas: alpha={a} has r="
                f"{Xhat_per_alpha[a].shape[1]}, expected {r}"
            )
        if Yhat_per_alpha[a].shape != Xhat_per_alpha[a].shape:
            raise ValueError(
                f"X/Y shape mismatch at alpha={a}: X={Xhat_per_alpha[a].shape}, "
                f"Y={Yhat_per_alpha[a].shape}"
            )

    layout = parametric_data_matrix_row_widths(r)
    D_blocks = []
    Y_blocks = []
    alpha_labels = []
    K_per_alpha = {}
    for alpha in alphas:
        X = Xhat_per_alpha[alpha]
        Y = Yhat_per_alpha[alpha]
        K_alpha = X.shape[0]
        K_per_alpha[float(alpha)] = K_alpha
        Q = get_quadratic_terms(X)             # (K_alpha, s)
        ones = np.ones((K_alpha, 1))
        alphas_col = alpha * ones
        # row layout: [X | alpha*X | Q | alpha*Q | 1 | alpha]
        row = np.concatenate([X, alpha * X, Q, alpha * Q, ones, alphas_col],
                             axis=1)
        D_blocks.append(row)
        Y_blocks.append(Y)
        alpha_labels.append(np.full(K_alpha, alpha))

    D = np.vstack(D_blocks)
    Y_out = np.vstack(Y_blocks)
    alpha_vec = np.concatenate(alpha_labels)
    return {
        "D": D,
        "Y": Y_out,
        "alphas": alpha_vec,
        "layout": layout,
        "K_per_alpha": K_per_alpha,
    }


def split_operator_block(
    O: np.ndarray,
    r: int,
) -> dict:
    """
    Split the solved operator block-stack into named affine blocks.

    Parameters
    ----------
    O : ndarray (r, 2*(r+s+1))
        Solution of the parametric regression.
    r : int
        Reduced dimension.

    Returns
    -------
    dict with keys A0, A1, F0, F1, c0, c1 in their natural shapes:
        A0, A1 : (r, r)
        F0, F1 : (r, s)
        c0, c1 : (r,)
    """
    layout = parametric_data_matrix_row_widths(r)
    if O.shape != (r, layout["total"]):
        raise ValueError(
            f"O has shape {O.shape}, expected ({r}, {layout['total']})"
        )
    off = layout["offsets"]
    blocks = {}
    for name in ("A0", "A1", "F0", "F1", "c0", "c1"):
        lo, hi = off[name]
        block = O[:, lo:hi]
        if name in ("c0", "c1"):
            block = block.reshape(r)
        blocks[name] = block
    return blocks


def assemble_operators_at_alpha(
    blocks: Mapping[str, np.ndarray],
    alpha: float,
) -> dict:
    """
    Combine the affine blocks at a queried alpha to produce the
    parameter-specific operators A(alpha), F(alpha), c(alpha).
    """
    return {
        "A": blocks["A0"] + alpha * blocks["A1"],
        "F": blocks["F0"] + alpha * blocks["F1"],
        "c": blocks["c0"] + alpha * blocks["c1"],
    }


def parametric_regularizer_diagonal(
    layout: Mapping,
    lambda1: float,
    lambda2: float,
) -> np.ndarray:
    """
    Build the diagonal Tikhonov regularizer for the grouped two-hyperparameter
    form (McQuarrie Eq 4.3, adapted to discrete time with constant):

        ||Lambda O^T||_F^2 = lambda1^2 * (||A0||^2 + ||A1||^2 + ||c0||^2 + ||c1||^2)
                           + lambda2^2 * (||F0||^2 + ||F1||^2)

    Returns a vector of length 2*(r+s+1) suitable for np.diag in the
    augmented normal equations (D^T D + diag(reg)) O^T = D^T Y.

    Note: the augmented normal equations use the *squared* lambdas as
    entries on the diagonal; callers can pass (lambda1**2, lambda2**2)
    or pass lambdas and we square here. We square here for consistency
    with existing `solve_opinf_operators` (which receives alpha = lambda^2).
    """
    total = layout["total"]
    off = layout["offsets"]
    reg = np.zeros(total)
    lam1_sq = float(lambda1)        # caller already passed lambda^2 in opinf convention
    lam2_sq = float(lambda2)
    for name in ("A0", "A1", "c0", "c1"):
        lo, hi = off[name]
        reg[lo:hi] = lam1_sq
    for name in ("F0", "F1"):
        lo, hi = off[name]
        reg[lo:hi] = lam2_sq
    return reg
