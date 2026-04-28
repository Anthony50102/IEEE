"""
Core OpInf algorithms for discrete-time operator inference.

This module contains the core mathematical operations for:
- Quadratic term computation
- Discrete-time model integration
- Regularized least squares solving

These functions are shared across all pipeline steps.

Author: Anthony Poole
"""

import numpy as np
from typing import Callable, Tuple


def get_quadratic_terms(X: np.ndarray) -> np.ndarray:
    """
    Compute non-redundant quadratic terms of X.
    
    For a vector x of length r, computes the upper triangular products:
    [x1*x1, x1*x2, ..., x1*xr, x2*x2, x2*x3, ..., xr*xr]
    
    Parameters
    ----------
    X : np.ndarray
        Input vector (r,) or matrix (K, r).
    
    Returns
    -------
    np.ndarray
        Quadratic terms: (s,) for vector input or (K, s) for matrix input,
        where s = r*(r+1)/2.
    """
    if X.ndim == 1:
        r = X.size
        prods = [X[i] * X[i:] for i in range(r)]
        return np.concatenate(prods)
    
    elif X.ndim == 2:
        K, r = X.shape
        prods = [X[:, i:i+1] * X[:, i:] for i in range(r)]
        return np.concatenate(prods, axis=1)
    
    else:
        raise ValueError(f"Invalid input shape: {X.shape}")


def get_cubic_diagonal_terms(X: np.ndarray) -> np.ndarray:
    """
    Compute diagonal cubic terms: [x1^3, x2^3, ..., xr^3].
    
    Parameters
    ----------
    X : np.ndarray
        Input vector (r,) or matrix (K, r).
    
    Returns
    -------
    np.ndarray
        Cubic diagonal terms: (r,) for vector input or (K, r) for matrix input.
    """
    return X ** 3


def project_to_stable(A: np.ndarray, max_spectral_radius: float = 0.999) -> np.ndarray:
    """
    Project the linear operator A so all eigenvalues satisfy |λ| ≤ max_spectral_radius.
    
    Eigenvalues exceeding the threshold are scaled down to the boundary of the
    stability disk while preserving their phase. This ensures the discrete-time
    system x_{k+1} = A x_k + ... does not exhibit exponential energy growth.
    """
    eigs, V = np.linalg.eig(A)
    magnitudes = np.abs(eigs)
    scale = np.where(magnitudes > max_spectral_radius,
                     max_spectral_radius / magnitudes, 1.0)
    eigs_clipped = eigs * scale
    A_stable = (V @ np.diag(eigs_clipped) @ np.linalg.inv(V)).real
    return A_stable


def solve_difference_model(
    x0: np.ndarray, 
    n_steps: int, 
    f: Callable[[np.ndarray], np.ndarray],
) -> Tuple[bool, np.ndarray]:
    """
    Integrate a discrete-time dynamical system forward.
    
    Solves the difference equation: x_{k+1} = f(x_k)
    
    Parameters
    ----------
    x0 : np.ndarray
        Initial state vector of shape (r,).
    n_steps : int
        Number of time steps to integrate.
    f : callable
        State transition function f: R^r -> R^r.
    
    Returns
    -------
    is_nan : bool
        True if NaN values were encountered during integration.
    X : np.ndarray
        State trajectory of shape (r, n_steps).
    """
    r = x0.size
    X = np.zeros((r, n_steps))
    X[:, 0] = x0
    
    for k in range(n_steps - 1):
        X[:, k + 1] = f(X[:, k])
        
        # Check for NaN or Inf (model blew up)
        if np.any(np.isnan(X[:, k + 1])) or np.any(np.isinf(X[:, k + 1])):
            return True, X
    
    return False, X


def solve_opinf_operators(
    D: np.ndarray,
    Y: np.ndarray,
    alpha_lin: float,
    alpha_quad: float,
    r: int,
    include_constant: bool = False,
    alpha_cubic: float = 0.0,
    include_cubic: bool = False,
) -> dict:
    """
    Solve the regularized OpInf least squares problem.
    
    Minimizes: ||D @ O.T - Y||_F^2 + regularization
    
    Parameters
    ----------
    D : np.ndarray
        Data matrix of shape (K, d).
    Y : np.ndarray
        Target matrix of shape (K, r) for state or (K, n_out) for output.
    alpha_lin : float
        Regularization for linear terms.
    alpha_quad : float
        Regularization for quadratic terms.
    r : int
        Number of POD modes.
    include_constant : bool
        Whether the data matrix includes a constant column.
    alpha_cubic : float
        Regularization for cubic diagonal terms (closure).
    include_cubic : bool
        Whether the data matrix includes cubic diagonal columns.
    
    Returns
    -------
    dict
        Dictionary with operator matrices.
    """
    s = r * (r + 1) // 2
    # Column layout: [linear(r) | quadratic(s) | cubic(r if enabled) | constant(1 if enabled)]
    d = r + s
    if include_cubic:
        d += r
    if include_constant:
        d += 1
    
    # Build regularization matrix
    reg = np.zeros(d)
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
        col += 1
    
    # Solve normal equations with Tikhonov regularization
    DtD = D.T @ D + np.diag(reg)
    DtY = D.T @ Y
    O = np.linalg.solve(DtD, DtY).T
    
    # Extract operators
    col = 0
    result = {
        'A': O[:, col:col + r],
    }
    col += r
    result['F'] = O[:, col:col + s]
    col += s
    if include_cubic:
        result['H'] = O[:, col:col + r]
        col += r
    if include_constant:
        result['c'] = O[:, col]
        col += 1
    
    return result


def build_data_matrix(
    X: np.ndarray,
    include_constant: bool = False,
    include_cubic: bool = False,
) -> np.ndarray:
    """
    Build the OpInf data matrix from reduced coordinates.
    
    Column layout: [linear | quadratic | cubic_diagonal (optional) | constant (optional)]
    
    Parameters
    ----------
    X : np.ndarray
        Reduced coordinates of shape (K, r).
    include_constant : bool
        Whether to append a column of ones.
    include_cubic : bool
        Whether to include cubic diagonal terms (z_i^3).
    
    Returns
    -------
    np.ndarray
        Data matrix of shape (K, d).
    """
    parts = [X, get_quadratic_terms(X)]
    
    if include_cubic:
        parts.append(get_cubic_diagonal_terms(X))
    
    if include_constant:
        K = X.shape[0]
        parts.append(np.ones((K, 1)))
    
    return np.concatenate(parts, axis=1)
