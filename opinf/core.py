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
        
        if np.any(np.isnan(X[:, k + 1])):
            return True, X
    
    return False, X


def solve_opinf_operators(
    D: np.ndarray,
    Y: np.ndarray,
    alpha_lin: float,
    alpha_quad: float,
    r: int,
    include_constant: bool = False,
) -> dict:
    """
    Solve the regularized OpInf least squares problem.
    
    Minimizes: ||D @ O.T - Y||_F^2 + regularization
    
    Parameters
    ----------
    D : np.ndarray
        Data matrix of shape (K, d) where d = r + s (+ 1 if constant).
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
    
    Returns
    -------
    dict
        Dictionary with operator matrices.
    """
    s = r * (r + 1) // 2
    d = r + s + (1 if include_constant else 0)
    
    # Build regularization matrix
    reg = np.zeros(d)
    reg[:r] = alpha_lin
    reg[r:r + s] = alpha_quad
    if include_constant:
        reg[r + s:] = alpha_lin
    
    # Solve normal equations with Tikhonov regularization
    DtD = D.T @ D + np.diag(reg)
    DtY = D.T @ Y
    O = np.linalg.solve(DtD, DtY).T
    
    # Extract operators
    result = {
        'A': O[:, :r],
        'F': O[:, r:r + s],
    }
    
    if include_constant:
        result['c'] = O[:, r + s]
    
    return result


def build_data_matrix(
    X: np.ndarray,
    include_constant: bool = False,
) -> np.ndarray:
    """
    Build the OpInf data matrix from reduced coordinates.
    
    Parameters
    ----------
    X : np.ndarray
        Reduced coordinates of shape (K, r).
    include_constant : bool
        Whether to append a column of ones.
    
    Returns
    -------
    np.ndarray
        Data matrix of shape (K, d) where d = r + s (+ 1 if constant).
    """
    X2 = get_quadratic_terms(X)
    
    if include_constant:
        K = X.shape[0]
        return np.concatenate([X, X2, np.ones((K, 1))], axis=1)
    else:
        return np.concatenate([X, X2], axis=1)
