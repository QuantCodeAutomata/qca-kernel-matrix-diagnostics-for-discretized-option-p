"""
SVD utilities for the pricing kernel matrix.

Provides truncated SVD, condition number computation, and singular-value
diagnostics used in the kernel matrix experiments.

Using scipy.linalg.svd — Context7 confirmed.
"""

import numpy as np
from scipy.linalg import svd as scipy_svd
from typing import Optional


def compute_svd(
    G: np.ndarray,
    full_matrices: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the SVD of G = U S V^T with singular values in descending order.

    Parameters
    ----------
    G : np.ndarray, shape (M, N)
        Input matrix.
    full_matrices : bool
        If True, return full U and V^T matrices.

    Returns
    -------
    U : np.ndarray, shape (M, K) or (M, M)
        Left singular vectors.
    s : np.ndarray, shape (K,)
        Singular values in descending order.
    Vt : np.ndarray, shape (K, N) or (N, N)
        Right singular vectors (transposed).
    """
    # Using scipy.linalg.svd — Context7 confirmed
    U, s, Vt = scipy_svd(G, full_matrices=full_matrices, check_finite=False)
    # scipy guarantees descending order, but verify
    assert np.all(np.diff(s) <= 1e-12), "Singular values not in descending order"
    return U, s, Vt


def truncate_svd(
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    Q: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Truncate SVD to the first Q components.

    Parameters
    ----------
    U : np.ndarray, shape (M, K)
        Left singular vectors.
    s : np.ndarray, shape (K,)
        Singular values.
    Vt : np.ndarray, shape (K, N)
        Right singular vectors (transposed).
    Q : int
        Number of components to retain.

    Returns
    -------
    U_tilde : np.ndarray, shape (M, Q)
    S_tilde : np.ndarray, shape (Q,)
    V_tilde : np.ndarray, shape (N, Q)
        Note: V_tilde = Vt[:Q, :].T for use as phi = V_tilde @ phi_prime.
    """
    assert Q <= len(s), f"Q={Q} exceeds available rank {len(s)}"
    U_tilde = U[:, :Q]
    S_tilde = s[:Q]
    V_tilde = Vt[:Q, :].T  # shape (N, Q)
    return U_tilde, S_tilde, V_tilde


def condition_number(s: np.ndarray, tol: float = 1e-15) -> float:
    """
    Compute the condition number as s_max / s_min over nonzero singular values.

    Parameters
    ----------
    s : np.ndarray
        Singular values in descending order.
    tol : float
        Threshold below which singular values are considered zero.

    Returns
    -------
    cond : float
        Condition number.
    """
    nonzero = s[s > tol * s[0]]
    if len(nonzero) < 2:
        return np.inf
    return nonzero[0] / nonzero[-1]


def truncated_condition_number(s: np.ndarray, Q: int) -> float:
    """
    Compute the truncated condition number C_tilde = s_1 / s_Q.

    Parameters
    ----------
    s : np.ndarray
        Singular values in descending order.
    Q : int
        Truncation rank.

    Returns
    -------
    cond_tilde : float
        Truncated condition number.
    """
    assert Q <= len(s), f"Q={Q} exceeds available rank {len(s)}"
    return s[0] / s[Q - 1]


def fit_power_law(
    x: np.ndarray,
    y: np.ndarray,
    fit_range: Optional[tuple[int, int]] = None,
) -> tuple[float, float]:
    """
    Fit a power law y ~ a * x^k in log-log space via linear regression.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (positive).
    y : np.ndarray
        Dependent variable (positive).
    fit_range : tuple (i_start, i_end), optional
        Index range [i_start, i_end) to use for fitting. If None, use all.

    Returns
    -------
    k : float
        Power-law exponent.
    log_a : float
        Log of the prefactor (log(a)).
    """
    if fit_range is not None:
        i0, i1 = fit_range
        x_fit = x[i0:i1]
        y_fit = y[i0:i1]
    else:
        x_fit = x
        y_fit = y

    log_x = np.log(x_fit)
    log_y = np.log(y_fit)

    # Linear regression: log_y = k * log_x + log_a
    coeffs = np.polyfit(log_x, log_y, 1)
    k = coeffs[0]
    log_a = coeffs[1]
    return k, log_a
