"""
Density recovery via truncated-SVD and constrained L1-regularized optimization.

Implements the paper's core methodology:
  minimize  1/2 ||Pr - U_tilde S_tilde phi'||_2^2 + lambda ||phi'||_1
  subject to  V_tilde phi' >= 0  (non-negativity of density)
              trapz(V_tilde phi') = 1  (unit mass)

Using cvxpy for constrained convex optimization — Context7 confirmed.
Custom implementation of the full pipeline — Context7 found no library equivalent.
"""

import numpy as np
import cvxpy as cp
from typing import Optional
from src.svd_utils import compute_svd, truncate_svd


def solve_density_recovery(
    Pr: np.ndarray,
    U_tilde: np.ndarray,
    S_tilde: np.ndarray,
    V_tilde: np.ndarray,
    dx: float,
    lam: float,
    solver: str = "SCS",
    verbose: bool = False,
    solver_opts: Optional[dict] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Solve the constrained L1-regularized density recovery problem.

    Objective (paper Eq. 17-style):
        minimize  1/2 ||Pr - U_tilde @ diag(S_tilde) @ phi'||_2^2
                  + lambda * ||phi'||_1
        subject to  V_tilde @ phi' >= 0  (elementwise)
                    trapz(V_tilde @ phi') = 1  (trapezoidal mass = 1)

    Parameters
    ----------
    Pr : np.ndarray, shape (M,)
        Observed option prices.
    U_tilde : np.ndarray, shape (M, Q)
        Truncated left singular vectors.
    S_tilde : np.ndarray, shape (Q,)
        Truncated singular values.
    V_tilde : np.ndarray, shape (N, Q)
        Truncated right singular vectors.
    dx : float
        Grid spacing for trapezoidal integration.
    lam : float
        L1 regularization parameter lambda.
    solver : str
        CVXPY solver name.
    verbose : bool
        Print solver output.
    solver_opts : dict, optional
        Additional solver options.

    Returns
    -------
    phi : np.ndarray, shape (N,) or None
        Recovered density, or None if solver failed.
    phi_prime : np.ndarray, shape (Q,) or None
        Transformed coefficients, or None if solver failed.
    status : str
        Solver status string.
    """
    Q = len(S_tilde)
    N = V_tilde.shape[0]

    # Precompute A = U_tilde @ diag(S_tilde), shape (M, Q)
    A = U_tilde * S_tilde[np.newaxis, :]  # broadcasting

    # CVXPY variable: transformed coefficients phi'
    phi_prime = cp.Variable(Q)

    # Density: phi = V_tilde @ phi'
    phi_expr = V_tilde @ phi_prime  # shape (N,)

    # Trapezoidal weights for mass constraint
    trap_weights = np.ones(N, dtype=np.float64)
    trap_weights[0] = 0.5
    trap_weights[-1] = 0.5
    trap_weights *= dx

    # Objective: 1/2 ||Pr - A phi'||^2 + lambda ||phi'||_1
    residual = Pr - A @ phi_prime
    objective = cp.Minimize(
        0.5 * cp.sum_squares(residual) + lam * cp.norm1(phi_prime)
    )

    # Constraints
    constraints = [
        phi_expr >= 0,                          # non-negativity
        trap_weights @ phi_expr == 1.0,         # unit mass
    ]

    prob = cp.Problem(objective, constraints)

    opts = solver_opts or {}
    try:
        if solver == "SCS":
            opts.setdefault("max_iters", 10000)
            opts.setdefault("eps", 1e-6)
            prob.solve(solver=cp.SCS, verbose=verbose, **opts)
        elif solver == "CLARABEL":
            prob.solve(solver=cp.CLARABEL, verbose=verbose, **opts)
        elif solver == "ECOS":
            prob.solve(solver=cp.ECOS, verbose=verbose, **opts)
        else:
            prob.solve(solver=solver, verbose=verbose, **opts)
    except cp.SolverError as e:
        return None, None, f"SolverError: {e}"

    status = prob.status
    if phi_prime.value is None:
        return None, None, status

    phi_val = V_tilde @ phi_prime.value
    return phi_val, phi_prime.value, status


def run_lambda_scan(
    Pr: np.ndarray,
    U_tilde: np.ndarray,
    S_tilde: np.ndarray,
    V_tilde: np.ndarray,
    x: np.ndarray,
    dx: float,
    lambdas: np.ndarray,
    phi_true: Optional[np.ndarray] = None,
    solver: str = "SCS",
    verbose: bool = False,
) -> dict:
    """
    Run density recovery for a grid of lambda values and collect diagnostics.

    Parameters
    ----------
    Pr : np.ndarray, shape (M,)
        Observed option prices.
    U_tilde : np.ndarray, shape (M, Q)
        Truncated left singular vectors.
    S_tilde : np.ndarray, shape (Q,)
        Truncated singular values.
    V_tilde : np.ndarray, shape (N, Q)
        Truncated right singular vectors.
    x : np.ndarray, shape (N,)
        Density grid.
    dx : float
        Grid spacing.
    lambdas : np.ndarray
        Array of lambda values to scan.
    phi_true : np.ndarray, shape (N,), optional
        True density for Bhattacharyya distance computation.
    solver : str
        CVXPY solver.
    verbose : bool
        Print solver output.

    Returns
    -------
    results : dict
        Keys: 'lambdas', 'phi_list', 'phi_prime_list', 'chi2_list',
              'dB_list', 'status_list'.
    """
    phi_list = []
    phi_prime_list = []
    chi2_list = []
    dB_list = []
    status_list = []

    # Precompute A for repricing
    A = U_tilde * S_tilde[np.newaxis, :]

    for lam in lambdas:
        phi, phi_prime, status = solve_density_recovery(
            Pr, U_tilde, S_tilde, V_tilde, dx, lam, solver=solver, verbose=verbose
        )
        phi_list.append(phi)
        phi_prime_list.append(phi_prime)
        status_list.append(status)

        if phi is not None and phi_prime is not None:
            # Pricing error chi^2 (Eq. 15-style: sum of squared residuals)
            # Reprice using full kernel: Pr_hat = A @ phi_prime
            Pr_hat = A @ phi_prime
            chi2 = float(np.sum((Pr - Pr_hat) ** 2))
            chi2_list.append(chi2)

            # Bhattacharyya distance
            if phi_true is not None:
                dB = bhattacharyya_distance(phi_true, phi, dx)
            else:
                dB = np.nan
            dB_list.append(dB)
        else:
            chi2_list.append(np.nan)
            dB_list.append(np.nan)

    return {
        "lambdas": lambdas,
        "phi_list": phi_list,
        "phi_prime_list": phi_prime_list,
        "chi2_list": np.array(chi2_list),
        "dB_list": np.array(dB_list),
        "status_list": status_list,
    }


def bhattacharyya_distance(
    phi1: np.ndarray,
    phi2: np.ndarray,
    dx: float,
) -> float:
    """
    Compute the Bhattacharyya distance between two densities.

    d_B = -log(integral sqrt(phi1(x) * phi2(x)) dx)

    Uses trapezoidal quadrature.

    Parameters
    ----------
    phi1 : np.ndarray, shape (N,)
        First density (may not be normalized for arbitrage case).
    phi2 : np.ndarray, shape (N,)
        Second density.
    dx : float
        Grid spacing.

    Returns
    -------
    dB : float
        Bhattacharyya distance.
    """
    # Clamp to non-negative for sqrt
    integrand = np.sqrt(np.maximum(phi1, 0.0) * np.maximum(phi2, 0.0))
    # Trapezoidal integration
    bc = np.trapz(integrand, dx=dx)
    if bc <= 0:
        return np.inf
    return -np.log(bc)


def sparsity_count(
    phi_prime: np.ndarray,
    thresholds: list[float],
) -> dict[float, int]:
    """
    Count the number of transformed coefficients with |phi'| > threshold.

    Parameters
    ----------
    phi_prime : np.ndarray
        Transformed coefficients.
    thresholds : list of float
        Threshold values a.

    Returns
    -------
    counts : dict
        Maps threshold -> count of |phi'| > threshold.
    """
    return {a: int(np.sum(np.abs(phi_prime) > a)) for a in thresholds}


def compute_chi2_eq15(
    Pr: np.ndarray,
    Pr_hat: np.ndarray,
) -> float:
    """
    Compute pricing error chi^2 as sum of squared residuals (Eq. 15-style).

    chi^2 = sum_i (Pr_i - Pr_hat_i)^2

    Parameters
    ----------
    Pr : np.ndarray
        Observed prices.
    Pr_hat : np.ndarray
        Repriced values.

    Returns
    -------
    chi2 : float
        Pricing error.
    """
    return float(np.sum((Pr - Pr_hat) ** 2))
