"""
Pricing kernel matrix construction for discretized option-pricing operator.

Constructs the matrix G that maps a terminal density phi to option prices Pr,
using trapezoidal quadrature on a uniform terminal-price grid.

Custom implementation — Context7 found no library equivalent (paper methodology).
"""

import numpy as np
from typing import Literal


def build_density_grid(x_min: float, x_max: float, N: int) -> tuple[np.ndarray, float]:
    """
    Build a uniform terminal-price grid with N points over [x_min, x_max].

    Parameters
    ----------
    x_min : float
        Left endpoint of the density support.
    x_max : float
        Right endpoint of the density support.
    N : int
        Number of grid points.

    Returns
    -------
    x : np.ndarray, shape (N,)
        Uniform grid points.
    dx : float
        Grid spacing.
    """
    assert N >= 2, "Need at least 2 grid points"
    assert x_max > x_min, "x_max must be greater than x_min"
    x = np.linspace(x_min, x_max, N, dtype=np.float64)
    dx = (x_max - x_min) / (N - 1)
    return x, dx


def build_kernel_row(
    x: np.ndarray,
    dx: float,
    K: float,
    r: float,
    tau: float,
    option_type: Literal["call", "put"],
) -> np.ndarray:
    """
    Build a single row of the pricing kernel matrix G for one option.

    Row i: (1/2 g_i(x_1), g_i(x_2), ..., g_i(x_{N-1}), 1/2 g_i(x_N))
    where g_i(x) = dx * exp(-r*tau) * payoff(x, K).

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Terminal-price grid.
    dx : float
        Grid spacing.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.
    option_type : str
        'call' or 'put'.

    Returns
    -------
    row : np.ndarray, shape (N,)
        Kernel row with trapezoidal endpoint half-weights applied.
    """
    discount = np.exp(-r * tau)
    if option_type == "call":
        payoff = np.maximum(0.0, x - K)
    else:
        payoff = np.maximum(0.0, K - x)

    # g_i(x_j) = dx * discount * payoff(x_j, K)
    g = dx * discount * payoff

    # Apply trapezoidal endpoint half-weights
    row = g.copy()
    row[0] *= 0.5
    row[-1] *= 0.5
    return row


def build_kernel_matrix(
    x: np.ndarray,
    dx: float,
    strikes: np.ndarray,
    r: float,
    tau: float,
    option_types: list[str],
) -> np.ndarray:
    """
    Build the full pricing kernel matrix G.

    G has shape (M_total, N) where M_total is the total number of options
    (calls + puts) and N is the number of density grid points.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Terminal-price grid.
    dx : float
        Grid spacing.
    strikes : np.ndarray, shape (M,)
        Strike prices for each option.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.
    option_types : list of str
        Option type ('call' or 'put') for each row, length M.

    Returns
    -------
    G : np.ndarray, shape (M, N)
        Pricing kernel matrix.
    """
    assert len(strikes) == len(option_types), "strikes and option_types must match"
    N = len(x)
    M = len(strikes)
    G = np.zeros((M, N), dtype=np.float64)
    for i, (K, otype) in enumerate(zip(strikes, option_types)):
        G[i] = build_kernel_row(x, dx, K, r, tau, otype)
    return G


def build_call_put_kernel(
    x: np.ndarray,
    dx: float,
    strikes: np.ndarray,
    r: float,
    tau: float,
) -> np.ndarray:
    """
    Build kernel matrix with both calls and puts at each strike.

    Rows are ordered: [call_1, ..., call_M, put_1, ..., put_M].

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Terminal-price grid.
    dx : float
        Grid spacing.
    strikes : np.ndarray, shape (M,)
        Strike prices.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.

    Returns
    -------
    G : np.ndarray, shape (2*M, N)
        Pricing kernel matrix with calls then puts.
    """
    M = len(strikes)
    option_types = ["call"] * M + ["put"] * M
    all_strikes = np.concatenate([strikes, strikes])
    return build_kernel_matrix(x, dx, all_strikes, r, tau, option_types)


def price_from_density(
    G: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """
    Compute option prices from a density vector using the kernel matrix.

    Pr = G @ phi

    Parameters
    ----------
    G : np.ndarray, shape (M, N)
        Pricing kernel matrix.
    phi : np.ndarray, shape (N,)
        Terminal density (non-negative, integrates to 1 under trapezoidal rule).

    Returns
    -------
    prices : np.ndarray, shape (M,)
        Option prices.
    """
    return G @ phi
