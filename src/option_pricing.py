"""
Analytic option pricing formulas.

Implements Bachelier (normal), Black-Scholes (log-normal), and Black model
pricing for calls and puts. Used to generate synthetic option prices for
density recovery experiments.

Custom implementation — Context7 found no library equivalent for Bachelier model.
"""

import numpy as np
from scipy.stats import norm
from typing import Union


# ---------------------------------------------------------------------------
# Bachelier (Normal) Model
# ---------------------------------------------------------------------------

def bachelier_call(
    S0: float,
    K: Union[float, np.ndarray],
    sigma: float,
    r: float,
    tau: float,
) -> Union[float, np.ndarray]:
    """
    Bachelier (normal model) call price.

    Under the Bachelier model, S_T ~ N(S0 * exp(r*tau), sigma^2 * tau).
    The forward is F = S0 * exp(r*tau).

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float or array
        Strike(s).
    sigma : float
        Normal volatility.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.

    Returns
    -------
    price : float or array
        Call price(s).
    """
    F = S0 * np.exp(r * tau)
    discount = np.exp(-r * tau)
    sigma_T = sigma * np.sqrt(tau)
    d = (F - K) / sigma_T
    price = discount * ((F - K) * norm.cdf(d) + sigma_T * norm.pdf(d))
    return price


def bachelier_put(
    S0: float,
    K: Union[float, np.ndarray],
    sigma: float,
    r: float,
    tau: float,
) -> Union[float, np.ndarray]:
    """
    Bachelier (normal model) put price via put-call parity.

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float or array
        Strike(s).
    sigma : float
        Normal volatility.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.

    Returns
    -------
    price : float or array
        Put price(s).
    """
    call = bachelier_call(S0, K, sigma, r, tau)
    discount = np.exp(-r * tau)
    F = S0 * np.exp(r * tau)
    # Put-call parity: P = C - discount*(F - K)
    put = call - discount * (F - K)
    return put


def bachelier_density(
    x: np.ndarray,
    S0: float,
    sigma: float,
    r: float,
    tau: float,
) -> np.ndarray:
    """
    Terminal density under the Bachelier model: N(F, sigma^2 * tau).

    Parameters
    ----------
    x : np.ndarray
        Terminal price grid.
    S0 : float
        Spot price.
    sigma : float
        Normal volatility.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.

    Returns
    -------
    phi : np.ndarray
        Density values at grid points.
    """
    F = S0 * np.exp(r * tau)
    sigma_T = sigma * np.sqrt(tau)
    return norm.pdf(x, loc=F, scale=sigma_T)


# ---------------------------------------------------------------------------
# Black-Scholes (Log-Normal) Model
# ---------------------------------------------------------------------------

def black_scholes_call(
    S0: float,
    K: Union[float, np.ndarray],
    sigma: float,
    r: float,
    tau: float,
) -> Union[float, np.ndarray]:
    """
    Black-Scholes call price.

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float or array
        Strike(s).
    sigma : float
        Log-normal volatility.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.

    Returns
    -------
    price : float or array
        Call price(s).
    """
    K = np.asarray(K, dtype=np.float64)
    discount = np.exp(-r * tau)
    F = S0 * np.exp(r * tau)
    sigma_T = sigma * np.sqrt(tau)

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.where(K > 0, (np.log(F / K) + 0.5 * sigma_T**2) / sigma_T, np.inf)
        d2 = d1 - sigma_T
        price = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
        price = np.where(K <= 0, discount * (F - K), price)
    return price


def black_scholes_put(
    S0: float,
    K: Union[float, np.ndarray],
    sigma: float,
    r: float,
    tau: float,
) -> Union[float, np.ndarray]:
    """
    Black-Scholes put price via put-call parity.

    Parameters
    ----------
    S0 : float
        Spot price.
    K : float or array
        Strike(s).
    sigma : float
        Log-normal volatility.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.

    Returns
    -------
    price : float or array
        Put price(s).
    """
    call = black_scholes_call(S0, K, sigma, r, tau)
    discount = np.exp(-r * tau)
    F = S0 * np.exp(r * tau)
    put = call - discount * (F - K)
    return put


def black_scholes_density(
    x: np.ndarray,
    S0: float,
    sigma: float,
    r: float,
    tau: float,
) -> np.ndarray:
    """
    Risk-neutral terminal density under Black-Scholes: log-normal.

    S_T ~ LogNormal(log(S0) + (r - 0.5*sigma^2)*tau, sigma^2*tau).

    Parameters
    ----------
    x : np.ndarray
        Terminal price grid (must be positive).
    S0 : float
        Spot price.
    sigma : float
        Log-normal volatility.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.

    Returns
    -------
    phi : np.ndarray
        Density values at grid points.
    """
    mu_log = np.log(S0) + (r - 0.5 * sigma**2) * tau
    sigma_log = sigma * np.sqrt(tau)
    phi = np.where(
        x > 0,
        norm.pdf(np.log(np.maximum(x, 1e-300)), loc=mu_log, scale=sigma_log) / np.maximum(x, 1e-300),
        0.0,
    )
    return phi


# ---------------------------------------------------------------------------
# Black Model (for forward-based pricing, used in Exp5)
# ---------------------------------------------------------------------------

def black_call(
    F: float,
    K: Union[float, np.ndarray],
    sigma: float,
    r: float,
    tau: float,
) -> Union[float, np.ndarray]:
    """
    Black model call price (forward-based).

    Parameters
    ----------
    F : float
        Forward price.
    K : float or array
        Strike(s).
    sigma : float
        Log-normal (Black) volatility.
    r : float
        Risk-free rate (for discounting).
    tau : float
        Time to maturity.

    Returns
    -------
    price : float or array
        Call price(s).
    """
    K = np.asarray(K, dtype=np.float64)
    discount = np.exp(-r * tau)
    sigma_T = sigma * np.sqrt(tau)

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.where(K > 0, (np.log(F / K) + 0.5 * sigma_T**2) / sigma_T, np.inf)
        d2 = d1 - sigma_T
        price = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
        price = np.where(K <= 0, discount * (F - K), price)
    return price


def black_put(
    F: float,
    K: Union[float, np.ndarray],
    sigma: float,
    r: float,
    tau: float,
) -> Union[float, np.ndarray]:
    """
    Black model put price via put-call parity.

    Parameters
    ----------
    F : float
        Forward price.
    K : float or array
        Strike(s).
    sigma : float
        Black volatility.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.

    Returns
    -------
    price : float or array
        Put price(s).
    """
    call = black_call(F, K, sigma, r, tau)
    discount = np.exp(-r * tau)
    put = call - discount * (F - K)
    return put


# ---------------------------------------------------------------------------
# Gaussian Mixture Density
# ---------------------------------------------------------------------------

def gaussian_mixture_density(
    x: np.ndarray,
    weights: list[float],
    means: list[float],
    stds: list[float],
) -> np.ndarray:
    """
    Evaluate a Gaussian mixture density at grid points x.

    phi(x) = sum_i c_i * N(x; mu_i, sigma_i)

    Note: weights may be negative (signed mixture for arbitrage experiments).

    Parameters
    ----------
    x : np.ndarray
        Evaluation points.
    weights : list of float
        Mixture weights c_i (may not sum to 1 or be non-negative).
    means : list of float
        Component means mu_i.
    stds : list of float
        Component standard deviations sigma_i.

    Returns
    -------
    phi : np.ndarray
        Density values (may be negative for signed mixtures).
    """
    phi = np.zeros_like(x, dtype=np.float64)
    for c, mu, sigma in zip(weights, means, stds):
        phi += c * norm.pdf(x, loc=mu, scale=sigma)
    return phi


def mixture_call_price(
    K: Union[float, np.ndarray],
    weights: list[float],
    means: list[float],
    stds: list[float],
    r: float,
    tau: float,
) -> Union[float, np.ndarray]:
    """
    Compute call prices from a Gaussian mixture by linearity.

    Each component contributes a Bachelier-style call with S0=mu_i, sigma=sigma_i.
    This works because the mixture is a sum of normal densities.

    Parameters
    ----------
    K : float or array
        Strike(s).
    weights : list of float
        Mixture weights.
    means : list of float
        Component means.
    stds : list of float
        Component standard deviations.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.

    Returns
    -------
    price : float or array
        Call price(s).
    """
    K = np.asarray(K, dtype=np.float64)
    price = np.zeros_like(K, dtype=np.float64)
    discount = np.exp(-r * tau)
    for c, mu, sigma in zip(weights, means, stds):
        # Component i contributes: c * E[max(0, X-K)] where X ~ N(mu, sigma^2)
        # = c * discount * ((mu - K)*N(d) + sigma*n(d)) where d=(mu-K)/sigma
        d = (mu - K) / sigma
        component = discount * ((mu - K) * norm.cdf(d) + sigma * norm.pdf(d))
        price += c * component
    return price


def mixture_put_price(
    K: Union[float, np.ndarray],
    weights: list[float],
    means: list[float],
    stds: list[float],
    r: float,
    tau: float,
) -> Union[float, np.ndarray]:
    """
    Compute put prices from a Gaussian mixture by linearity.

    Parameters
    ----------
    K : float or array
        Strike(s).
    weights : list of float
        Mixture weights.
    means : list of float
        Component means.
    stds : list of float
        Component standard deviations.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.

    Returns
    -------
    price : float or array
        Put price(s).
    """
    K = np.asarray(K, dtype=np.float64)
    price = np.zeros_like(K, dtype=np.float64)
    discount = np.exp(-r * tau)
    for c, mu, sigma in zip(weights, means, stds):
        # Component i contributes: c * E[max(0, K-X)] where X ~ N(mu, sigma^2)
        # = c * discount * ((K - mu)*N(-d) + sigma*n(d)) where d=(mu-K)/sigma
        d = (mu - K) / sigma
        component = discount * ((K - mu) * norm.cdf(-d) + sigma * norm.pdf(d))
        price += c * component
    return price
