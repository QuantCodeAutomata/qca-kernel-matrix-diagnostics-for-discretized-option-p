"""
Implied volatility inversion via numerical root-finding.

Inverts Black/Black-Scholes option prices to implied volatilities using
Brent's method from scipy.optimize.

Using scipy.optimize.brentq — Context7 confirmed.
"""

import numpy as np
from scipy.optimize import brentq
from typing import Optional
from src.option_pricing import black_call, black_put


def black_implied_vol(
    price: float,
    F: float,
    K: float,
    r: float,
    tau: float,
    option_type: str = "call",
    vol_lo: float = 1e-8,
    vol_hi: float = 20.0,
) -> Optional[float]:
    """
    Invert a Black model price to implied volatility using Brent's method.

    Parameters
    ----------
    price : float
        Observed option price.
    F : float
        Forward price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.
    option_type : str
        'call' or 'put'.
    vol_lo : float
        Lower bracket for volatility search.
    vol_hi : float
        Upper bracket for volatility search.

    Returns
    -------
    iv : float or None
        Implied volatility, or None if inversion fails.
    """
    discount = np.exp(-r * tau)

    # Intrinsic value bounds
    if option_type == "call":
        intrinsic = max(0.0, discount * (F - K))
        price_hi = discount * F  # upper bound: F * discount
    else:
        intrinsic = max(0.0, discount * (K - F))
        price_hi = discount * K

    # Check if price is within no-arbitrage bounds
    if price < intrinsic - 1e-10 or price > price_hi + 1e-10:
        return None

    # Clamp price to valid range
    price_clamped = np.clip(price, intrinsic + 1e-12, price_hi - 1e-12)

    def objective(sigma: float) -> float:
        if option_type == "call":
            model_price = black_call(F, K, sigma, r, tau)
        else:
            model_price = black_put(F, K, sigma, r, tau)
        return float(model_price) - price_clamped

    try:
        f_lo = objective(vol_lo)
        f_hi = objective(vol_hi)
        if f_lo * f_hi > 0:
            return None
        iv = brentq(objective, vol_lo, vol_hi, xtol=1e-10, rtol=1e-10, maxiter=200)
        return iv
    except (ValueError, RuntimeError):
        return None


def compute_implied_vol_smile(
    prices: np.ndarray,
    F: float,
    strikes: np.ndarray,
    r: float,
    tau: float,
    option_types: list[str],
    vol_lo: float = 1e-8,
    vol_hi: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute implied volatility smile for a cross-section of options.

    Parameters
    ----------
    prices : np.ndarray, shape (M,)
        Option prices.
    F : float
        Forward price.
    strikes : np.ndarray, shape (M,)
        Strike prices.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.
    option_types : list of str
        Option type for each observation.
    vol_lo : float
        Lower volatility bracket.
    vol_hi : float
        Upper volatility bracket.

    Returns
    -------
    ivols : np.ndarray, shape (M,)
        Implied volatilities (NaN where inversion failed).
    valid_mask : np.ndarray of bool, shape (M,)
        True where inversion succeeded.
    """
    M = len(prices)
    ivols = np.full(M, np.nan)
    valid_mask = np.zeros(M, dtype=bool)

    for i in range(M):
        iv = black_implied_vol(
            prices[i], F, strikes[i], r, tau, option_types[i], vol_lo, vol_hi
        )
        if iv is not None:
            ivols[i] = iv
            valid_mask[i] = True

    return ivols, valid_mask
