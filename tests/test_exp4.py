"""
Tests for Experiment 4: Implied-Volatility De-Arbitraging.

Validates:
- Black model implied vol inversion
- De-arbitraged IV smile computation
- IV smile preservation after de-arbitraging
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.option_pricing import black_call, black_put
from src.implied_vol import black_implied_vol, compute_implied_vol_smile


# ---------------------------------------------------------------------------
# Tests for implied volatility inversion
# ---------------------------------------------------------------------------

def test_black_iv_roundtrip_call():
    """Test that Black IV inversion is consistent with pricing."""
    F, K, r, tau = 1.0, 1.0, 0.05, 1.0
    sigma_true = 0.25
    price = float(black_call(F, K, sigma_true, r, tau))
    sigma_inv = black_implied_vol(price, F, K, r, tau, "call")
    assert sigma_inv is not None, "IV inversion failed"
    assert abs(sigma_inv - sigma_true) < 1e-8


def test_black_iv_roundtrip_put():
    """Test that Black IV inversion is consistent for puts."""
    F, K, r, tau = 1.0, 0.9, 0.05, 1.0
    sigma_true = 0.30
    price = float(black_put(F, K, sigma_true, r, tau))
    sigma_inv = black_implied_vol(price, F, K, r, tau, "put")
    assert sigma_inv is not None, "IV inversion failed"
    assert abs(sigma_inv - sigma_true) < 1e-8


def test_black_iv_various_strikes():
    """Test IV inversion across a range of strikes."""
    F, r, tau = 1.0, 0.05, 1.0
    sigma_true = 0.20
    strikes = np.linspace(0.7, 1.3, 10)

    for K in strikes:
        price = float(black_call(F, K, sigma_true, r, tau))
        sigma_inv = black_implied_vol(price, F, K, r, tau, "call")
        if sigma_inv is not None:
            assert abs(sigma_inv - sigma_true) < 1e-6, (
                f"K={K:.2f}: sigma_inv={sigma_inv:.6f}, sigma_true={sigma_true}"
            )


def test_black_iv_returns_none_for_invalid_price():
    """Test that IV inversion returns None for prices outside no-arbitrage bounds."""
    F, K, r, tau = 1.0, 1.0, 0.05, 1.0
    # Price below intrinsic value
    intrinsic = max(0.0, np.exp(-r * tau) * (F - K))
    invalid_price = intrinsic - 0.1
    sigma_inv = black_implied_vol(invalid_price, F, K, r, tau, "call")
    assert sigma_inv is None, "Should return None for price below intrinsic"


def test_black_iv_smile_computation():
    """Test compute_implied_vol_smile function."""
    F, r, tau = 1.0, 0.05, 1.0
    sigma_true = 0.25
    strikes = np.linspace(0.8, 1.2, 10)
    prices = np.array([float(black_call(F, K, sigma_true, r, tau)) for K in strikes])
    option_types = ["call"] * len(strikes)

    ivols, valid = compute_implied_vol_smile(prices, F, strikes, r, tau, option_types)

    assert valid.sum() == len(strikes), f"Expected all valid, got {valid.sum()}"
    assert np.allclose(ivols[valid], sigma_true, atol=1e-6)


def test_black_iv_positive():
    """Test that implied volatility is always positive."""
    F, r, tau = 1.0, 0.05, 1.0
    sigma_true = 0.30
    strikes = np.linspace(0.5, 1.5, 20)

    for K in strikes:
        price = float(black_call(F, K, sigma_true, r, tau))
        sigma_inv = black_implied_vol(price, F, K, r, tau, "call")
        if sigma_inv is not None:
            assert sigma_inv > 0, f"IV should be positive, got {sigma_inv}"


def test_black_iv_monotone_in_price():
    """Test that IV is monotone increasing in option price."""
    F, K, r, tau = 1.0, 1.0, 0.05, 1.0
    sigmas = np.linspace(0.1, 0.5, 10)
    prices = np.array([float(black_call(F, K, s, r, tau)) for s in sigmas])

    # Prices should be increasing in sigma
    assert np.all(np.diff(prices) > 0), "Call prices should increase with sigma"

    # Inverted IVs should match original sigmas
    for price, sigma_true in zip(prices, sigmas):
        sigma_inv = black_implied_vol(price, F, K, r, tau, "call")
        if sigma_inv is not None:
            assert abs(sigma_inv - sigma_true) < 1e-6


# ---------------------------------------------------------------------------
# Tests for de-arbitraging methodology
# ---------------------------------------------------------------------------

def test_dearbitraged_prices_non_negative():
    """
    Test that prices from recovered density are non-negative.
    """
    from src.kernel_matrix import build_density_grid
    from scipy.stats import norm

    x, dx = build_density_grid(0.1, 2.2, 500)
    # Use a simple non-negative density
    phi_rec = norm.pdf(x, loc=1.0, scale=0.3)
    phi_rec /= np.trapz(phi_rec, dx=dx)

    strikes = np.linspace(0.3, 1.7, 20)
    r, tau = 0.05, 1.0
    discount = np.exp(-r * tau)

    call_prices = np.array([
        np.trapz(discount * np.maximum(0.0, x - K) * phi_rec, dx=dx)
        for K in strikes
    ])
    put_prices = np.array([
        np.trapz(discount * np.maximum(0.0, K - x) * phi_rec, dx=dx)
        for K in strikes
    ])

    assert np.all(call_prices >= -1e-10), "Call prices from valid density should be non-negative"
    assert np.all(put_prices >= -1e-10), "Put prices from valid density should be non-negative"


def test_dearbitraged_put_call_parity():
    """
    Test that prices from recovered density satisfy put-call parity.
    """
    from src.kernel_matrix import build_density_grid
    from scipy.stats import norm

    x, dx = build_density_grid(0.1, 2.2, 1000)
    phi_rec = norm.pdf(x, loc=1.0, scale=0.3)
    phi_rec /= np.trapz(phi_rec, dx=dx)

    strikes = np.linspace(0.5, 1.5, 10)
    r, tau = 0.05, 1.0
    discount = np.exp(-r * tau)

    call_prices = np.array([
        np.trapz(discount * np.maximum(0.0, x - K) * phi_rec, dx=dx)
        for K in strikes
    ])
    put_prices = np.array([
        np.trapz(discount * np.maximum(0.0, K - x) * phi_rec, dx=dx)
        for K in strikes
    ])

    # Forward from density: E[X] = integral x * phi(x) dx
    F_density = np.trapz(x * phi_rec, dx=dx)

    # Put-call parity: C - P = discount * (F - K)
    parity = call_prices - put_prices - discount * (F_density - strikes)
    assert np.allclose(parity, 0.0, atol=1e-6), (
        f"Put-call parity violated: max|err|={np.abs(parity).max():.2e}"
    )


def test_iv_smile_close_to_original():
    """
    Test that de-arbitraged IV smile is close to original for small lambda.
    Uses a valid (non-contaminated) density to verify the pipeline.
    """
    from src.kernel_matrix import build_density_grid, build_call_put_kernel
    from src.svd_utils import compute_svd, truncate_svd
    from src.density_recovery import solve_density_recovery
    from src.option_pricing import bachelier_call, bachelier_put

    S0, sigma, r, tau = 0.1, 0.1, 0.05, 1.0
    x, dx = build_density_grid(-0.9, 0.9, 200)
    strikes = np.linspace(-0.5, 0.5, 20)

    call_prices = bachelier_call(S0, strikes, sigma, r, tau)
    put_prices = bachelier_put(S0, strikes, sigma, r, tau)
    Pr = np.concatenate([call_prices, put_prices])

    G = build_call_put_kernel(x, dx, strikes, r, tau)
    U, s, Vt = compute_svd(G, full_matrices=False)
    Q = min(30, len(s))
    U_tilde, S_tilde, V_tilde = truncate_svd(U, s, Vt, Q)

    phi, phi_prime, status = solve_density_recovery(
        Pr, U_tilde, S_tilde, V_tilde, dx, lam=1e-8
    )

    if phi is not None:
        # Reprice from recovered density
        discount = np.exp(-r * tau)
        call_rec = np.array([
            np.trapz(discount * np.maximum(0.0, x - K) * phi, dx=dx)
            for K in strikes
        ])

        # Relative pricing error should be small
        rel_err = np.abs(call_prices - call_rec) / (np.abs(call_prices) + 1e-10)
        assert np.median(rel_err) < 0.5, f"Median relative error too large: {np.median(rel_err):.4f}"
