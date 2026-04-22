"""
Tests for Experiment 5: Empirical SPX Validation.

Validates:
- SPX data loading and scaling
- Black model price conversion
- Density recovery with Q=70
- IV smile reconstruction
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.option_pricing import black_call, black_put
from src.implied_vol import black_implied_vol
from src.exp5_spx_empirical import (
    SPX_DATA,
    SPX_SPOT,
    SPX_TAU,
    SPX_R,
    SPX_F,
    SCALE,
    X_MIN,
    X_MAX,
    N,
    Q,
    load_spx_data,
    convert_ivols_to_prices,
)


# ---------------------------------------------------------------------------
# Tests for SPX data loading
# ---------------------------------------------------------------------------

def test_spx_data_count():
    """Test that SPX data has the expected number of strikes."""
    assert len(SPX_DATA) == 101, f"Expected 101 strikes, got {len(SPX_DATA)}"


def test_spx_data_strike_range():
    """Test that SPX strikes are in [1900, 2900]."""
    strikes = [row[0] for row in SPX_DATA]
    assert min(strikes) == 1900
    assert max(strikes) == 2900


def test_spx_data_ivols_positive():
    """Test that all implied volatilities are positive."""
    ivols = [row[1] for row in SPX_DATA]
    assert all(iv > 0 for iv in ivols), "All IVs should be positive"


def test_spx_data_ivols_reasonable():
    """Test that implied volatilities are in a reasonable range [0.1, 1.0]."""
    ivols = [row[1] for row in SPX_DATA]
    assert all(0.1 <= iv <= 1.0 for iv in ivols), (
        f"IVs should be in [0.1, 1.0], got range [{min(ivols):.3f}, {max(ivols):.3f}]"
    )


def test_spx_scaling():
    """Test that scaling divides strikes and forward by 1000."""
    strikes_scaled, ivols, F_scaled, r, tau = load_spx_data()
    assert abs(F_scaled - SPX_F / SCALE) < 1e-10
    assert abs(strikes_scaled[0] - 1900 / SCALE) < 1e-10
    assert abs(strikes_scaled[-1] - 2900 / SCALE) < 1e-10


def test_spx_forward_reasonable():
    """Test that SPX forward price is reasonable."""
    _, _, F_scaled, r, tau = load_spx_data()
    F_unscaled = F_scaled * SCALE
    # Forward should be close to spot
    assert abs(F_unscaled - SPX_SPOT) / SPX_SPOT < 0.05, (
        f"Forward {F_unscaled:.2f} too far from spot {SPX_SPOT:.2f}"
    )


def test_price_conversion_non_negative():
    """Test that converted prices are non-negative."""
    strikes_scaled, ivols, F_scaled, r, tau = load_spx_data()
    call_prices, put_prices, Pr = convert_ivols_to_prices(
        strikes_scaled, ivols, F_scaled, r, tau
    )
    assert np.all(call_prices >= -1e-10), "Call prices should be non-negative"
    assert np.all(put_prices >= -1e-10), "Put prices should be non-negative"


def test_price_conversion_put_call_parity():
    """Test that converted prices satisfy put-call parity."""
    strikes_scaled, ivols, F_scaled, r, tau = load_spx_data()
    call_prices, put_prices, Pr = convert_ivols_to_prices(
        strikes_scaled, ivols, F_scaled, r, tau
    )
    discount = np.exp(-r * tau)
    parity = call_prices - put_prices - discount * (F_scaled - strikes_scaled)
    assert np.allclose(parity, 0.0, atol=1e-10), (
        f"Put-call parity violated: max|err|={np.abs(parity).max():.2e}"
    )


def test_price_conversion_roundtrip_iv():
    """Test that converting IV to price and back gives original IV."""
    strikes_scaled, ivols, F_scaled, r, tau = load_spx_data()
    call_prices, put_prices, Pr = convert_ivols_to_prices(
        strikes_scaled, ivols, F_scaled, r, tau
    )

    # Invert a few prices back to IV
    n_test = min(10, len(strikes_scaled))
    for i in range(n_test):
        K = strikes_scaled[i]
        price = call_prices[i]
        iv_inv = black_implied_vol(price, F_scaled, K, r, tau, "call")
        if iv_inv is not None:
            assert abs(iv_inv - ivols[i]) < 1e-6, (
                f"IV roundtrip failed at K={K:.3f}: "
                f"original={ivols[i]:.4f}, inverted={iv_inv:.4f}"
            )


# ---------------------------------------------------------------------------
# Tests for density grid and kernel
# ---------------------------------------------------------------------------

def test_density_grid_covers_strikes():
    """Test that density grid covers the strike range."""
    from src.kernel_matrix import build_density_grid
    x, dx = build_density_grid(X_MIN, X_MAX, N)
    strikes_scaled, _, _, _, _ = load_spx_data()

    # All strikes should be within the density support
    assert np.all(strikes_scaled >= X_MIN), (
        f"Some strikes below x_min={X_MIN}: min={strikes_scaled.min():.3f}"
    )
    assert np.all(strikes_scaled <= X_MAX), (
        f"Some strikes above x_max={X_MAX}: max={strikes_scaled.max():.3f}"
    )


def test_kernel_matrix_shape_exp5():
    """Test kernel matrix shape for Exp5 parameters."""
    from src.kernel_matrix import build_density_grid, build_call_put_kernel
    strikes_scaled, ivols, F_scaled, r, tau = load_spx_data()
    x, dx = build_density_grid(X_MIN, X_MAX, N)
    G = build_call_put_kernel(x, dx, strikes_scaled, r, tau)
    M = len(strikes_scaled)
    assert G.shape == (2 * M, N), f"Expected shape ({2*M}, {N}), got {G.shape}"


def test_svd_rank_exp5():
    """Test that SVD rank is sufficient for Q=70 truncation."""
    from src.kernel_matrix import build_density_grid, build_call_put_kernel
    from src.svd_utils import compute_svd
    strikes_scaled, ivols, F_scaled, r, tau = load_spx_data()
    x, dx = build_density_grid(X_MIN, X_MAX, N)
    G = build_call_put_kernel(x, dx, strikes_scaled, r, tau)
    _, s, _ = compute_svd(G, full_matrices=False)
    assert len(s) >= Q, f"SVD rank {len(s)} < Q={Q}"


# ---------------------------------------------------------------------------
# Tests for Black model
# ---------------------------------------------------------------------------

def test_black_call_atm():
    """Test Black model ATM call price."""
    F, K, sigma, r, tau = 1.0, 1.0, 0.2, 0.05, 1.0
    price = float(black_call(F, K, sigma, r, tau))
    # ATM call should be approximately F * sigma * sqrt(tau) * discount / sqrt(2*pi)
    discount = np.exp(-r * tau)
    approx = discount * F * sigma * np.sqrt(tau) / np.sqrt(2 * np.pi)
    assert abs(price - approx) / approx < 0.1, (
        f"ATM call price {price:.4f} far from approximation {approx:.4f}"
    )


def test_black_put_call_parity():
    """Test Black model put-call parity."""
    F, r, tau = 2.6, 0.015, 0.107
    sigma = 0.30
    K = np.linspace(1.9, 2.9, 20)
    C = black_call(F, K, sigma, r, tau)
    P = black_put(F, K, sigma, r, tau)
    discount = np.exp(-r * tau)
    parity = C - P - discount * (F - K)
    assert np.allclose(parity, 0.0, atol=1e-12)


def test_black_iv_spx_range():
    """Test IV inversion for SPX-like parameters."""
    F_scaled = SPX_F / SCALE
    r, tau = SPX_R, SPX_TAU
    sigma_test = 0.30
    K_scaled = F_scaled  # ATM

    price = float(black_call(F_scaled, K_scaled, sigma_test, r, tau))
    sigma_inv = black_implied_vol(price, F_scaled, K_scaled, r, tau, "call")
    assert sigma_inv is not None, "IV inversion failed for SPX-like parameters"
    assert abs(sigma_inv - sigma_test) < 1e-6
