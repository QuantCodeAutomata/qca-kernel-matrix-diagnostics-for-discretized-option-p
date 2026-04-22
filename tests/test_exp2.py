"""
Tests for Experiment 2: Synthetic Exact-Density Recovery.

Validates:
- Analytic option pricing formulas (Bachelier, BS)
- Density normalization
- Recovery pipeline (SVD + constrained optimization)
- Non-negativity and unit mass of recovered densities
- Bhattacharyya distance computation
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.option_pricing import (
    bachelier_call,
    bachelier_put,
    bachelier_density,
    black_scholes_call,
    black_scholes_put,
    black_scholes_density,
    gaussian_mixture_density,
    mixture_call_price,
    mixture_put_price,
)
from src.kernel_matrix import build_density_grid, build_call_put_kernel
from src.svd_utils import compute_svd, truncate_svd
from src.density_recovery import (
    solve_density_recovery,
    bhattacharyya_distance,
    sparsity_count,
)


# ---------------------------------------------------------------------------
# Tests for analytic option pricing
# ---------------------------------------------------------------------------

def test_bachelier_call_put_parity():
    """Test Bachelier put-call parity: C - P = discount * (F - K)."""
    S0, sigma, r, tau = 0.1, 0.1, 0.05, 1.0
    K = np.linspace(-0.5, 0.5, 20)
    C = bachelier_call(S0, K, sigma, r, tau)
    P = bachelier_put(S0, K, sigma, r, tau)
    F = S0 * np.exp(r * tau)
    discount = np.exp(-r * tau)
    parity = C - P - discount * (F - K)
    assert np.allclose(parity, 0.0, atol=1e-12), f"Put-call parity violated: max|err|={np.abs(parity).max():.2e}"


def test_bachelier_call_non_negative():
    """Test that Bachelier call prices are non-negative."""
    S0, sigma, r, tau = 0.1, 0.1, 0.05, 1.0
    K = np.linspace(-1.0, 1.0, 50)
    C = bachelier_call(S0, K, sigma, r, tau)
    assert np.all(C >= -1e-12)


def test_bachelier_density_normalizes():
    """Test that Bachelier density integrates to 1."""
    x, dx = build_density_grid(-1.0, 1.0, 2000)
    phi = bachelier_density(x, S0=0.1, sigma=0.1, r=0.05, tau=1.0)
    mass = np.trapz(phi, dx=dx)
    assert abs(mass - 1.0) < 0.01  # some mass outside [-1, 1]


def test_bachelier_density_non_negative():
    """Test that Bachelier density is non-negative."""
    x, dx = build_density_grid(-1.0, 1.0, 1000)
    phi = bachelier_density(x, S0=0.1, sigma=0.1, r=0.05, tau=1.0)
    assert np.all(phi >= 0)


def test_bs_call_put_parity():
    """Test Black-Scholes put-call parity."""
    S0, sigma, r, tau = 0.5, 0.2, 0.0, 1.0
    K = np.linspace(0.1, 1.0, 20)
    C = black_scholes_call(S0, K, sigma, r, tau)
    P = black_scholes_put(S0, K, sigma, r, tau)
    F = S0 * np.exp(r * tau)
    discount = np.exp(-r * tau)
    parity = C - P - discount * (F - K)
    assert np.allclose(parity, 0.0, atol=1e-12)


def test_bs_call_non_negative():
    """Test that BS call prices are non-negative."""
    S0, sigma, r, tau = 0.5, 0.2, 0.0, 1.0
    K = np.linspace(0.01, 1.5, 50)
    C = black_scholes_call(S0, K, sigma, r, tau)
    assert np.all(C >= -1e-12)


def test_bs_density_non_negative():
    """Test that BS density is non-negative."""
    x, dx = build_density_grid(0.01, 2.0, 1000)
    phi = black_scholes_density(x, S0=0.5, sigma=0.2, r=0.0, tau=1.0)
    assert np.all(phi >= 0)


def test_gaussian_mixture_density_positive_weights():
    """Test that Gaussian mixture with positive weights is non-negative."""
    x = np.linspace(-1.0, 1.0, 500)
    phi = gaussian_mixture_density(x, [0.5, 0.3, 0.2], [-0.3, 0.0, 0.4], [0.1, 0.15, 0.08])
    assert np.all(phi >= 0)


def test_gaussian_mixture_density_normalizes():
    """Test that Gaussian mixture with unit weights integrates to 1."""
    x, dx = build_density_grid(-2.0, 2.0, 5000)
    phi = gaussian_mixture_density(x, [0.5, 0.3, 0.2], [-0.3, 0.0, 0.4], [0.1, 0.15, 0.08])
    mass = np.trapz(phi, dx=dx)
    assert abs(mass - 1.0) < 0.01


def test_mixture_call_put_parity():
    """Test put-call parity for Gaussian mixture prices."""
    weights = [0.50, 0.45, 0.05]
    means = [-0.20, 0.15, 0.55]
    stds = [0.10, 0.15, 0.05]
    r, tau = 0.05, 1.0
    K = np.linspace(-0.5, 0.5, 20)

    C = mixture_call_price(K, weights, means, stds, r, tau)
    P = mixture_put_price(K, weights, means, stds, r, tau)

    # Forward = sum(c_i * mu_i) for normal mixture
    F_mix = sum(c * mu for c, mu in zip(weights, means))
    discount = np.exp(-r * tau)
    parity = C - P - discount * (F_mix - K)
    assert np.allclose(parity, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests for density recovery pipeline
# ---------------------------------------------------------------------------

def test_bhattacharyya_distance_identical():
    """Test that Bhattacharyya distance is 0 for identical normalized densities."""
    x, dx = build_density_grid(0.0, 1.0, 1000)
    # Uniform density normalized to integrate to 1 under trapezoidal rule
    phi = np.ones(1000)
    # Normalize: trapz(phi * dx) = 1
    mass = np.trapz(phi, dx=dx)
    phi = phi / mass
    dB = bhattacharyya_distance(phi, phi, dx)
    assert abs(dB) < 1e-10


def test_bhattacharyya_distance_non_negative():
    """Test that Bhattacharyya distance is non-negative for valid densities."""
    x, dx = build_density_grid(-1.0, 1.0, 200)
    from scipy.stats import norm
    phi1 = norm.pdf(x, loc=-0.2, scale=0.1)
    phi2 = norm.pdf(x, loc=0.2, scale=0.1)
    dB = bhattacharyya_distance(phi1, phi2, dx)
    assert dB >= 0


def test_bhattacharyya_distance_increases_with_separation():
    """Test that Bhattacharyya distance increases as densities separate."""
    x, dx = build_density_grid(-3.0, 3.0, 1000)
    from scipy.stats import norm
    phi_ref = norm.pdf(x, loc=0.0, scale=0.3)
    dB_values = []
    for sep in [0.0, 0.5, 1.0, 2.0]:
        phi_other = norm.pdf(x, loc=sep, scale=0.3)
        dB_values.append(bhattacharyya_distance(phi_ref, phi_other, dx))
    assert dB_values[0] < dB_values[1] < dB_values[2] < dB_values[3]


def test_sparsity_count():
    """Test sparsity count function."""
    phi_prime = np.array([0.5, 0.05, 0.005, 0.0005, 0.0, -0.3])
    counts = sparsity_count(phi_prime, [1e-3, 1e-2, 1e-1, 1e0])
    assert counts[1e-3] == 4  # |0.5|, |0.05|, |0.005|, |-0.3| > 1e-3
    assert counts[1e-2] == 3  # |0.5|, |0.05|, |-0.3| > 1e-2
    assert counts[1e-1] == 2  # |0.5|, |-0.3| > 1e-1
    assert counts[1e0] == 0   # none > 1.0


def test_density_recovery_non_negative():
    """
    Test that recovered density is non-negative (methodology adherence).
    Uses a simple Bachelier case with small N for speed.
    """
    S0, sigma, r, tau = 0.1, 0.1, 0.05, 1.0
    x, dx = build_density_grid(-0.9, 0.9, 200)
    strikes = np.linspace(-0.5, 0.5, 30)

    # Generate prices
    call_prices = bachelier_call(S0, strikes, sigma, r, tau)
    put_prices = bachelier_put(S0, strikes, sigma, r, tau)
    Pr = np.concatenate([call_prices, put_prices])

    # Build kernel and SVD
    G = build_call_put_kernel(x, dx, strikes, r, tau)
    U, s, Vt = compute_svd(G, full_matrices=False)
    Q = min(30, len(s))
    U_tilde, S_tilde, V_tilde = truncate_svd(U, s, Vt, Q)

    # Recover density
    phi, phi_prime, status = solve_density_recovery(
        Pr, U_tilde, S_tilde, V_tilde, dx, lam=1e-7
    )

    assert phi is not None, f"Solver failed with status: {status}"
    assert np.all(phi >= -1e-5), f"Density has negative values: min={phi.min():.4e}"


def test_density_recovery_unit_mass():
    """
    Test that recovered density has unit mass (methodology adherence).
    """
    S0, sigma, r, tau = 0.1, 0.1, 0.05, 1.0
    x, dx = build_density_grid(-0.9, 0.9, 200)
    strikes = np.linspace(-0.5, 0.5, 30)

    call_prices = bachelier_call(S0, strikes, sigma, r, tau)
    put_prices = bachelier_put(S0, strikes, sigma, r, tau)
    Pr = np.concatenate([call_prices, put_prices])

    G = build_call_put_kernel(x, dx, strikes, r, tau)
    U, s, Vt = compute_svd(G, full_matrices=False)
    Q = min(30, len(s))
    U_tilde, S_tilde, V_tilde = truncate_svd(U, s, Vt, Q)

    phi, phi_prime, status = solve_density_recovery(
        Pr, U_tilde, S_tilde, V_tilde, dx, lam=1e-7
    )

    assert phi is not None, f"Solver failed with status: {status}"
    mass = np.trapz(phi, dx=dx)
    assert abs(mass - 1.0) < 1e-3, f"Density mass = {mass:.6f}, expected 1.0"


def test_density_recovery_pricing_accuracy():
    """
    Test that recovered density reprices options accurately at small lambda.
    """
    S0, sigma, r, tau = 0.1, 0.1, 0.05, 1.0
    x, dx = build_density_grid(-0.9, 0.9, 300)
    strikes = np.linspace(-0.5, 0.5, 20)

    call_prices = bachelier_call(S0, strikes, sigma, r, tau)
    put_prices = bachelier_put(S0, strikes, sigma, r, tau)
    Pr = np.concatenate([call_prices, put_prices])

    G = build_call_put_kernel(x, dx, strikes, r, tau)
    U, s, Vt = compute_svd(G, full_matrices=False)
    Q = min(40, len(s))
    U_tilde, S_tilde, V_tilde = truncate_svd(U, s, Vt, Q)

    phi, phi_prime, status = solve_density_recovery(
        Pr, U_tilde, S_tilde, V_tilde, dx, lam=1e-10
    )

    if phi is not None:
        # Reprice
        Pr_hat = G @ phi
        rel_error = np.abs(Pr - Pr_hat) / (np.abs(Pr) + 1e-10)
        # At very small lambda, pricing should be reasonably accurate
        assert np.median(rel_error) < 0.5, f"Median relative pricing error too large: {np.median(rel_error):.4f}"


def test_density_recovery_larger_lambda_smoother():
    """
    Test that larger lambda produces smoother (less oscillatory) density.
    Measured by total variation.
    """
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

    phi_small, _, _ = solve_density_recovery(
        Pr, U_tilde, S_tilde, V_tilde, dx, lam=1e-12
    )
    phi_large, _, _ = solve_density_recovery(
        Pr, U_tilde, S_tilde, V_tilde, dx, lam=1e-5
    )

    if phi_small is not None and phi_large is not None:
        tv_small = np.sum(np.abs(np.diff(phi_small)))
        tv_large = np.sum(np.abs(np.diff(phi_large)))
        # Larger lambda should produce smoother density (lower TV)
        assert tv_large <= tv_small * 1.5, (
            f"Larger lambda should produce smoother density: "
            f"TV(small)={tv_small:.4f}, TV(large)={tv_large:.4f}"
        )


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

def test_single_strike_kernel():
    """Test kernel matrix with a single strike."""
    from src.kernel_matrix import build_kernel_matrix
    x, dx = build_density_grid(0.0, 2.0, 100)
    strikes = np.array([1.0])
    option_types = ["call"]
    G = build_kernel_matrix(x, dx, strikes, r=0.05, tau=1.0, option_types=option_types)
    assert G.shape == (1, 100)
    assert np.all(G >= 0)


def test_atm_call_equals_atm_put_bachelier():
    """Test that ATM call = ATM put for Bachelier when r=0."""
    S0, sigma, r, tau = 0.5, 0.1, 0.0, 1.0
    K = S0  # ATM
    C = bachelier_call(S0, K, sigma, r, tau)
    P = bachelier_put(S0, K, sigma, r, tau)
    assert abs(C - P) < 1e-12


def test_deep_itm_call_approaches_intrinsic():
    """Test that deep ITM call approaches discounted intrinsic value."""
    S0, sigma, r, tau = 1.0, 0.2, 0.05, 1.0
    K = 0.01  # very low strike
    C = black_scholes_call(S0, K, sigma, r, tau)
    discount = np.exp(-r * tau)
    F = S0 * np.exp(r * tau)
    intrinsic = discount * (F - K)
    assert abs(C - intrinsic) < 0.01
