"""
Tests for Experiment 1: Kernel Matrix Diagnostics.

Validates:
- Kernel matrix construction with trapezoidal weights
- SVD computation and condition number
- Power-law fitting
- Condition number growth with M
- Singular value decay pattern
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kernel_matrix import build_density_grid, build_kernel_row, build_kernel_matrix
from src.svd_utils import (
    compute_svd,
    condition_number,
    truncated_condition_number,
    fit_power_law,
)


# ---------------------------------------------------------------------------
# Tests for kernel matrix construction
# ---------------------------------------------------------------------------

def test_build_density_grid_basic():
    """Test that density grid has correct shape and spacing."""
    x, dx = build_density_grid(0.0, 1.0, 100)
    assert len(x) == 100
    assert abs(x[0] - 0.0) < 1e-12
    assert abs(x[-1] - 1.0) < 1e-12
    assert abs(dx - 1.0 / 99) < 1e-12


def test_build_density_grid_uniform():
    """Test that grid is uniformly spaced."""
    x, dx = build_density_grid(-1.0, 1.0, 50)
    diffs = np.diff(x)
    assert np.allclose(diffs, diffs[0], rtol=1e-10)


def test_build_density_grid_invalid():
    """Test that invalid inputs raise assertions."""
    with pytest.raises(AssertionError):
        build_density_grid(1.0, 0.0, 100)  # x_max < x_min
    with pytest.raises(AssertionError):
        build_density_grid(0.0, 1.0, 1)  # N < 2


def test_kernel_row_call_non_negative():
    """Test that call kernel row is non-negative."""
    x, dx = build_density_grid(0.5, 1.5, 100)
    row = build_kernel_row(x, dx, K=1.0, r=0.05, tau=1.0, option_type="call")
    assert np.all(row >= 0)


def test_kernel_row_put_non_negative():
    """Test that put kernel row is non-negative."""
    x, dx = build_density_grid(0.5, 1.5, 100)
    row = build_kernel_row(x, dx, K=1.0, r=0.05, tau=1.0, option_type="put")
    assert np.all(row >= 0)


def test_kernel_row_trapezoidal_weights():
    """Test that trapezoidal endpoint half-weights are applied correctly."""
    x, dx = build_density_grid(0.0, 2.0, 5)
    K = 0.5
    row = build_kernel_row(x, dx, K=K, r=0.0, tau=1.0, option_type="call")

    # Manually compute expected row
    payoff = np.maximum(0.0, x - K)
    g = dx * payoff  # r=0, tau=1 → discount=1
    g_expected = g.copy()
    g_expected[0] *= 0.5
    g_expected[-1] *= 0.5

    assert np.allclose(row, g_expected, rtol=1e-12)


def test_kernel_row_call_zero_below_strike():
    """Test that call payoff is zero for x < K."""
    x, dx = build_density_grid(0.0, 2.0, 200)
    K = 1.0
    row = build_kernel_row(x, dx, K=K, r=0.0, tau=1.0, option_type="call")
    below_strike = x < K
    assert np.all(row[below_strike] == 0.0)


def test_kernel_row_put_zero_above_strike():
    """Test that put payoff is zero for x > K."""
    x, dx = build_density_grid(0.0, 2.0, 200)
    K = 1.0
    row = build_kernel_row(x, dx, K=K, r=0.0, tau=1.0, option_type="put")
    above_strike = x > K
    assert np.all(row[above_strike] == 0.0)


def test_kernel_matrix_shape():
    """Test that kernel matrix has correct shape."""
    x, dx = build_density_grid(0.5, 1.5, 100)
    strikes = np.linspace(0.7, 1.3, 10)
    option_types = ["call"] * 10
    G = build_kernel_matrix(x, dx, strikes, r=0.05, tau=1.0, option_types=option_types)
    assert G.shape == (10, 100)


def test_kernel_matrix_non_negative():
    """Test that all kernel matrix entries are non-negative."""
    x, dx = build_density_grid(0.5, 1.5, 100)
    strikes = np.linspace(0.7, 1.3, 10)
    option_types = ["call"] * 5 + ["put"] * 5
    G = build_kernel_matrix(x, dx, strikes, r=0.05, tau=1.0, option_types=option_types)
    assert np.all(G >= 0)


def test_kernel_matrix_discount_factor():
    """Test that discount factor is correctly applied."""
    x, dx = build_density_grid(0.0, 2.0, 100)
    K = 0.5
    r, tau = 0.1, 2.0
    discount = np.exp(-r * tau)

    row_discounted = build_kernel_row(x, dx, K, r, tau, "call")
    row_undiscounted = build_kernel_row(x, dx, K, 0.0, tau, "call")

    # Ratio should equal discount factor (where row is nonzero)
    nonzero = row_undiscounted > 0
    ratios = row_discounted[nonzero] / row_undiscounted[nonzero]
    assert np.allclose(ratios, discount, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests for SVD utilities
# ---------------------------------------------------------------------------

def test_svd_descending_order():
    """Test that singular values are in descending order."""
    np.random.seed(42)
    G = np.random.randn(20, 50)
    _, s, _ = compute_svd(G)
    assert np.all(np.diff(s) <= 1e-10)


def test_condition_number_identity():
    """Test condition number of identity matrix is 1."""
    G = np.eye(10)
    _, s, _ = compute_svd(G)
    cond = condition_number(s)
    assert abs(cond - 1.0) < 1e-10


def test_condition_number_ill_conditioned():
    """Test that ill-conditioned matrix has large condition number."""
    # Create matrix with exponentially decaying singular values
    n = 20
    s_true = np.exp(-np.arange(n, dtype=float))
    U = np.eye(n)
    G = U * s_true
    _, s, _ = compute_svd(G)
    cond = condition_number(s)
    assert cond > 1e5


def test_truncated_condition_number():
    """Test truncated condition number C_tilde = s_1/s_Q."""
    np.random.seed(42)
    G = np.random.randn(30, 50)
    _, s, _ = compute_svd(G)
    Q = 10
    cond_tilde = truncated_condition_number(s, Q)
    assert abs(cond_tilde - s[0] / s[Q - 1]) < 1e-10
    # Truncated cond should be <= full cond
    assert cond_tilde <= condition_number(s) + 1e-10


def test_fit_power_law_exact():
    """Test power-law fitting on exact power-law data."""
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    k_true = 2.0
    a_true = 3.0
    y = a_true * x**k_true

    k_fit, log_a_fit = fit_power_law(x, y)
    assert abs(k_fit - k_true) < 1e-8
    assert abs(np.exp(log_a_fit) - a_true) < 1e-6


def test_fit_power_law_decay():
    """Test power-law fitting on decaying data."""
    x = np.arange(1, 20, dtype=float)
    k_true = -2.7
    y = 5.0 * x**k_true

    k_fit, log_a_fit = fit_power_law(x, y)
    assert abs(k_fit - k_true) < 1e-6


# ---------------------------------------------------------------------------
# Tests for methodology adherence (Exp1)
# ---------------------------------------------------------------------------

def test_condition_number_grows_with_M():
    """
    Verify that condition number grows with M (methodology adherence).
    Uses small N for speed.
    """
    x, dx = build_density_grid(0.5, 1.5, 500)
    cond_numbers = []
    M_values = [5, 10, 20, 40]

    for M in M_values:
        strikes = np.linspace(0.7, 1.3, M)
        option_types = ["call"] * M
        G = build_kernel_matrix(x, dx, strikes, r=0.05, tau=1.0, option_types=option_types)
        _, s, _ = compute_svd(G)
        cond_numbers.append(condition_number(s))

    # Condition number should be monotonically increasing with M
    for i in range(len(cond_numbers) - 1):
        assert cond_numbers[i] < cond_numbers[i + 1], (
            f"Condition number should increase with M: "
            f"C(M={M_values[i]})={cond_numbers[i]:.2e} >= "
            f"C(M={M_values[i+1]})={cond_numbers[i+1]:.2e}"
        )


def test_kernel_is_ill_conditioned():
    """
    Verify that the pricing kernel is ill-conditioned (methodology adherence).
    The condition number should be >> 1 for any reasonable M.
    With N=1000 and M=25, condition number is typically ~1e3-1e4.
    With N=10000 and M=25, it grows much larger.
    """
    x, dx = build_density_grid(0.5, 1.5, 1000)
    strikes = np.linspace(0.7, 1.3, 25)
    option_types = ["call"] * 25
    G = build_kernel_matrix(x, dx, strikes, r=0.05, tau=1.0, option_types=option_types)
    _, s, _ = compute_svd(G)
    cond = condition_number(s)
    # Condition number should be significantly > 1 (ill-conditioned)
    assert cond > 1e2, f"Expected ill-conditioned matrix, got cond={cond:.2e}"


def test_singular_values_decay():
    """
    Verify that singular values decay rapidly (methodology adherence).
    The ratio s_1/s_10 should be >> 1.
    """
    x, dx = build_density_grid(0.5, 1.5, 1000)
    strikes = np.linspace(0.7, 1.3, 25)
    option_types = ["call"] * 25
    G = build_kernel_matrix(x, dx, strikes, r=0.05, tau=1.0, option_types=option_types)
    _, s, _ = compute_svd(G)
    # Normalized singular values should decay steeply
    s_norm = s / s[0]
    assert s_norm[9] < 0.01, f"Expected rapid decay, s_10/s_1 = {s_norm[9]:.4f}"


def test_trapezoidal_integration_of_constant():
    """
    Test that trapezoidal integration of a constant density gives 1.
    This validates the kernel construction for pricing.
    """
    x, dx = build_density_grid(0.0, 1.0, 1000)
    # Uniform density: phi(x) = 1 on [0,1]
    phi = np.ones(len(x))
    # Trapezoidal mass
    trap_weights = np.ones(len(x))
    trap_weights[0] = 0.5
    trap_weights[-1] = 0.5
    mass = np.sum(trap_weights * phi * dx)
    assert abs(mass - 1.0) < 1e-10


def test_power_law_exponent_near_2():
    """
    Regression test: condition number power-law exponent should be near 2.
    Uses a small N for speed.
    """
    x, dx = build_density_grid(0.5, 1.5, 500)
    M_values = np.array([5, 10, 20, 40, 80])
    cond_numbers = []

    for M in M_values:
        strikes = np.linspace(0.7, 1.3, M)
        option_types = ["call"] * M
        G = build_kernel_matrix(x, dx, strikes, r=0.05, tau=1.0, option_types=option_types)
        _, s, _ = compute_svd(G)
        cond_numbers.append(condition_number(s))

    k, _ = fit_power_law(M_values.astype(float), np.array(cond_numbers))
    # Exponent should be in range [1.0, 4.0] (near 2)
    assert 1.0 <= k <= 4.0, f"Power-law exponent k={k:.3f} outside expected range [1, 4]"
