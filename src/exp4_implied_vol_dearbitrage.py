"""
Experiment 4: Implied-Volatility De-Arbitraging from Arbitrage-Contaminated Inputs.

Uses the recovered feasible density from Exp3 to generate arbitrage-free option
prices and compares the resulting implied-volatility smile with the original
contaminated smile.

Custom implementation — Context7 found no library equivalent (paper methodology).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from src.kernel_matrix import build_density_grid, build_call_put_kernel, price_from_density
from src.svd_utils import compute_svd, truncate_svd
from src.density_recovery import solve_density_recovery
from src.option_pricing import gaussian_mixture_density, mixture_call_price, mixture_put_price
from src.implied_vol import compute_implied_vol_smile, black_implied_vol
from src.option_pricing import black_call, black_put


# ---------------------------------------------------------------------------
# Experiment parameters (same as Exp3)
# ---------------------------------------------------------------------------

WEIGHTS = [0.55, -0.20, 0.65]
MEANS = [0.80, 1.15, 1.35]
STDS = [0.10, 0.07, 0.20]

R = 0.05
TAU = 1.0
K_MIN = 0.3
K_MAX = 1.7
M = 200
X_MIN = 0.1
X_MAX = 2.2
N = 1000
Q = 150
HIGHLIGHTED_LAMBDAS = [1e-7, 1e-5]

# S0 chosen arbitrarily by authors for IV inversion (paper specification)
S0_IV = 1.0


def compute_original_iv_smile(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    put_prices: np.ndarray,
    F: float,
    r: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute original implied-volatility smile from contaminated prices.

    Uses OTM convention: puts for K < F, calls for K >= F.

    Parameters
    ----------
    strikes : np.ndarray
        Strike prices.
    call_prices : np.ndarray
        Call prices.
    put_prices : np.ndarray
        Put prices.
    F : float
        Forward price (= S0 * exp(r*tau)).
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.

    Returns
    -------
    iv_calls : np.ndarray
        Implied vols from calls (NaN where inversion failed).
    iv_puts : np.ndarray
        Implied vols from puts (NaN where inversion failed).
    iv_otm : np.ndarray
        OTM implied vols.
    valid_otm : np.ndarray of bool
        Valid mask for OTM.
    """
    M = len(strikes)

    # Invert calls
    iv_calls, valid_calls = compute_implied_vol_smile(
        call_prices, F, strikes, r, tau, ["call"] * M
    )

    # Invert puts
    iv_puts, valid_puts = compute_implied_vol_smile(
        put_prices, F, strikes, r, tau, ["put"] * M
    )

    # OTM convention
    iv_otm = np.full(M, np.nan)
    valid_otm = np.zeros(M, dtype=bool)
    for i in range(M):
        if strikes[i] >= F:
            iv_otm[i] = iv_calls[i]
            valid_otm[i] = valid_calls[i]
        else:
            iv_otm[i] = iv_puts[i]
            valid_otm[i] = valid_puts[i]

    return iv_calls, iv_puts, iv_otm, valid_otm


def compute_dearbitraged_iv_smile(
    phi_rec: np.ndarray,
    x: np.ndarray,
    dx: float,
    strikes: np.ndarray,
    F: float,
    r: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute de-arbitraged implied-volatility smile from recovered density.

    Parameters
    ----------
    phi_rec : np.ndarray, shape (N,)
        Recovered non-negative density.
    x : np.ndarray, shape (N,)
        Density grid.
    dx : float
        Grid spacing.
    strikes : np.ndarray
        Strike prices.
    F : float
        Forward price.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.

    Returns
    -------
    call_prices_rec : np.ndarray
        Repriced call prices.
    put_prices_rec : np.ndarray
        Repriced put prices.
    iv_otm_rec : np.ndarray
        OTM implied vols from recovered density.
    """
    M = len(strikes)
    discount = np.exp(-r * tau)

    # Reprice calls and puts from recovered density
    call_prices_rec = np.array([
        np.trapz(discount * np.maximum(0.0, x - K) * phi_rec, dx=dx)
        for K in strikes
    ])
    put_prices_rec = np.array([
        np.trapz(discount * np.maximum(0.0, K - x) * phi_rec, dx=dx)
        for K in strikes
    ])

    # Invert to implied vols (OTM convention)
    iv_otm_rec = np.full(M, np.nan)
    for i in range(M):
        if strikes[i] >= F:
            iv = black_implied_vol(call_prices_rec[i], F, strikes[i], r, tau, "call")
        else:
            iv = black_implied_vol(put_prices_rec[i], F, strikes[i], r, tau, "put")
        if iv is not None:
            iv_otm_rec[i] = iv

    return call_prices_rec, put_prices_rec, iv_otm_rec


def run_experiment4(results_dir: str = "results") -> dict:
    """
    Run Experiment 4: Implied-Volatility De-Arbitraging.

    Parameters
    ----------
    results_dir : str
        Directory to save all outputs.

    Returns
    -------
    results : dict
        Original and de-arbitraged IV smiles, recovered densities.
    """
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Implied-Volatility De-Arbitraging")
    print("=" * 70)
    print(f"Signed mixture: weights={WEIGHTS}, means={MEANS}, stds={STDS}")
    print(f"r={R}, tau={TAU}, S0_IV={S0_IV} (for IV inversion)")
    print(f"K in [{K_MIN},{K_MAX}], x in [{X_MIN},{X_MAX}], N={N}, Q={Q}")

    # Build density grid
    x, dx = build_density_grid(X_MIN, X_MAX, N)

    # Generate arbitrage-contaminated prices
    strikes = np.linspace(K_MIN, K_MAX, M)
    call_prices_orig = mixture_call_price(strikes, WEIGHTS, MEANS, STDS, R, TAU)
    put_prices_orig = mixture_put_price(strikes, WEIGHTS, MEANS, STDS, R, TAU)
    Pr = np.concatenate([call_prices_orig, put_prices_orig])

    # Forward price for IV inversion (S0=1.0 as per paper)
    F = S0_IV * np.exp(R * TAU)
    print(f"\nForward price F = {F:.4f} (S0={S0_IV}, r={R}, tau={TAU})")

    # Compute original IV smile from contaminated prices
    iv_calls_orig, iv_puts_orig, iv_otm_orig, valid_otm_orig = compute_original_iv_smile(
        strikes, call_prices_orig, put_prices_orig, F, R, TAU
    )
    n_valid_orig = valid_otm_orig.sum()
    n_failed_orig = (~valid_otm_orig).sum()
    print(f"\nOriginal IV inversion: {n_valid_orig} valid, {n_failed_orig} failed")
    if n_failed_orig > 0:
        failed_strikes = strikes[~valid_otm_orig]
        print(f"  Failed strikes: {failed_strikes[:5]}{'...' if n_failed_orig > 5 else ''}")

    # Build kernel matrix and SVD
    G = build_call_put_kernel(x, dx, strikes, R, TAU)
    U, s, Vt = compute_svd(G, full_matrices=False)
    U_tilde, S_tilde, V_tilde = truncate_svd(U, s, Vt, Q)

    # Recover densities at highlighted lambdas
    recovered = {}
    for lam in HIGHLIGHTED_LAMBDAS:
        phi_rec, phi_prime, status = solve_density_recovery(
            Pr, U_tilde, S_tilde, V_tilde, dx, lam
        )
        if phi_rec is not None:
            mass = np.trapz(phi_rec, dx=dx)
            min_phi = phi_rec.min()
            print(f"\n  lambda={lam:.1e}: status={status}, "
                  f"min(phi)={min_phi:.4e}, mass={mass:.6f}")

            # Compute de-arbitraged IV smile
            call_rec, put_rec, iv_otm_rec = compute_dearbitraged_iv_smile(
                phi_rec, x, dx, strikes, F, R, TAU
            )
            n_valid_rec = (~np.isnan(iv_otm_rec)).sum()
            print(f"    De-arbitraged IV: {n_valid_rec} valid inversions")

            recovered[lam] = {
                "phi": phi_rec,
                "phi_prime": phi_prime,
                "call_prices": call_rec,
                "put_prices": put_rec,
                "iv_otm": iv_otm_rec,
            }
        else:
            print(f"\n  lambda={lam:.1e}: FAILED (status={status})")
            recovered[lam] = None

    # Plots
    _plot_iv_comparison(
        strikes, iv_otm_orig, valid_otm_orig, recovered, results_dir
    )
    _plot_density_and_iv(
        x, strikes, iv_otm_orig, valid_otm_orig, recovered, results_dir
    )

    return {
        "x": x,
        "strikes": strikes,
        "call_prices_orig": call_prices_orig,
        "put_prices_orig": put_prices_orig,
        "iv_otm_orig": iv_otm_orig,
        "valid_otm_orig": valid_otm_orig,
        "recovered": recovered,
        "F": F,
        "parameters": {
            "weights": WEIGHTS,
            "means": MEANS,
            "stds": STDS,
            "r": R,
            "tau": TAU,
            "S0_IV": S0_IV,
            "K_min": K_MIN,
            "K_max": K_MAX,
            "x_min": X_MIN,
            "x_max": X_MAX,
            "N": N,
            "Q": Q,
            "M": M,
        },
    }


def _plot_iv_comparison(
    strikes: np.ndarray,
    iv_otm_orig: np.ndarray,
    valid_otm_orig: np.ndarray,
    recovered: dict,
    results_dir: str,
) -> None:
    """Plot original vs de-arbitraged IV smiles."""
    fig, axes = plt.subplots(1, len(HIGHLIGHTED_LAMBDAS), figsize=(7 * len(HIGHLIGHTED_LAMBDAS), 5))
    if len(HIGHLIGHTED_LAMBDAS) == 1:
        axes = [axes]

    colors = ["steelblue", "tomato"]
    for ax, (lam, color) in zip(axes, zip(HIGHLIGHTED_LAMBDAS, colors)):
        # Original smile
        ax.plot(
            strikes[valid_otm_orig],
            iv_otm_orig[valid_otm_orig],
            "k.-",
            linewidth=1.5,
            markersize=4,
            label="Original (contaminated)",
        )

        # De-arbitraged smile
        if recovered.get(lam) is not None:
            iv_rec = recovered[lam]["iv_otm"]
            valid_rec = ~np.isnan(iv_rec)
            ax.plot(
                strikes[valid_rec],
                iv_rec[valid_rec],
                "-",
                color=color,
                linewidth=2,
                label=f"De-arbitraged ($\\lambda={lam:.0e}$)",
            )

        ax.set_xlabel("Strike $K$", fontsize=12)
        ax.set_ylabel("Implied volatility", fontsize=12)
        ax.set_title(f"IV Smile Comparison\n$\\lambda = {lam:.0e}$", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "exp4_iv_comparison.png"), dpi=150)
    plt.close(fig)


def _plot_density_and_iv(
    x: np.ndarray,
    strikes: np.ndarray,
    iv_otm_orig: np.ndarray,
    valid_otm_orig: np.ndarray,
    recovered: dict,
    results_dir: str,
) -> None:
    """Plot recovered densities and IV smiles side by side."""
    fig, axes = plt.subplots(2, len(HIGHLIGHTED_LAMBDAS),
                              figsize=(7 * len(HIGHLIGHTED_LAMBDAS), 10))

    colors = ["steelblue", "tomato"]
    for col, (lam, color) in enumerate(zip(HIGHLIGHTED_LAMBDAS, colors)):
        # Density plot
        ax_d = axes[0, col]
        if recovered.get(lam) is not None:
            phi_rec = recovered[lam]["phi"]
            ax_d.plot(x, phi_rec, "-", color=color, linewidth=2)
            ax_d.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax_d.set_xlabel("Terminal price $x$", fontsize=11)
        ax_d.set_ylabel("Density $\\phi(x)$", fontsize=11)
        ax_d.set_title(f"Recovered Density\n$\\lambda = {lam:.0e}$", fontsize=11)
        ax_d.grid(True, alpha=0.3)

        # IV plot
        ax_iv = axes[1, col]
        ax_iv.plot(
            strikes[valid_otm_orig],
            iv_otm_orig[valid_otm_orig],
            "k.-",
            linewidth=1.5,
            markersize=4,
            label="Original",
        )
        if recovered.get(lam) is not None:
            iv_rec = recovered[lam]["iv_otm"]
            valid_rec = ~np.isnan(iv_rec)
            ax_iv.plot(
                strikes[valid_rec],
                iv_rec[valid_rec],
                "-",
                color=color,
                linewidth=2,
                label=f"De-arbitraged",
            )
        ax_iv.set_xlabel("Strike $K$", fontsize=11)
        ax_iv.set_ylabel("Implied volatility", fontsize=11)
        ax_iv.set_title(f"IV Smile\n$\\lambda = {lam:.0e}$", fontsize=11)
        ax_iv.legend(fontsize=10)
        ax_iv.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "exp4_density_and_iv.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_experiment4()
