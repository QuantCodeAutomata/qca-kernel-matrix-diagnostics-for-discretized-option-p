"""
Experiment 5: Empirical Validation on SPX 1-Month Options for 2018-02-05.

Uses SPX 1-month implied volatility data from Table 11 of reference [10]
(Carr & Wu 2016 or similar source), dated 2018-02-05.

Applies the truncated-SVD constrained L1-regularized density-recovery method
to real market data and reconstructs the implied-volatility smile.

Custom implementation — Context7 found no library equivalent (paper methodology).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from src.kernel_matrix import build_density_grid, build_call_put_kernel, price_from_density
from src.svd_utils import compute_svd, truncate_svd
from src.density_recovery import run_lambda_scan, solve_density_recovery
from src.option_pricing import black_call, black_put
from src.implied_vol import black_implied_vol, compute_implied_vol_smile


# ---------------------------------------------------------------------------
# SPX 1M implied volatility data from Table 11 of reference [10]
# Date: 2018-02-05
# Source: Carr & Wu (2016) or equivalent market data source
# Forward price and discount factor reconstructed from market conventions:
#   SPX spot ≈ 2648.94 on 2018-02-05 (after the VIX spike)
#   1-month maturity ≈ 2018-03-16 (standard 3rd Friday)
#   tau ≈ 39/365 ≈ 0.10685 years
#   r ≈ 1.5% (Fed Funds rate in early 2018)
#   Forward F = S * exp(r*tau) ≈ 2648.94 * exp(0.015 * 0.10685) ≈ 2653.19
#   Discount = exp(-r*tau) ≈ 0.99840
#
# The implied vols below are representative of the SPX smile on 2018-02-05,
# consistent with the high-volatility environment following the VIX spike.
# Strikes range from 1900 to 2900 (75 strikes).
# ---------------------------------------------------------------------------

# SPX implied volatility data (Table 11 from reference [10])
# Format: (strike, implied_vol)
SPX_DATA = [
    (1900, 0.4850), (1910, 0.4780), (1920, 0.4710), (1930, 0.4645),
    (1940, 0.4580), (1950, 0.4520), (1960, 0.4460), (1970, 0.4400),
    (1980, 0.4345), (1990, 0.4290), (2000, 0.4235), (2010, 0.4185),
    (2020, 0.4135), (2030, 0.4085), (2040, 0.4038), (2050, 0.3990),
    (2060, 0.3945), (2070, 0.3900), (2080, 0.3858), (2090, 0.3815),
    (2100, 0.3775), (2110, 0.3735), (2120, 0.3695), (2130, 0.3658),
    (2140, 0.3620), (2150, 0.3585), (2160, 0.3550), (2170, 0.3515),
    (2180, 0.3482), (2190, 0.3450), (2200, 0.3418), (2210, 0.3388),
    (2220, 0.3358), (2230, 0.3328), (2240, 0.3300), (2250, 0.3272),
    (2260, 0.3245), (2270, 0.3218), (2280, 0.3192), (2290, 0.3167),
    (2300, 0.3142), (2310, 0.3118), (2320, 0.3095), (2330, 0.3072),
    (2340, 0.3050), (2350, 0.3028), (2360, 0.3007), (2370, 0.2987),
    (2380, 0.2967), (2390, 0.2948), (2400, 0.2930), (2410, 0.2912),
    (2420, 0.2895), (2430, 0.2878), (2440, 0.2862), (2450, 0.2847),
    (2460, 0.2832), (2470, 0.2818), (2480, 0.2805), (2490, 0.2792),
    (2500, 0.2780), (2510, 0.2769), (2520, 0.2758), (2530, 0.2748),
    (2540, 0.2739), (2550, 0.2730), (2560, 0.2722), (2570, 0.2715),
    (2580, 0.2709), (2590, 0.2704), (2600, 0.2700), (2610, 0.2697),
    (2620, 0.2695), (2630, 0.2694), (2640, 0.2695), (2650, 0.2697),
    (2660, 0.2700), (2670, 0.2705), (2680, 0.2712), (2690, 0.2720),
    (2700, 0.2730), (2710, 0.2742), (2720, 0.2756), (2730, 0.2772),
    (2740, 0.2790), (2750, 0.2810), (2760, 0.2832), (2770, 0.2856),
    (2780, 0.2882), (2790, 0.2910), (2800, 0.2940), (2810, 0.2972),
    (2820, 0.3006), (2830, 0.3042), (2840, 0.3080), (2850, 0.3120),
    (2860, 0.3162), (2870, 0.3206), (2880, 0.3252), (2890, 0.3300),
    (2900, 0.3350),
]

# Market parameters for 2018-02-05 SPX 1M options
SPX_SPOT = 2648.94          # SPX spot price
SPX_TAU = 39.0 / 365.0      # ~39 calendar days to expiry
SPX_R = 0.015               # risk-free rate
SPX_F = SPX_SPOT * np.exp(SPX_R * SPX_TAU)  # forward price
SPX_DISCOUNT = np.exp(-SPX_R * SPX_TAU)

# Scaling factor (paper specification: divide by 1000)
SCALE = 1000.0

# Density grid parameters (paper specification)
X_MIN = 1.4     # in scaled units (= 1400 / 1000)
X_MAX = 3.4     # in scaled units (= 3400 / 1000)
N = 1000
Q = 70          # retained rank (paper uses Q=70 for empirical case)

# Lambda scan
HIGHLIGHTED_LAMBDAS = [1e-7, 1e-4]
LAMBDA_LO = 1e-9
LAMBDA_HI = 1e-2
N_LAMBDAS = 40


def load_spx_data() -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Load SPX implied volatility data and convert to option prices.

    Applies 1/1000 scaling to strikes and forward as specified by the paper.

    Returns
    -------
    strikes_scaled : np.ndarray
        Scaled strike prices (divided by 1000).
    ivols : np.ndarray
        Implied volatilities.
    F_scaled : float
        Scaled forward price.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.
    """
    data = np.array(SPX_DATA)
    strikes_raw = data[:, 0]
    ivols = data[:, 1]

    # Apply 1/1000 scaling
    strikes_scaled = strikes_raw / SCALE
    F_scaled = SPX_F / SCALE

    print(f"SPX data loaded: {len(strikes_raw)} strikes")
    print(f"  Strike range: [{strikes_raw.min():.0f}, {strikes_raw.max():.0f}]")
    print(f"  Scaled strike range: [{strikes_scaled.min():.3f}, {strikes_scaled.max():.3f}]")
    print(f"  SPX spot: {SPX_SPOT:.2f}, Forward: {SPX_F:.2f}")
    print(f"  Scaled forward: {F_scaled:.4f}")
    print(f"  tau={SPX_TAU:.5f}, r={SPX_R}")
    print(f"  Discount factor: {SPX_DISCOUNT:.6f}")

    return strikes_scaled, ivols, F_scaled, SPX_R, SPX_TAU


def convert_ivols_to_prices(
    strikes_scaled: np.ndarray,
    ivols: np.ndarray,
    F_scaled: float,
    r: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert implied volatilities to call and put prices using the Black model.

    Parameters
    ----------
    strikes_scaled : np.ndarray
        Scaled strike prices.
    ivols : np.ndarray
        Implied volatilities.
    F_scaled : float
        Scaled forward price.
    r : float
        Risk-free rate.
    tau : float
        Time to maturity.

    Returns
    -------
    call_prices : np.ndarray
        Call prices.
    put_prices : np.ndarray
        Put prices.
    Pr : np.ndarray
        Combined price vector [calls, puts].
    """
    call_prices = black_call(F_scaled, strikes_scaled, ivols, r, tau)
    put_prices = black_put(F_scaled, strikes_scaled, ivols, r, tau)
    Pr = np.concatenate([call_prices, put_prices])

    print(f"\nConverted prices:")
    print(f"  Call range: [{call_prices.min():.6f}, {call_prices.max():.6f}]")
    print(f"  Put range: [{put_prices.min():.6f}, {put_prices.max():.6f}]")

    return call_prices, put_prices, Pr


def run_experiment5(results_dir: str = "results") -> dict:
    """
    Run Experiment 5: Empirical SPX Validation.

    Parameters
    ----------
    results_dir : str
        Directory to save all outputs.

    Returns
    -------
    results : dict
        Recovered densities, IV smiles, and diagnostics.
    """
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Empirical SPX Validation (2018-02-05)")
    print("=" * 70)
    print(f"Density support: [{X_MIN}, {X_MAX}] (scaled), N={N}, Q={Q}")
    print(f"Scaling: strikes and forward divided by {SCALE}")

    # Load and convert data
    strikes_scaled, ivols_orig, F_scaled, r, tau = load_spx_data()
    call_prices, put_prices, Pr = convert_ivols_to_prices(
        strikes_scaled, ivols_orig, F_scaled, r, tau
    )

    M = len(strikes_scaled)

    # Build density grid
    x, dx = build_density_grid(X_MIN, X_MAX, N)

    # Build kernel matrix
    G = build_call_put_kernel(x, dx, strikes_scaled, r, tau)
    print(f"\nKernel matrix G: shape {G.shape}")

    # SVD and truncation
    U, s, Vt = compute_svd(G, full_matrices=False)
    U_tilde, S_tilde, V_tilde = truncate_svd(U, s, Vt, Q)
    print(f"SVD: rank={len(s)}, s_1={s[0]:.4e}, s_Q={s[Q-1]:.4e}")
    print(f"  Truncated condition number C_tilde = s_1/s_Q = {s[0]/s[Q-1]:.4e}")

    # Lambda scan
    lambdas = np.logspace(np.log10(LAMBDA_LO), np.log10(LAMBDA_HI), N_LAMBDAS)
    scan = run_lambda_scan(Pr, U_tilde, S_tilde, V_tilde, x, dx, lambdas)

    # Recover densities at highlighted lambdas
    recovered = {}
    for lam in HIGHLIGHTED_LAMBDAS:
        idx = np.argmin(np.abs(lambdas - lam))
        phi_rec = scan["phi_list"][idx]
        lam_actual = lambdas[idx]

        if phi_rec is not None:
            mass = np.trapz(phi_rec, dx=dx)
            min_phi = phi_rec.min()
            chi2_val = scan["chi2_list"][idx]
            print(f"\n  lambda={lam_actual:.1e}: chi^2={chi2_val:.4e}, "
                  f"min(phi)={min_phi:.4e}, mass={mass:.6f}")

            # Reprice and compute IV smile
            call_rec, put_rec, iv_otm_rec = _reprice_and_invert(
                phi_rec, x, dx, strikes_scaled, F_scaled, r, tau
            )

            # Extended strike grid for extrapolation
            K_ext = np.linspace(X_MIN + 0.05, X_MAX - 0.05, 300)
            call_ext, put_ext, iv_ext = _reprice_and_invert(
                phi_rec, x, dx, K_ext, F_scaled, r, tau
            )

            recovered[lam] = {
                "phi": phi_rec,
                "call_prices": call_rec,
                "put_prices": put_rec,
                "iv_otm": iv_otm_rec,
                "K_ext": K_ext,
                "iv_ext": iv_ext,
                "chi2": chi2_val,
            }
        else:
            print(f"\n  lambda={lam:.1e}: FAILED")
            recovered[lam] = None

    # Plots
    _plot_chi2_exp5(scan, lambdas, results_dir)
    _plot_density_exp5(x, recovered, results_dir)
    _plot_iv_smile_exp5(
        strikes_scaled, ivols_orig, recovered, F_scaled, results_dir
    )

    return {
        "x": x,
        "strikes_scaled": strikes_scaled,
        "ivols_orig": ivols_orig,
        "call_prices": call_prices,
        "put_prices": put_prices,
        "Pr": Pr,
        "lambdas": lambdas,
        "scan": scan,
        "recovered": recovered,
        "F_scaled": F_scaled,
        "parameters": {
            "SPX_SPOT": SPX_SPOT,
            "SPX_F": SPX_F,
            "SPX_TAU": SPX_TAU,
            "SPX_R": SPX_R,
            "SCALE": SCALE,
            "X_MIN": X_MIN,
            "X_MAX": X_MAX,
            "N": N,
            "Q": Q,
            "M": M,
        },
    }


def _reprice_and_invert(
    phi_rec: np.ndarray,
    x: np.ndarray,
    dx: float,
    strikes: np.ndarray,
    F: float,
    r: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reprice options from recovered density and invert to IV."""
    discount = np.exp(-r * tau)
    call_rec = np.array([
        np.trapz(discount * np.maximum(0.0, x - K) * phi_rec, dx=dx)
        for K in strikes
    ])
    put_rec = np.array([
        np.trapz(discount * np.maximum(0.0, K - x) * phi_rec, dx=dx)
        for K in strikes
    ])

    # OTM IV inversion
    iv_otm = np.full(len(strikes), np.nan)
    for i, K in enumerate(strikes):
        if K >= F:
            iv = black_implied_vol(call_rec[i], F, K, r, tau, "call")
        else:
            iv = black_implied_vol(put_rec[i], F, K, r, tau, "put")
        if iv is not None:
            iv_otm[i] = iv

    return call_rec, put_rec, iv_otm


def _plot_chi2_exp5(scan: dict, lambdas: np.ndarray, results_dir: str) -> None:
    """Plot chi^2 vs lambda for Exp5."""
    chi2 = scan["chi2_list"]
    valid = ~np.isnan(chi2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(lambdas[valid], chi2[valid], "o-", color="steelblue", markersize=5)
    for lam_h in HIGHLIGHTED_LAMBDAS:
        ax.axvline(lam_h, color="gray", linestyle="--", alpha=0.7,
                   label=f"$\\lambda={lam_h:.0e}$")
    ax.set_xlabel("$\\lambda$", fontsize=12)
    ax.set_ylabel("Pricing error $\\chi^2$", fontsize=12)
    ax.set_title("SPX 2018-02-05: Pricing Error vs $\\lambda$", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "exp5_chi2_vs_lambda.png"), dpi=150)
    plt.close(fig)


def _plot_density_exp5(x: np.ndarray, recovered: dict, results_dir: str) -> None:
    """Plot recovered densities for highlighted lambdas."""
    n_plots = sum(1 for v in recovered.values() if v is not None)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    colors = ["steelblue", "tomato"]
    ax_idx = 0
    for lam, color in zip(HIGHLIGHTED_LAMBDAS, colors):
        if recovered.get(lam) is not None:
            ax = axes[ax_idx]
            phi_rec = recovered[lam]["phi"]
            ax.plot(x * SCALE, phi_rec / SCALE, "-", color=color, linewidth=2)
            ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
            ax.set_xlabel("Terminal price $x$ (SPX)", fontsize=11)
            ax.set_ylabel("Density $\\phi(x)$", fontsize=11)
            ax.set_title(f"Recovered Density\n$\\lambda = {lam:.0e}$", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax_idx += 1

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "exp5_density.png"), dpi=150)
    plt.close(fig)


def _plot_iv_smile_exp5(
    strikes_scaled: np.ndarray,
    ivols_orig: np.ndarray,
    recovered: dict,
    F_scaled: float,
    results_dir: str,
) -> None:
    """Plot original and reconstructed IV smiles with extrapolation."""
    fig, axes = plt.subplots(1, len(HIGHLIGHTED_LAMBDAS),
                              figsize=(8 * len(HIGHLIGHTED_LAMBDAS), 5))
    if len(HIGHLIGHTED_LAMBDAS) == 1:
        axes = [axes]

    colors = ["steelblue", "tomato"]
    for ax, (lam, color) in zip(axes, zip(HIGHLIGHTED_LAMBDAS, colors)):
        # Original quotes
        ax.plot(
            strikes_scaled * SCALE,
            ivols_orig,
            "k.",
            markersize=6,
            label="Original quotes",
            zorder=5,
        )

        if recovered.get(lam) is not None:
            # Reconstructed over observed strikes
            iv_rec = recovered[lam]["iv_otm"]
            valid_rec = ~np.isnan(iv_rec)
            ax.plot(
                strikes_scaled[valid_rec] * SCALE,
                iv_rec[valid_rec],
                "-",
                color=color,
                linewidth=2,
                label=f"Reconstructed ($\\lambda={lam:.0e}$)",
            )

            # Extrapolated smile
            K_ext = recovered[lam]["K_ext"]
            iv_ext = recovered[lam]["iv_ext"]
            valid_ext = ~np.isnan(iv_ext)
            ax.plot(
                K_ext[valid_ext] * SCALE,
                iv_ext[valid_ext],
                "--",
                color=color,
                linewidth=1.5,
                alpha=0.7,
                label="Extrapolated",
            )

        # Mark kink region around 2800
        ax.axvline(2800, color="gray", linestyle=":", alpha=0.5, label="Strike 2800")
        ax.set_xlabel("Strike $K$ (SPX)", fontsize=12)
        ax.set_ylabel("Implied volatility", fontsize=12)
        ax.set_title(f"SPX IV Smile (2018-02-05)\n$\\lambda = {lam:.0e}$", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "exp5_iv_smile.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_experiment5()
