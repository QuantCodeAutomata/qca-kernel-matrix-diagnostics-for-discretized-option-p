"""
Experiment 3: Robust Density Recovery from Arbitrage-Contaminated Synthetic Prices.

Tests whether the constrained truncated-SVD reconstruction method can produce
a non-negative, normalized, arbitrage-free density when input prices are generated
from a signed (invalid) mixture density.

Custom implementation — Context7 found no library equivalent (paper methodology).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from src.kernel_matrix import build_density_grid, build_call_put_kernel, price_from_density
from src.svd_utils import compute_svd, truncate_svd
from src.density_recovery import (
    run_lambda_scan,
    bhattacharyya_distance,
    solve_density_recovery,
)
from src.option_pricing import gaussian_mixture_density, mixture_call_price, mixture_put_price


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

# Signed Gaussian mixture (c_2 < 0 → arbitrage)
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
LAMBDA_LO = 1e-10
LAMBDA_HI = 1e-3
N_LAMBDAS = 50


def run_experiment3(results_dir: str = "results") -> dict:
    """
    Run Experiment 3: Robust density recovery from arbitrage-contaminated prices.

    Parameters
    ----------
    results_dir : str
        Directory to save all outputs.

    Returns
    -------
    results : dict
        Diagnostics, recovered densities, and feasibility checks.
    """
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Arbitrage-Contaminated Density Recovery")
    print("=" * 70)
    print(f"Signed mixture: weights={WEIGHTS}, means={MEANS}, stds={STDS}")
    print(f"r={R}, tau={TAU}, K in [{K_MIN},{K_MAX}], x in [{X_MIN},{X_MAX}]")
    print(f"N={N}, Q={Q}, M={M}")

    # Build density grid
    x, dx = build_density_grid(X_MIN, X_MAX, N)

    # Signed comparator density phi_M (may be negative)
    phi_M = gaussian_mixture_density(x, WEIGHTS, MEANS, STDS)
    print(f"\nSigned density: min={phi_M.min():.4f}, max={phi_M.max():.4f}")
    print(f"  Negative fraction: {(phi_M < 0).mean():.3f}")

    # Positive-part comparator (without renormalization, as per paper)
    phi_M_plus = np.maximum(phi_M, 0.0)
    print(f"  phi_M^+ mass (trapz): {np.trapz(phi_M_plus, dx=dx):.4f}")

    # Generate synthetic prices from signed density (may violate arbitrage)
    strikes = np.linspace(K_MIN, K_MAX, M)
    call_prices = mixture_call_price(strikes, WEIGHTS, MEANS, STDS, R, TAU)
    put_prices = mixture_put_price(strikes, WEIGHTS, MEANS, STDS, R, TAU)
    Pr = np.concatenate([call_prices, put_prices])

    print(f"\nSynthetic prices: min={Pr.min():.6f}, max={Pr.max():.6f}")
    print(f"  Negative prices: {(Pr < 0).sum()} out of {len(Pr)}")

    # Build kernel matrix
    G = build_call_put_kernel(x, dx, strikes, R, TAU)

    # SVD and truncation
    U, s, Vt = compute_svd(G, full_matrices=False)
    U_tilde, S_tilde, V_tilde = truncate_svd(U, s, Vt, Q)

    # Lambda scan — use phi_M_plus as comparator for Bhattacharyya distance
    lambdas = np.logspace(np.log10(LAMBDA_LO), np.log10(LAMBDA_HI), N_LAMBDAS)
    scan = run_lambda_scan(
        Pr, U_tilde, S_tilde, V_tilde, x, dx, lambdas,
        phi_true=phi_M_plus,  # compare against positive part
    )

    # Verify feasibility for all recovered densities
    # Use tolerance 1e-4 to account for SCS solver numerical precision
    feasibility_checks = []
    for i, phi in enumerate(scan["phi_list"]):
        if phi is not None:
            min_val = phi.min()
            mass = np.trapz(phi, dx=dx)
            feasible = (min_val >= -1e-4) and (abs(mass - 1.0) < 1e-3)
            feasibility_checks.append({
                "lambda": lambdas[i],
                "min_phi": min_val,
                "mass": mass,
                "feasible": feasible,
            })
        else:
            feasibility_checks.append({
                "lambda": lambdas[i],
                "min_phi": np.nan,
                "mass": np.nan,
                "feasible": False,
            })

    n_feasible = sum(f["feasible"] for f in feasibility_checks)
    print(f"\nFeasibility: {n_feasible}/{len(lambdas)} solutions are non-negative and normalized")

    # Print chi^2 summary
    chi2 = scan["chi2_list"]
    dB = scan["dB_list"]
    valid = ~np.isnan(chi2)
    if valid.any():
        print(f"\nchi^2 range: [{np.nanmin(chi2):.4e}, {np.nanmax(chi2):.4e}]")
        print(f"  (chi^2 > 0 confirms exact recovery is impossible)")

    # Plots
    _plot_chi2_dB_exp3(scan, lambdas, results_dir)
    _plot_density_comparison_exp3(x, phi_M, phi_M_plus, scan, lambdas, results_dir)

    # Print highlighted lambda results
    for lam_target in HIGHLIGHTED_LAMBDAS:
        idx = np.argmin(np.abs(lambdas - lam_target))
        phi_rec = scan["phi_list"][idx]
        lam_actual = lambdas[idx]
        if phi_rec is not None:
            chi2_val = scan["chi2_list"][idx]
            dB_val = scan["dB_list"][idx]
            print(f"\n  lambda={lam_actual:.1e}: chi^2={chi2_val:.4e}, d_B={dB_val:.4f}, "
                  f"min(phi)={phi_rec.min():.4e}, mass={np.trapz(phi_rec, dx=dx):.6f}")

    return {
        "x": x,
        "phi_M": phi_M,
        "phi_M_plus": phi_M_plus,
        "strikes": strikes,
        "Pr": Pr,
        "lambdas": lambdas,
        "scan": scan,
        "feasibility_checks": feasibility_checks,
        "parameters": {
            "weights": WEIGHTS,
            "means": MEANS,
            "stds": STDS,
            "r": R,
            "tau": TAU,
            "K_min": K_MIN,
            "K_max": K_MAX,
            "x_min": X_MIN,
            "x_max": X_MAX,
            "N": N,
            "Q": Q,
            "M": M,
        },
    }


def _plot_chi2_dB_exp3(scan: dict, lambdas: np.ndarray, results_dir: str) -> None:
    """Plot chi^2 and d_B vs lambda for Exp3."""
    chi2 = scan["chi2_list"]
    dB = scan["dB_list"]
    valid = ~np.isnan(chi2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.loglog(lambdas[valid], chi2[valid], "o-", color="steelblue", markersize=4)
    for lam_h in HIGHLIGHTED_LAMBDAS:
        ax.axvline(lam_h, color="gray", linestyle="--", alpha=0.7, label=f"$\\lambda={lam_h:.0e}$")
    ax.set_xlabel("$\\lambda$", fontsize=12)
    ax.set_ylabel("Pricing error $\\chi^2$", fontsize=12)
    ax.set_title("Pricing Error vs $\\lambda$\n(Arbitrage-Contaminated)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    valid_dB = ~np.isnan(dB)
    if valid_dB.any():
        # Paper uses direct d_B (not log) for arbitrage case
        ax.semilogx(lambdas[valid_dB], dB[valid_dB], "o-", color="tomato", markersize=4)
    for lam_h in HIGHLIGHTED_LAMBDAS:
        ax.axvline(lam_h, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("$\\lambda$", fontsize=12)
    ax.set_ylabel("Bhattacharyya distance $d_B$", fontsize=12)
    ax.set_title("Bhattacharyya Distance vs $\\lambda$\n(vs $\\phi_M^+$)", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "exp3_chi2_dB.png"), dpi=150)
    plt.close(fig)


def _plot_density_comparison_exp3(
    x: np.ndarray,
    phi_M: np.ndarray,
    phi_M_plus: np.ndarray,
    scan: dict,
    lambdas: np.ndarray,
    results_dir: str,
) -> None:
    """Plot recovered density vs signed mixture and positive part."""
    fig, axes = plt.subplots(1, len(HIGHLIGHTED_LAMBDAS), figsize=(7 * len(HIGHLIGHTED_LAMBDAS), 5))
    if len(HIGHLIGHTED_LAMBDAS) == 1:
        axes = [axes]

    for ax, lam_target in zip(axes, HIGHLIGHTED_LAMBDAS):
        idx = np.argmin(np.abs(lambdas - lam_target))
        phi_rec = scan["phi_list"][idx]
        lam_actual = lambdas[idx]

        ax.plot(x, phi_M, "k--", linewidth=1.5, label="Signed mixture $\\phi_M$", alpha=0.7)
        ax.plot(x, phi_M_plus, "g-", linewidth=1.5, label="$\\phi_M^+ = \\max(0, \\phi_M)$")
        if phi_rec is not None:
            ax.plot(x, phi_rec, "-", color="steelblue", linewidth=2,
                    label=f"Recovered ($\\lambda={lam_actual:.1e}$)")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Terminal price $x$", fontsize=11)
        ax.set_ylabel("Density $\\phi(x)$", fontsize=11)
        ax.set_title(f"Arbitrage-Contaminated Recovery\n$\\lambda = {lam_actual:.1e}$", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "exp3_density_comparison.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_experiment3()
