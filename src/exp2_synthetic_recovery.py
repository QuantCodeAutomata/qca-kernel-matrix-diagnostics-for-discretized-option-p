"""
Experiment 2: Synthetic Exact-Density Recovery from Arbitrage-Free Option Prices.

Three independent synthetic cases:
  A: Normal/Bachelier density
  B: Log-normal/Black-Scholes density
  C: Multimodal Gaussian-mixture density

Validates density recovery accuracy, pricing fit, sparsity, and regularization
trade-off as lambda varies.

Custom implementation — Context7 found no library equivalent (paper methodology).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from typing import Optional

from src.kernel_matrix import build_density_grid, build_call_put_kernel, price_from_density
from src.svd_utils import compute_svd, truncate_svd
from src.density_recovery import (
    run_lambda_scan,
    bhattacharyya_distance,
    sparsity_count,
    compute_chi2_eq15,
)
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


# ---------------------------------------------------------------------------
# Case A: Normal/Bachelier
# ---------------------------------------------------------------------------
CASE_A = {
    "name": "Normal/Bachelier",
    "S0": 0.1,
    "sigma": 0.1,
    "r": 0.05,
    "tau": 1.0,
    "K_min": -0.7,
    "K_max": 0.7,
    "x_min": -0.9,
    "x_max": 0.9,
    "N": 1000,
    "Q": 150,
    "M": 200,
    "highlighted_lambdas": [1e-12, 10**(-7.5)],
    "lambda_lo": 1e-13,
    "lambda_hi": 1e-2,
    "n_lambdas": 50,
}

# Case B: Log-normal/Black-Scholes
CASE_B = {
    "name": "Log-normal/Black-Scholes",
    "S0": 0.5,
    "sigma": 0.2,
    "r": 0.0,
    "tau": 1.0,
    "K_min": 0.01,
    "K_max": 1.0,
    "x_min": 0.0,
    "x_max": 1.5,
    "N": 1000,
    "Q": 150,
    "M": 200,
    "highlighted_lambdas": [10**(-7.5), 1e-3],
    "lambda_lo": 1e-12,
    "lambda_hi": 1e-1,
    "n_lambdas": 50,
}

# Case C: Multimodal Gaussian mixture
CASE_C = {
    "name": "Multimodal Gaussian Mixture",
    "weights": [0.50, 0.45, 0.05],
    "means": [-0.20, 0.15, 0.55],
    "stds": [0.10, 0.15, 0.05],
    "r": 0.05,
    "tau": 1.0,
    "K_min": -0.7,
    "K_max": 0.7,
    "x_min": -0.9,
    "x_max": 0.9,
    "N": 1000,
    "Q": 150,
    "M": 200,
    "highlighted_lambdas": [10**(-8.5), 1e-4],
    "lambda_lo": 1e-12,
    "lambda_hi": 1e-2,
    "n_lambdas": 50,
}

# Sparsity thresholds for Case A
SPARSITY_THRESHOLDS = [1e-3, 1e-2, 1e-1, 1e0]


def run_case_a(results_dir: str) -> dict:
    """
    Case A: Normal/Bachelier density recovery.

    Parameters
    ----------
    results_dir : str
        Output directory.

    Returns
    -------
    results : dict
        Diagnostics and recovered densities.
    """
    cfg = CASE_A
    print(f"\n{'='*60}")
    print(f"Case A: {cfg['name']}")
    print(f"  S0={cfg['S0']}, sigma={cfg['sigma']}, r={cfg['r']}, tau={cfg['tau']}")
    print(f"  K in [{cfg['K_min']}, {cfg['K_max']}], x in [{cfg['x_min']}, {cfg['x_max']}]")
    print(f"  N={cfg['N']}, Q={cfg['Q']}, M={cfg['M']}")
    print(f"{'='*60}")

    # Build density grid
    x, dx = build_density_grid(cfg["x_min"], cfg["x_max"], cfg["N"])

    # True density
    phi_true = bachelier_density(x, cfg["S0"], cfg["sigma"], cfg["r"], cfg["tau"])

    # Strikes and prices
    strikes = np.linspace(cfg["K_min"], cfg["K_max"], cfg["M"])
    call_prices = bachelier_call(cfg["S0"], strikes, cfg["sigma"], cfg["r"], cfg["tau"])
    put_prices = bachelier_put(cfg["S0"], strikes, cfg["sigma"], cfg["r"], cfg["tau"])
    Pr = np.concatenate([call_prices, put_prices])

    # Build kernel matrix
    G = build_call_put_kernel(x, dx, strikes, cfg["r"], cfg["tau"])

    # SVD and truncation
    U, s, Vt = compute_svd(G, full_matrices=False)
    U_tilde, S_tilde, V_tilde = truncate_svd(U, s, Vt, cfg["Q"])

    # Lambda scan
    lambdas = np.logspace(
        np.log10(cfg["lambda_lo"]), np.log10(cfg["lambda_hi"]), cfg["n_lambdas"]
    )
    scan = run_lambda_scan(Pr, U_tilde, S_tilde, V_tilde, x, dx, lambdas, phi_true)

    # Sparsity diagnostics for Case A
    sparsity_results = {}
    for i, lam in enumerate(lambdas):
        pp = scan["phi_prime_list"][i]
        if pp is not None:
            sparsity_results[lam] = sparsity_count(pp, SPARSITY_THRESHOLDS)

    # Plots
    _plot_chi2_dB(scan, cfg["name"], results_dir, "exp2_caseA")
    _plot_density_comparison(
        x, phi_true, scan, lambdas, cfg["highlighted_lambdas"],
        cfg["name"], results_dir, "exp2_caseA"
    )
    _plot_sparsity(lambdas, sparsity_results, cfg["name"], results_dir, "exp2_caseA")

    # Print summary
    _print_summary(cfg["name"], lambdas, scan)

    return {
        "case": "A",
        "x": x,
        "phi_true": phi_true,
        "lambdas": lambdas,
        "scan": scan,
        "sparsity": sparsity_results,
        "singular_values": s,
    }


def run_case_b(results_dir: str) -> dict:
    """
    Case B: Log-normal/Black-Scholes density recovery.

    Parameters
    ----------
    results_dir : str
        Output directory.

    Returns
    -------
    results : dict
        Diagnostics and recovered densities.
    """
    cfg = CASE_B
    print(f"\n{'='*60}")
    print(f"Case B: {cfg['name']}")
    print(f"  S0={cfg['S0']}, sigma={cfg['sigma']}, r={cfg['r']}, tau={cfg['tau']}")
    print(f"  K in [{cfg['K_min']}, {cfg['K_max']}], x in [{cfg['x_min']}, {cfg['x_max']}]")
    print(f"  N={cfg['N']}, Q={cfg['Q']}, M={cfg['M']}")
    print(f"{'='*60}")

    x, dx = build_density_grid(cfg["x_min"], cfg["x_max"], cfg["N"])

    # True density (log-normal, zero for x <= 0)
    phi_true = black_scholes_density(x, cfg["S0"], cfg["sigma"], cfg["r"], cfg["tau"])

    # Strikes and prices
    strikes = np.linspace(cfg["K_min"], cfg["K_max"], cfg["M"])
    call_prices = black_scholes_call(cfg["S0"], strikes, cfg["sigma"], cfg["r"], cfg["tau"])
    put_prices = black_scholes_put(cfg["S0"], strikes, cfg["sigma"], cfg["r"], cfg["tau"])
    Pr = np.concatenate([call_prices, put_prices])

    # Build kernel matrix
    G = build_call_put_kernel(x, dx, strikes, cfg["r"], cfg["tau"])

    # SVD and truncation
    U, s, Vt = compute_svd(G, full_matrices=False)
    U_tilde, S_tilde, V_tilde = truncate_svd(U, s, Vt, cfg["Q"])

    # Lambda scan
    lambdas = np.logspace(
        np.log10(cfg["lambda_lo"]), np.log10(cfg["lambda_hi"]), cfg["n_lambdas"]
    )
    scan = run_lambda_scan(Pr, U_tilde, S_tilde, V_tilde, x, dx, lambdas, phi_true)

    # Plots
    _plot_chi2_dB(scan, cfg["name"], results_dir, "exp2_caseB")
    _plot_density_comparison(
        x, phi_true, scan, lambdas, cfg["highlighted_lambdas"],
        cfg["name"], results_dir, "exp2_caseB"
    )

    _print_summary(cfg["name"], lambdas, scan)

    return {
        "case": "B",
        "x": x,
        "phi_true": phi_true,
        "lambdas": lambdas,
        "scan": scan,
        "singular_values": s,
    }


def run_case_c(results_dir: str) -> dict:
    """
    Case C: Multimodal Gaussian-mixture density recovery.

    Parameters
    ----------
    results_dir : str
        Output directory.

    Returns
    -------
    results : dict
        Diagnostics and recovered densities.
    """
    cfg = CASE_C
    print(f"\n{'='*60}")
    print(f"Case C: {cfg['name']}")
    print(f"  weights={cfg['weights']}, means={cfg['means']}, stds={cfg['stds']}")
    print(f"  r={cfg['r']}, tau={cfg['tau']}")
    print(f"  K in [{cfg['K_min']}, {cfg['K_max']}], x in [{cfg['x_min']}, {cfg['x_max']}]")
    print(f"  N={cfg['N']}, Q={cfg['Q']}, M={cfg['M']}")
    print(f"{'='*60}")

    x, dx = build_density_grid(cfg["x_min"], cfg["x_max"], cfg["N"])

    # True density (Gaussian mixture)
    phi_true = gaussian_mixture_density(x, cfg["weights"], cfg["means"], cfg["stds"])

    # Strikes and prices (using mixture linearity)
    strikes = np.linspace(cfg["K_min"], cfg["K_max"], cfg["M"])
    call_prices = mixture_call_price(
        strikes, cfg["weights"], cfg["means"], cfg["stds"], cfg["r"], cfg["tau"]
    )
    put_prices = mixture_put_price(
        strikes, cfg["weights"], cfg["means"], cfg["stds"], cfg["r"], cfg["tau"]
    )
    Pr = np.concatenate([call_prices, put_prices])

    # Build kernel matrix
    G = build_call_put_kernel(x, dx, strikes, cfg["r"], cfg["tau"])

    # SVD and truncation
    U, s, Vt = compute_svd(G, full_matrices=False)
    U_tilde, S_tilde, V_tilde = truncate_svd(U, s, Vt, cfg["Q"])

    # Lambda scan
    lambdas = np.logspace(
        np.log10(cfg["lambda_lo"]), np.log10(cfg["lambda_hi"]), cfg["n_lambdas"]
    )
    scan = run_lambda_scan(Pr, U_tilde, S_tilde, V_tilde, x, dx, lambdas, phi_true)

    # Plots
    _plot_chi2_dB(scan, cfg["name"], results_dir, "exp2_caseC")
    _plot_density_comparison(
        x, phi_true, scan, lambdas, cfg["highlighted_lambdas"],
        cfg["name"], results_dir, "exp2_caseC"
    )

    _print_summary(cfg["name"], lambdas, scan)

    return {
        "case": "C",
        "x": x,
        "phi_true": phi_true,
        "lambdas": lambdas,
        "scan": scan,
        "singular_values": s,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_chi2_dB(
    scan: dict,
    case_name: str,
    results_dir: str,
    prefix: str,
) -> None:
    """Plot chi^2 and Bhattacharyya distance vs lambda."""
    lambdas = scan["lambdas"]
    chi2 = scan["chi2_list"]
    dB = scan["dB_list"]

    valid = ~np.isnan(chi2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.loglog(lambdas[valid], chi2[valid], "o-", color="steelblue", markersize=4)
    ax.set_xlabel("$\\lambda$", fontsize=12)
    ax.set_ylabel("Pricing error $\\chi^2$", fontsize=12)
    ax.set_title(f"Pricing Error vs $\\lambda$\n({case_name})", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    valid_dB = ~np.isnan(dB)
    if valid_dB.any():
        ax.loglog(lambdas[valid_dB], dB[valid_dB], "o-", color="tomato", markersize=4)
    ax.set_xlabel("$\\lambda$", fontsize=12)
    ax.set_ylabel("Bhattacharyya distance $d_B$", fontsize=12)
    ax.set_title(f"Bhattacharyya Distance vs $\\lambda$\n({case_name})", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{prefix}_chi2_dB.png"), dpi=150)
    plt.close(fig)


def _plot_density_comparison(
    x: np.ndarray,
    phi_true: np.ndarray,
    scan: dict,
    lambdas: np.ndarray,
    highlighted_lambdas: list,
    case_name: str,
    results_dir: str,
    prefix: str,
) -> None:
    """Plot recovered density vs true density at highlighted lambda values."""
    fig, axes = plt.subplots(1, len(highlighted_lambdas), figsize=(7 * len(highlighted_lambdas), 5))
    if len(highlighted_lambdas) == 1:
        axes = [axes]

    for ax, lam_target in zip(axes, highlighted_lambdas):
        # Find closest lambda in scan
        idx = np.argmin(np.abs(lambdas - lam_target))
        phi_rec = scan["phi_list"][idx]
        lam_actual = lambdas[idx]

        ax.plot(x, phi_true, "k-", linewidth=2, label="True density")
        if phi_rec is not None:
            ax.plot(x, phi_rec, "--", color="steelblue", linewidth=2,
                    label=f"Recovered ($\\lambda={lam_actual:.1e}$)")
        ax.set_xlabel("Terminal price $x$", fontsize=11)
        ax.set_ylabel("Density $\\phi(x)$", fontsize=11)
        ax.set_title(f"{case_name}\n$\\lambda = {lam_actual:.1e}$", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{prefix}_density_comparison.png"), dpi=150)
    plt.close(fig)


def _plot_sparsity(
    lambdas: np.ndarray,
    sparsity_results: dict,
    case_name: str,
    results_dir: str,
    prefix: str,
) -> None:
    """Plot sparsity n(|phi'|>a) vs lambda for Case A."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["steelblue", "tomato", "green", "purple"]

    for color, a in zip(colors, SPARSITY_THRESHOLDS):
        counts = []
        lam_vals = []
        for lam in lambdas:
            if lam in sparsity_results:
                counts.append(sparsity_results[lam][a])
                lam_vals.append(lam)
        if counts:
            ax.semilogx(lam_vals, counts, "o-", color=color, label=f"$a={a:.0e}$", markersize=4)

    ax.set_xlabel("$\\lambda$", fontsize=12)
    ax.set_ylabel("$n(|\\phi^\\prime| > a)$", fontsize=12)
    ax.set_title(f"Sparsity of Transformed Coefficients\n({case_name})", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{prefix}_sparsity.png"), dpi=150)
    plt.close(fig)


def _print_summary(case_name: str, lambdas: np.ndarray, scan: dict) -> None:
    """Print summary of scan results."""
    chi2 = scan["chi2_list"]
    dB = scan["dB_list"]
    valid = ~np.isnan(chi2)
    if valid.any():
        best_idx = np.nanargmin(chi2)
        print(f"\n  {case_name} summary:")
        print(f"    Best chi^2 = {chi2[best_idx]:.4e} at lambda = {lambdas[best_idx]:.2e}")
        if not np.isnan(dB[best_idx]):
            print(f"    d_B at best chi^2 = {dB[best_idx]:.4f}")
        # Find preferred lambda (elbow: largest lambda with chi2 near minimum)
        chi2_min = np.nanmin(chi2)
        elbow_mask = valid & (chi2 <= 10 * chi2_min)
        if elbow_mask.any():
            elbow_idx = np.where(elbow_mask)[0][-1]
            print(f"    Preferred lambda (elbow) ≈ {lambdas[elbow_idx]:.2e}")


def run_experiment2(results_dir: str = "results") -> dict:
    """
    Run the full Experiment 2: Synthetic Exact-Density Recovery.

    Parameters
    ----------
    results_dir : str
        Directory to save all outputs.

    Returns
    -------
    all_results : dict
        Results from all three cases.
    """
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Synthetic Exact-Density Recovery")
    print("=" * 70)

    results_a = run_case_a(results_dir)
    results_b = run_case_b(results_dir)
    results_c = run_case_c(results_dir)

    print("\n" + "=" * 70)
    print("Experiment 2 Complete.")
    print("=" * 70)

    return {
        "case_A": results_a,
        "case_B": results_b,
        "case_C": results_c,
    }


if __name__ == "__main__":
    run_experiment2()
