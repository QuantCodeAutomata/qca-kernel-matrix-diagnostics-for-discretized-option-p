"""
Experiment 1: Kernel Matrix Diagnostics for Discretized Option-Pricing Operator.

Analyzes the numerical conditioning of the discretized option-pricing kernel G.
Studies:
  (a) Condition number growth as number of strikes M varies (N=10000 fixed).
  (b) Singular-value decay pattern for fixed M=25, varying N.

Custom implementation — Context7 found no library equivalent (paper methodology).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from pathlib import Path

from src.kernel_matrix import build_density_grid, build_kernel_matrix
from src.svd_utils import compute_svd, condition_number, fit_power_law, truncated_condition_number


# ---------------------------------------------------------------------------
# Experiment parameters (documented for reproducibility)
# ---------------------------------------------------------------------------

# Fixed parameters
R = 0.05          # risk-free rate
TAU = 1.0         # time to maturity (1 year)

# Condition-number sub-study
COND_N = 10000                          # density grid size (fixed)
COND_X_MIN = 0.5                        # density support left endpoint
COND_X_MAX = 1.5                        # density support right endpoint
COND_K_MIN = 0.7                        # strike range left endpoint
COND_K_MAX = 1.3                        # strike range right endpoint
COND_M_VALUES = np.array([5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200])

# Singular-value decay sub-study
SV_M = 25                               # fixed number of strikes
SV_K_MIN = 0.7                          # strike range
SV_K_MAX = 1.3
SV_X_MIN = 0.5                          # density support
SV_X_MAX = 1.5
SV_N_VALUES = np.array([500, 1000, 2000, 5000, 10000])

# Power-law fit windows (index ranges, documented)
COND_FIT_RANGE = (0, len(COND_M_VALUES))   # use all M values
SV_FIT_RANGE = (0, 15)                      # fit first 15 singular values


def run_condition_number_study(results_dir: str) -> dict:
    """
    Sub-study (a): Condition number vs. number of strikes M.

    Fix N=10000, vary M over COND_M_VALUES. For each M, build G with
    uniformly spaced call strikes and compute the condition number.

    Parameters
    ----------
    results_dir : str
        Directory to save plots and data.

    Returns
    -------
    results : dict
        Contains M_values, condition_numbers, fitted exponent k, log_a.
    """
    print("=" * 60)
    print("Exp1 Sub-study (a): Condition number vs. M")
    print(f"  N={COND_N}, x in [{COND_X_MIN}, {COND_X_MAX}]")
    print(f"  K in [{COND_K_MIN}, {COND_K_MAX}]")
    print(f"  r={R}, tau={TAU}")
    print("=" * 60)

    x, dx = build_density_grid(COND_X_MIN, COND_X_MAX, COND_N)
    cond_numbers = []

    for M in COND_M_VALUES:
        strikes = np.linspace(COND_K_MIN, COND_K_MAX, int(M))
        option_types = ["call"] * int(M)
        G = build_kernel_matrix(x, dx, strikes, R, TAU, option_types)

        # Compute SVD — using scipy.linalg.svd (Context7 confirmed)
        _, s, _ = compute_svd(G, full_matrices=False)
        cond = condition_number(s)
        cond_numbers.append(cond)
        print(f"  M={M:4d}: cond(G) = {cond:.4e}")

    cond_numbers = np.array(cond_numbers)

    # Fit power law C ~ M^k in log-log space
    k, log_a = fit_power_law(
        COND_M_VALUES.astype(float),
        cond_numbers,
        fit_range=COND_FIT_RANGE,
    )
    print(f"\nFitted power law: C ~ M^{k:.3f}  (target k≈2)")
    print(f"  log(a) = {log_a:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(COND_M_VALUES, cond_numbers, "o-", color="steelblue", label="Condition number")
    M_fit = np.linspace(COND_M_VALUES[0], COND_M_VALUES[-1], 200)
    ax.loglog(
        M_fit,
        np.exp(log_a) * M_fit**k,
        "--",
        color="tomato",
        label=f"Power-law fit: $C \\propto M^{{{k:.2f}}}$",
    )
    ax.set_xlabel("Number of strikes M", fontsize=12)
    ax.set_ylabel("Condition number $C$", fontsize=12)
    ax.set_title("Condition Number vs. Number of Strikes (N=10000)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "exp1_condition_number_vs_M.png"), dpi=150)
    plt.close(fig)

    return {
        "M_values": COND_M_VALUES,
        "condition_numbers": cond_numbers,
        "fitted_exponent_k": k,
        "fitted_log_a": log_a,
    }


def run_singular_value_decay_study(results_dir: str) -> dict:
    """
    Sub-study (b): Singular-value decay for fixed M=25, varying N.

    Fix M=25 uniformly spaced call strikes. For each N, build G and compute
    normalized singular values s_i/s_1. Fit initial decay to inverse-power form.

    Parameters
    ----------
    results_dir : str
        Directory to save plots and data.

    Returns
    -------
    results : dict
        Contains N_values, singular_spectra, fitted exponent k.
    """
    print("\n" + "=" * 60)
    print("Exp1 Sub-study (b): Singular-value decay, M=25")
    print(f"  M={SV_M}, K in [{SV_K_MIN}, {SV_K_MAX}]")
    print(f"  x in [{SV_X_MIN}, {SV_X_MAX}]")
    print(f"  r={R}, tau={TAU}")
    print("=" * 60)

    strikes = np.linspace(SV_K_MIN, SV_K_MAX, SV_M)
    option_types = ["call"] * SV_M

    sv_spectra = {}
    fitted_exponents = {}

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(SV_N_VALUES)))

    for idx, N in enumerate(SV_N_VALUES):
        x, dx = build_density_grid(SV_X_MIN, SV_X_MAX, int(N))
        G = build_kernel_matrix(x, dx, strikes, R, TAU, option_types)

        _, s, _ = compute_svd(G, full_matrices=False)
        s_norm = s / s[0]  # normalize by s_1
        sv_spectra[N] = s_norm

        # Fit initial decay region (indices 1..SV_FIT_RANGE[1])
        i0, i1 = SV_FIT_RANGE
        i1 = min(i1, len(s_norm) - 1)
        indices = np.arange(1, i1 + 1, dtype=float)  # 1-based index
        sv_fit = s_norm[1 : i1 + 1]

        # Fit s_i/s_1 ~ a * i^(-k) in log-log space
        k, log_a = fit_power_law(indices, sv_fit)
        fitted_exponents[N] = k
        print(f"  N={N:6d}: rank={len(s)}, s_1={s[0]:.4e}, "
              f"s_min={s[-1]:.4e}, decay exponent k={k:.3f}")

        # Plot normalized singular values
        ax.semilogy(
            np.arange(1, len(s_norm) + 1),
            s_norm,
            "-",
            color=colors[idx],
            label=f"N={N}",
            linewidth=1.5,
        )

    # Overlay power-law fit for largest N
    N_ref = SV_N_VALUES[-1]
    s_ref = sv_spectra[N_ref]
    i0, i1 = SV_FIT_RANGE
    i1 = min(i1, len(s_ref) - 1)
    indices_ref = np.arange(1, i1 + 1, dtype=float)
    k_ref, log_a_ref = fit_power_law(indices_ref, s_ref[1 : i1 + 1])
    i_plot = np.linspace(1, i1, 100)
    ax.semilogy(
        i_plot,
        np.exp(log_a_ref) * i_plot**(-k_ref),
        "--",
        color="tomato",
        linewidth=2,
        label=f"Power-law fit: $s_i/s_1 \\propto i^{{-{k_ref:.2f}}}$ (N={N_ref})",
    )

    ax.set_xlabel("Singular value index $i$", fontsize=12)
    ax.set_ylabel("Normalized singular value $s_i / s_1$", fontsize=12)
    ax.set_title(f"Singular Value Decay (M={SV_M} calls)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim([1, SV_M])
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "exp1_singular_value_decay.png"), dpi=150)
    plt.close(fig)

    return {
        "N_values": SV_N_VALUES,
        "sv_spectra": sv_spectra,
        "fitted_exponents": fitted_exponents,
    }


def run_truncated_condition_numbers(results_dir: str) -> dict:
    """
    Optional: Compute truncated condition numbers C_tilde = s_1/s_Q for
    selected truncation ranks Q to illustrate conditioning improvement.

    Parameters
    ----------
    results_dir : str
        Directory to save plots and data.

    Returns
    -------
    results : dict
        Contains Q_values, truncated_cond_numbers.
    """
    print("\n" + "=" * 60)
    print("Exp1 Optional: Truncated condition numbers")
    print("=" * 60)

    M = 50
    strikes = np.linspace(COND_K_MIN, COND_K_MAX, M)
    option_types = ["call"] * M
    x, dx = build_density_grid(COND_X_MIN, COND_X_MAX, COND_N)
    G = build_kernel_matrix(x, dx, strikes, R, TAU, option_types)
    _, s, _ = compute_svd(G, full_matrices=False)

    Q_values = np.arange(1, min(M, len(s)) + 1)
    trunc_conds = np.array([truncated_condition_number(s, int(Q)) for Q in Q_values])

    print(f"  Full condition number: {condition_number(s):.4e}")
    for Q in [5, 10, 20, 30, 50]:
        if Q <= len(s):
            print(f"  C_tilde(Q={Q}): {truncated_condition_number(s, Q):.4e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(Q_values, trunc_conds, "o-", color="steelblue")
    ax.set_xlabel("Truncation rank Q", fontsize=12)
    ax.set_ylabel("Truncated condition number $\\tilde{C} = s_1/s_Q$", fontsize=12)
    ax.set_title(f"Truncated Condition Number vs. Q (M={M}, N={COND_N})", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "exp1_truncated_condition_numbers.png"), dpi=150)
    plt.close(fig)

    return {
        "Q_values": Q_values,
        "truncated_cond_numbers": trunc_conds,
        "full_cond": condition_number(s),
    }


def run_experiment1(results_dir: str = "results") -> dict:
    """
    Run the full Experiment 1: Kernel Matrix Diagnostics.

    Parameters
    ----------
    results_dir : str
        Directory to save all outputs.

    Returns
    -------
    all_results : dict
        Combined results from all sub-studies.
    """
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Kernel Matrix Diagnostics")
    print("=" * 70)
    print(f"Parameters: r={R}, tau={TAU}")
    print(f"Condition study: N={COND_N}, K in [{COND_K_MIN},{COND_K_MAX}], "
          f"x in [{COND_X_MIN},{COND_X_MAX}]")
    print(f"SV decay study: M={SV_M}, K in [{SV_K_MIN},{SV_K_MAX}], "
          f"x in [{SV_X_MIN},{SV_X_MAX}]")

    cond_results = run_condition_number_study(results_dir)
    sv_results = run_singular_value_decay_study(results_dir)
    trunc_results = run_truncated_condition_numbers(results_dir)

    all_results = {
        "condition_study": cond_results,
        "sv_decay_study": sv_results,
        "truncated_cond": trunc_results,
        "parameters": {
            "r": R,
            "tau": TAU,
            "cond_N": COND_N,
            "cond_x_min": COND_X_MIN,
            "cond_x_max": COND_X_MAX,
            "cond_K_min": COND_K_MIN,
            "cond_K_max": COND_K_MAX,
            "sv_M": SV_M,
            "sv_x_min": SV_X_MIN,
            "sv_x_max": SV_X_MAX,
            "sv_K_min": SV_K_MIN,
            "sv_K_max": SV_K_MAX,
        },
    }

    print("\n" + "=" * 70)
    print("Experiment 1 Summary:")
    print(f"  Condition number power-law exponent: k = {cond_results['fitted_exponent_k']:.3f} (target ≈ 2)")
    print(f"  SV decay exponents by N:")
    for N, k in sv_results["fitted_exponents"].items():
        print(f"    N={N}: k = {k:.3f} (target ≈ 2.7)")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    run_experiment1()
