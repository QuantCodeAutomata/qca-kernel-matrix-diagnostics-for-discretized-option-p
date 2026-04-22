"""
Main runner for all experiments.

Executes all five experiments in sequence and saves results to the results/ directory.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exp1_kernel_diagnostics import run_experiment1
from src.exp2_synthetic_recovery import run_experiment2
from src.exp3_arbitrage_contaminated import run_experiment3
from src.exp4_implied_vol_dearbitrage import run_experiment4
from src.exp5_spx_empirical import run_experiment5


RESULTS_DIR = "results"


def save_results_md(all_results: dict, elapsed_times: dict) -> None:
    """Save experiment results summary to RESULTS.md."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    md_path = os.path.join(RESULTS_DIR, "RESULTS.md")

    with open(md_path, "w") as f:
        f.write("# Experiment Results: Kernel Matrix Diagnostics for Discretized Option-Pricing Operator\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Experiment 1
        f.write("## Experiment 1: Kernel Matrix Diagnostics\n\n")
        if "exp1" in all_results:
            r1 = all_results["exp1"]
            params = r1["parameters"]
            f.write("### Parameters\n")
            f.write(f"- Risk-free rate r = {params['r']}\n")
            f.write(f"- Time to maturity tau = {params['tau']}\n")
            f.write(f"- Condition study: N={params['cond_N']}, "
                    f"x ∈ [{params['cond_x_min']}, {params['cond_x_max']}], "
                    f"K ∈ [{params['cond_K_min']}, {params['cond_K_max']}]\n")
            f.write(f"- SV decay study: M={params['sv_M']}, "
                    f"x ∈ [{params['sv_x_min']}, {params['sv_x_max']}], "
                    f"K ∈ [{params['sv_K_min']}, {params['sv_K_max']}]\n\n")

            f.write("### Condition Number Study (N=10000)\n")
            cond = r1["condition_study"]
            f.write(f"| M | Condition Number |\n|---|---|\n")
            for M, C in zip(cond["M_values"], cond["condition_numbers"]):
                f.write(f"| {M} | {C:.4e} |\n")
            f.write(f"\n**Fitted power-law exponent k = {cond['fitted_exponent_k']:.3f}** (target ≈ 2)\n\n")

            f.write("### Singular Value Decay Study (M=25)\n")
            sv = r1["sv_decay_study"]
            f.write(f"| N | Decay Exponent k |\n|---|---|\n")
            for N, k in sv["fitted_exponents"].items():
                f.write(f"| {N} | {k:.3f} |\n")
            f.write(f"\n**Target decay exponent ≈ 2.7**\n\n")

            f.write(f"**Runtime:** {elapsed_times.get('exp1', 0):.1f}s\n\n")

        # Experiment 2
        f.write("## Experiment 2: Synthetic Exact-Density Recovery\n\n")
        if "exp2" in all_results:
            r2 = all_results["exp2"]
            for case_key, case_name in [("case_A", "Normal/Bachelier"),
                                         ("case_B", "Log-normal/Black-Scholes"),
                                         ("case_C", "Multimodal Gaussian Mixture")]:
                if case_key in r2:
                    case = r2[case_key]
                    scan = case["scan"]
                    chi2 = scan["chi2_list"]
                    dB = scan["dB_list"]
                    lambdas = scan["lambdas"]
                    valid = ~np.isnan(chi2)

                    f.write(f"### Case {case_key[-1]}: {case_name}\n")
                    if valid.any():
                        best_idx = np.nanargmin(chi2)
                        f.write(f"- Best chi² = {chi2[best_idx]:.4e} at λ = {lambdas[best_idx]:.2e}\n")
                        if not np.isnan(dB[best_idx]):
                            f.write(f"- Bhattacharyya distance at best λ: {dB[best_idx]:.4f}\n")
                        # Elbow lambda
                        chi2_min = np.nanmin(chi2)
                        elbow_mask = valid & (chi2 <= 10 * chi2_min)
                        if elbow_mask.any():
                            elbow_idx = np.where(elbow_mask)[0][-1]
                            f.write(f"- Preferred λ (elbow): {lambdas[elbow_idx]:.2e}\n")
                    f.write("\n")

            f.write(f"**Runtime:** {elapsed_times.get('exp2', 0):.1f}s\n\n")

        # Experiment 3
        f.write("## Experiment 3: Arbitrage-Contaminated Density Recovery\n\n")
        if "exp3" in all_results:
            r3 = all_results["exp3"]
            scan = r3["scan"]
            chi2 = scan["chi2_list"]
            dB = scan["dB_list"]
            lambdas = r3["lambdas"]
            valid = ~np.isnan(chi2)

            f.write("### Signed Mixture Parameters\n")
            params = r3["parameters"]
            f.write(f"- Weights: {params['weights']}\n")
            f.write(f"- Means: {params['means']}\n")
            f.write(f"- Stds: {params['stds']}\n\n")

            f.write("### Feasibility\n")
            n_feasible = sum(fc["feasible"] for fc in r3["feasibility_checks"])
            f.write(f"- Feasible solutions: {n_feasible}/{len(lambdas)}\n")
            f.write(f"- chi² range: [{np.nanmin(chi2):.4e}, {np.nanmax(chi2):.4e}]\n")
            f.write(f"- chi² > 0 confirms exact recovery is impossible (as expected)\n\n")

            f.write("### Highlighted Lambda Results\n")
            from src.exp3_arbitrage_contaminated import HIGHLIGHTED_LAMBDAS
            for lam_target in HIGHLIGHTED_LAMBDAS:
                idx = np.argmin(np.abs(lambdas - lam_target))
                phi_rec = scan["phi_list"][idx]
                if phi_rec is not None:
                    fc = r3["feasibility_checks"][idx]
                    f.write(f"- λ={lam_target:.0e}: chi²={chi2[idx]:.4e}, "
                            f"d_B={dB[idx]:.4f}, "
                            f"min(φ)={fc['min_phi']:.4e}, "
                            f"mass={fc['mass']:.6f}\n")

            f.write(f"\n**Runtime:** {elapsed_times.get('exp3', 0):.1f}s\n\n")

        # Experiment 4
        f.write("## Experiment 4: Implied-Volatility De-Arbitraging\n\n")
        if "exp4" in all_results:
            r4 = all_results["exp4"]
            f.write(f"- Forward price F = {r4['F']:.4f} (S0=1.0, r=0.05, tau=1)\n")
            f.write(f"- Original IV inversions: {r4['valid_otm_orig'].sum()} valid, "
                    f"{(~r4['valid_otm_orig']).sum()} failed\n\n")

            f.write("### De-Arbitraged IV Results\n")
            from src.exp4_implied_vol_dearbitrage import HIGHLIGHTED_LAMBDAS as HL4
            for lam in HL4:
                rec = r4["recovered"].get(lam)
                if rec is not None:
                    n_valid = (~np.isnan(rec["iv_otm"])).sum()
                    f.write(f"- λ={lam:.0e}: {n_valid} valid IV inversions\n")

            f.write(f"\n**Runtime:** {elapsed_times.get('exp4', 0):.1f}s\n\n")

        # Experiment 5
        f.write("## Experiment 5: Empirical SPX Validation (2018-02-05)\n\n")
        if "exp5" in all_results:
            r5 = all_results["exp5"]
            params = r5["parameters"]
            f.write("### Market Parameters\n")
            f.write(f"- SPX spot: {params['SPX_SPOT']:.2f}\n")
            f.write(f"- Forward: {params['SPX_F']:.2f}\n")
            f.write(f"- tau: {params['SPX_TAU']:.5f} years\n")
            f.write(f"- r: {params['SPX_R']}\n")
            f.write(f"- Scaling: 1/{params['SCALE']}\n")
            f.write(f"- N={params['N']}, Q={params['Q']}, M={params['M']}\n\n")

            f.write("### Recovery Results\n")
            from src.exp5_spx_empirical import HIGHLIGHTED_LAMBDAS as HL5
            for lam in HL5:
                rec = r5["recovered"].get(lam)
                if rec is not None:
                    n_valid = (~np.isnan(rec["iv_otm"])).sum()
                    f.write(f"- λ={lam:.0e}: chi²={rec['chi2']:.4e}, "
                            f"{n_valid} valid IV inversions\n")

            f.write(f"\n**Runtime:** {elapsed_times.get('exp5', 0):.1f}s\n\n")

        # Output files
        f.write("## Output Files\n\n")
        f.write("### Plots\n")
        for fname in sorted(os.listdir(RESULTS_DIR)):
            if fname.endswith(".png"):
                f.write(f"- `{fname}`\n")

    print(f"\nResults saved to {md_path}")


def main() -> None:
    """Run all experiments and save results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}
    elapsed_times = {}

    print("\n" + "=" * 70)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 70)

    # Experiment 1
    t0 = time.time()
    try:
        all_results["exp1"] = run_experiment1(RESULTS_DIR)
        elapsed_times["exp1"] = time.time() - t0
        print(f"✓ Experiment 1 completed in {elapsed_times['exp1']:.1f}s")
    except Exception as e:
        print(f"✗ Experiment 1 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Experiment 2
    t0 = time.time()
    try:
        all_results["exp2"] = run_experiment2(RESULTS_DIR)
        elapsed_times["exp2"] = time.time() - t0
        print(f"✓ Experiment 2 completed in {elapsed_times['exp2']:.1f}s")
    except Exception as e:
        print(f"✗ Experiment 2 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Experiment 3
    t0 = time.time()
    try:
        all_results["exp3"] = run_experiment3(RESULTS_DIR)
        elapsed_times["exp3"] = time.time() - t0
        print(f"✓ Experiment 3 completed in {elapsed_times['exp3']:.1f}s")
    except Exception as e:
        print(f"✗ Experiment 3 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Experiment 4
    t0 = time.time()
    try:
        all_results["exp4"] = run_experiment4(RESULTS_DIR)
        elapsed_times["exp4"] = time.time() - t0
        print(f"✓ Experiment 4 completed in {elapsed_times['exp4']:.1f}s")
    except Exception as e:
        print(f"✗ Experiment 4 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Experiment 5
    t0 = time.time()
    try:
        all_results["exp5"] = run_experiment5(RESULTS_DIR)
        elapsed_times["exp5"] = time.time() - t0
        print(f"✓ Experiment 5 completed in {elapsed_times['exp5']:.1f}s")
    except Exception as e:
        print(f"✗ Experiment 5 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    save_results_md(all_results, elapsed_times)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    total = sum(elapsed_times.values())
    print(f"Total runtime: {total:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
