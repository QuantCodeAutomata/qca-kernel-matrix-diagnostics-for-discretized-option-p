# Repository: qca-kernel-matrix-diagnostics-for-discretized-option-p

## Overview
This repository implements experiments from a paper on kernel matrix diagnostics for discretized option-pricing operators. The core methodology involves:
1. Constructing a discretized pricing kernel matrix G using trapezoidal quadrature
2. Analyzing its numerical conditioning via SVD
3. Recovering terminal densities from option prices via truncated-SVD + constrained L1-regularized optimization

## Key Libraries
- `scipy.linalg.svd` — SVD computation (Context7 confirmed)
- `cvxpy` — Constrained L1-regularized convex optimization (Context7 confirmed)
- `numpy` — Numerical arrays and linear algebra
- `matplotlib` / `seaborn` — Visualization
- `massive` — Financial data retrieval (SPX data for Exp5)

## Repository Structure
```
src/
  kernel_matrix.py      # Pricing kernel construction (G matrix)
  svd_utils.py          # SVD truncation and condition number utilities
  density_recovery.py   # Constrained L1-regularized optimization pipeline
  option_pricing.py     # Analytic option pricing (Bachelier, BS, Black)
  implied_vol.py        # Implied volatility inversion
  diagnostics.py        # Condition number and singular value diagnostics
  exp1_kernel_diagnostics.py
  exp2_synthetic_recovery.py
  exp3_arbitrage_contaminated.py
  exp4_implied_vol_dearbitrage.py
  exp5_spx_empirical.py
tests/
  test_exp1.py
  test_exp2.py
  test_exp3.py
  test_exp4.py
  test_exp5.py
results/
  RESULTS.md
  *.png (plots)
data/
  spx_ivol_20180205.csv  (SPX implied vol data for Exp5)
```

## Key Mathematical Conventions
- Pricing kernel row i: `(1/2 g_i(x_1), g_i(x_2), ..., g_i(x_{N-1}), 1/2 g_i(x_N))`
- Call payoff: `g_i(x) = Delta_x * exp(-r*tau) * max(0, x - K_i)`
- Put payoff: `g_i(x) = Delta_x * exp(-r*tau) * max(0, K_i - x)`
- Optimization: `min 1/2 ||Pr - U_tilde S_tilde phi'||^2 + lambda ||phi'||_1`
  subject to `V_tilde phi' >= 0` and `trapz(V_tilde phi') = 1`
- Bhattacharyya distance: `d_B = -log(sum(sqrt(phi_true * phi_recovered) * dx))`

## Parameter Defaults
- Exp1: N=10000, M varies; M=25 fixed for SV decay study
- Exp2: N=1000, Q=150 for all three cases
- Exp3: N=1000, Q=150, support [0.1, 2.2]
- Exp4: Same as Exp3, S0=1.0 for IV inversion
- Exp5: N=1000, Q=70, support [1.4, 3.4], strikes/forward scaled by 1/1000

## Notes
- Always use double precision (float64)
- Sort singular values in descending order
- Use trapezoidal endpoint half-weights exactly
- CVXPY solver: SCS or CLARABEL for large problems, ECOS for smaller ones
- SPX data for Exp5 is embedded in the code (Table 11 from reference [10])
- Feasibility tolerance for Exp3: min(phi) >= -1e-4 (SCS solver numerical precision)
- Bhattacharyya distance can be negative when comparator phi_M^+ is not normalized (Exp3)

## Experiment Results Summary
- Exp1: Condition number power-law exponent k=2.091 (target ≈ 2) ✓
- Exp2: All three cases recover density with d_B ≈ 0 at best lambda ✓
- Exp3: 50/50 feasible solutions despite arbitrage-contaminated inputs ✓
- Exp4: 196-200 valid IV inversions for de-arbitraged smile ✓
- Exp5: SPX smile reconstructed with chi²=3.16e-6 at lambda=1e-7 ✓

## Running
```bash
# Run all experiments
python src/run_all_experiments.py

# Run tests
python -m pytest tests/ -v
```
