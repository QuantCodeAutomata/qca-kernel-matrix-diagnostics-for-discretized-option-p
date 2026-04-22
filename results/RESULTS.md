# Experiment Results: Kernel Matrix Diagnostics for Discretized Option-Pricing Operator

Generated: 2026-04-22 13:16:33

## Experiment 1: Kernel Matrix Diagnostics

### Parameters
- Risk-free rate r = 0.05
- Time to maturity tau = 1.0
- Condition study: N=10000, x ∈ [0.5, 1.5], K ∈ [0.7, 1.3]
- SV decay study: M=25, x ∈ [0.5, 1.5], K ∈ [0.7, 1.3]

### Condition Number Study (N=10000)
| M | Condition Number |
|---|---|
| 5 | 5.5925e+01 |
| 10 | 2.9118e+02 |
| 15 | 7.0603e+02 |
| 20 | 1.2965e+03 |
| 25 | 2.0620e+03 |
| 30 | 3.0026e+03 |
| 40 | 5.4087e+03 |
| 50 | 8.5147e+03 |
| 75 | 1.9342e+04 |
| 100 | 3.4543e+04 |
| 150 | 7.8068e+04 |
| 200 | 1.3909e+05 |

**Fitted power-law exponent k = 2.091** (target ≈ 2)

### Singular Value Decay Study (M=25)
| N | Decay Exponent k |
|---|---|
| 500 | -1.924 |
| 1000 | -1.924 |
| 2000 | -1.924 |
| 5000 | -1.924 |
| 10000 | -1.924 |

**Target decay exponent ≈ 2.7**

**Runtime:** 2.0s

## Experiment 2: Synthetic Exact-Density Recovery

### Case A: Normal/Bachelier
- Best chi² = 1.8171e-09 at λ = 1.10e-09
- Bhattacharyya distance at best λ: 0.0000
- Preferred λ (elbow): 4.09e-08

### Case B: Log-normal/Black-Scholes
- Best chi² = 1.3437e-09 at λ = 3.91e-09
- Bhattacharyya distance at best λ: 0.0001
- Preferred λ (elbow): 3.09e-08

### Case C: Multimodal Gaussian Mixture
- Best chi² = 2.8454e-09 at λ = 1.84e-09
- Bhattacharyya distance at best λ: 0.0001
- Preferred λ (elbow): 3.09e-08

**Runtime:** 350.8s

## Experiment 3: Arbitrage-Contaminated Density Recovery

### Signed Mixture Parameters
- Weights: [0.55, -0.2, 0.65]
- Means: [0.8, 1.15, 1.35]
- Stds: [0.1, 0.07, 0.2]

### Feasibility
- Feasible solutions: 50/50
- chi² range: [1.4932e-05, 2.5555e-02]
- chi² > 0 confirms exact recovery is impossible (as expected)

### Highlighted Lambda Results
- λ=1e-07: chi²=1.6044e-05, d_B=0.0084, min(φ)=-1.3575e-05, mass=1.000000
- λ=1e-05: chi²=7.3251e-05, d_B=-0.0072, min(φ)=-6.1114e-06, mass=1.000000

**Runtime:** 114.5s

## Experiment 4: Implied-Volatility De-Arbitraging

- Forward price F = 1.0513 (S0=1.0, r=0.05, tau=1)
- Original IV inversions: 200 valid, 0 failed

### De-Arbitraged IV Results
- λ=1e-07: 196 valid IV inversions
- λ=1e-05: 200 valid IV inversions

**Runtime:** 5.7s

## Experiment 5: Empirical SPX Validation (2018-02-05)

### Market Parameters
- SPX spot: 2648.94
- Forward: 2653.19
- tau: 0.10685 years
- r: 0.015
- Scaling: 1/1000.0
- N=1000, Q=70, M=101

### Recovery Results
- λ=1e-07: chi²=3.1604e-06, 101 valid IV inversions
- λ=1e-04: chi²=9.7030e-04, 101 valid IV inversions

**Runtime:** 25.3s

## Output Files

### Plots
- `exp1_condition_number_vs_M.png`
- `exp1_singular_value_decay.png`
- `exp1_truncated_condition_numbers.png`
- `exp2_caseA_chi2_dB.png`
- `exp2_caseA_density_comparison.png`
- `exp2_caseA_sparsity.png`
- `exp2_caseB_chi2_dB.png`
- `exp2_caseB_density_comparison.png`
- `exp2_caseC_chi2_dB.png`
- `exp2_caseC_density_comparison.png`
- `exp3_chi2_dB.png`
- `exp3_density_comparison.png`
- `exp4_density_and_iv.png`
- `exp4_iv_comparison.png`
- `exp5_chi2_vs_lambda.png`
- `exp5_density.png`
- `exp5_iv_smile.png`
