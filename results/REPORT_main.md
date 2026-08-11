# RIH results (main, seeds [0, 1, 2])

## H* distribution

| split | mean | sd | IQR | at ceiling | at zero | degenerate |
|---|---|---|---|---|---|---|
| A_gtrain | 15.71 | 10.29 | 21.0 | 0.216 | 0.027 | False |
| A_val | 15.70 | 10.33 | 21.0 | 0.220 | 0.025 | False |
| A_test | 15.52 | 10.29 | 20.0 | 0.212 | 0.026 | False |
| B_test | 7.75 | 8.68 | 11.0 | 0.052 | 0.227 | False |
| C_test | 4.14 | 4.99 | 5.0 | 0.004 | 0.139 | False |
| D_test | 5.11 | 6.61 | 7.0 | 0.013 | 0.293 | False |

Primary g_phi variant (selected on validation, per seed): **['full', 'no_latent', 'no_latent']**
Best Claim-A ceiling signal: **imagined_disp_spread**

## Predicted vs actual RIH on A-test (mean +/- sd over seeds)

| method | MAE | RMSE | Spearman | R2 |
|---|---|---|---|---|
| Baseline 8: oracle (uses the future) | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| g_phi: no_latent | 6.394 ± 0.068 | 8.369 ± 0.065 | 0.580 ± 0.005 | 0.331 ± 0.004 |
| g_phi: full | 6.457 ± 0.080 | 8.535 ± 0.161 | 0.566 ± 0.005 | 0.304 ± 0.014 |
| g_phi: uncertainty_only | 6.604 ± 0.109 | 8.553 ± 0.141 | 0.553 ± 0.012 | 0.301 ± 0.015 |
| Imagined displacement spread (isotonic) | 7.487 ± 0.131 | 9.047 ± 0.152 | 0.456 ± 0.012 | 0.218 ± 0.011 |
| g_phi: strict_apriori | 8.065 ± 0.087 | 9.603 ± 0.101 | 0.372 ± 0.003 | 0.119 ± 0.002 |
| g_phi: hand_actions_only | 8.274 ± 0.095 | 9.670 ± 0.087 | 0.320 ± 0.005 | 0.107 ± 0.001 |
| Online truncation (disp spread >= 1) | 8.384 ± 0.082 | 9.726 ± 0.095 | 0.298 ± 0.011 | 0.096 ± 0.004 |
| g_phi: latent_only | 8.454 ± 0.077 | 9.885 ± 0.089 | 0.294 ± 0.009 | 0.067 ± 0.005 |
| Baseline 8b: PRIVILEGED true local grid | 8.628 ± 0.067 | 10.080 ± 0.123 | 0.226 ± 0.006 | 0.030 ± 0.006 |
| Ensemble disagreement over rollout (isotonic) | 8.724 ± 0.132 | 9.992 ± 0.142 | 0.203 ± 0.024 | 0.046 ± 0.010 |
| g_phi: actions_only | 8.916 ± 0.104 | 10.181 ± 0.102 | 0.108 ± 0.007 | 0.010 ± 0.003 |
| Baseline 1: fixed horizon (median H*) | 8.975 ± 0.101 | 10.283 ± 0.119 | 0.000 ± 0.000 | -0.010 ± 0.008 |
| Baseline 4: model confidence (isotonic) | 8.991 ± 0.102 | 10.143 ± 0.090 | 0.128 ± 0.006 | 0.017 ± 0.002 |
| Baseline 7b: planned forward count (isotonic) | 9.005 ± 0.107 | 10.188 ± 0.099 | 0.086 ± 0.005 | 0.009 ± 0.001 |
| Online truncation (JS >= 0.1) | 9.022 ± 0.102 | 10.212 ± 0.100 | 0.056 ± 0.025 | 0.004 ± 0.002 |
| Baseline 5: ensemble disagreement (isotonic) | 9.030 ± 0.104 | 10.191 ± 0.092 | 0.103 ± 0.008 | 0.008 ± 0.003 |
| Baseline 6b: latent prior/posterior KL (isotonic) | 9.038 ± 0.097 | 10.208 ± 0.088 | 0.055 ± 0.032 | 0.005 ± 0.003 |
| Baseline 7: visible free cells ahead (isotonic) | 9.049 ± 0.128 | 10.219 ± 0.112 | 0.065 ± 0.014 | 0.003 ± 0.003 |
| Baseline 6a: latent prior std (isotonic) | 9.054 ± 0.102 | 10.232 ± 0.096 | 0.030 ± 0.012 | 0.000 ± 0.001 |
| Baseline 3: current-step error (isotonic) | 9.063 ± 0.107 | 10.241 ± 0.098 | 0.026 ± 0.003 | -0.002 ± 0.002 |
| Baseline 2: random horizon | 11.002 ± 0.047 | 13.423 ± 0.036 | 0.001 ± 0.004 | -0.721 ± 0.026 |

## Claim-A ceilings (isotonic fitted ON TEST -- deliberately optimistic)

| signal | MAE | Spearman |
|---|---|---|
| imagined_disp_spread | 7.456 | 0.459 |
| internal_truncation_disp | 8.364 | 0.299 |
| ensemble_disagreement_mean | 8.690 | 0.211 |
| predictive_entropy_1step | 8.946 | 0.136 |
| ensemble_disagreement_1step | 8.989 | 0.104 |
| n_forward_planned | 8.960 | 0.089 |
| latent_post_prior_kl | 8.991 | 0.079 |
| latent_prior_std | 9.015 | 0.044 |
| internal_truncation_js | 8.992 | 0.072 |
| free_ahead | 9.015 | 0.065 |
| context_onestep_error | 9.028 | 0.024 |

## Claim A: uncertainty vs future error (Spearman, A-test)

| signal | vs E_1 | vs E_5 | vs E_10 | vs E_20 | vs H* |
|---|---|---|---|---|---|
| ensemble_disagreement_1step | +0.207 | +0.115 | +0.076 | +0.035 | -0.079 |
| predictive_entropy_1step | +0.233 | +0.138 | +0.092 | +0.041 | -0.097 |
| latent_prior_std | -0.012 | -0.029 | -0.033 | -0.018 | +0.029 |
| latent_post_prior_kl | +0.189 | +0.065 | +0.051 | +0.013 | -0.050 |
| ensemble_disagreement_mean | +0.050 | +0.109 | +0.150 | +0.197 | -0.204 |
| imagined_disp_spread | +0.113 | +0.269 | +0.373 | +0.400 | -0.455 |
| internal_truncation_disp | -0.073 | -0.159 | -0.234 | -0.299 | +0.299 |
| internal_truncation_js | -0.028 | -0.031 | -0.038 | -0.045 | +0.049 |
| free_ahead | -0.145 | -0.054 | -0.002 | +0.016 | +0.004 |
| n_forward_planned | +0.008 | +0.048 | +0.073 | +0.118 | -0.085 |
| context_onestep_error | +0.053 | +0.008 | -0.013 | -0.030 | +0.016 |

## H3: transfer under distribution shift

| split | actual H* | predicted H* | MAE | Spearman |
|---|---|---|---|---|
| A_test (in-dist) | 15.52 | - | 6.416 | 0.573 |
| B_test | 7.75 | 21.01 | 14.435 | 0.219 |
| C_test | 4.11 | 12.82 | 9.606 | 0.271 |
| D_test | 5.11 | 19.97 | 15.384 | 0.165 |

## Pre-registered verdict

- (a) beats constant on every seed: True (MAE reduction [2.505, 2.631, 2.541], CI lower bounds [2.335, 2.469, 2.384])
- (b) beats best Claim-A ceiling on every seed: True (MAE reduction [1.08, 1.102, 0.936], CI lower bounds [0.978, 1.01, 0.854])
- (b') STRICT a-priori variant (no imagined rollout) beats the ceiling: False (MAE reduction [-0.464, -0.63, -0.735])
- (c) Spearman > 0.5 with CI lower bound > 0.3: True (rho [0.559, 0.587, 0.574], lo [0.539, 0.567, 0.554])
- validation degeneracy triggered: False

### **SUPPORTED**

## H4: adaptive vs fixed planning

| arm | success | steps to goal | imagined states / decision | mean H |
|---|---|---|---|---|
| fixed_3 | 0.917 ± 0.036 | 28.1 | 288 | 3.0 |
| fixed_5 | 0.933 ± 0.032 | 34.5 | 480 | 5.0 |
| fixed_10 | 0.767 ± 0.055 | 46.3 | 960 | 10.0 |
| fixed_20 | 0.517 ± 0.065 | 58.1 | 1920 | 20.0 |
| fixed_30 | 0.367 ± 0.062 | 63.7 | 2880 | 30.0 |
| adaptive | 0.517 ± 0.065 | 60.7 | 2637 | 25.9 |

adaptive vs fixed_5: success delta -0.417 [-0.550, -0.300], compute ratio 5.49x
