device: cuda
--- main: seed 0 ---
--- main: seed 1 ---
--- main: seed 2 ---

========================================================================
 H6 main  (mean over 3 seeds)
========================================================================

training anchors per regime: {'A': 22200, 'A+B': 28860, 'A+C': 28834, 'A+D': 28860, 'A+B+C': 35494}
test anchors per config: {'A': 11100, 'B': 2220, 'C': 2202, 'D': 2220}

### Predicted vs actual H* (bold = config unseen in training)

| regime | test A | test B | test C | test D |
|---|---|---|---|---|
| *actual* | 15.60 | 8.08 | 3.95 | 4.89 |
| A | 15.68 | **21.60** | **13.83** | **21.21** |
| A+B | 15.46 | 7.82 | **12.08** | **5.07** |
| A+C | 14.76 | **9.88** | 6.10 | **8.64** |
| A+D | 15.59 | **6.89** | **12.01** | 4.54 |
| A+B+C | 14.59 | 7.73 | 6.25 | **5.23** |

### MAE (bold = unseen)

| regime | test A | test B | test C | test D |
|---|---|---|---|---|
| A | 6.394 | **14.747** | **10.609** | **16.768** |
| A+B | 6.407 | 4.658 | **9.044** | **3.540** |
| A+C | 6.506 | **7.007** | 4.142 | **6.155** |
| A+D | 6.406 | **4.871** | **9.009** | 2.705 |
| A+B+C | 6.490 | 4.640 | 4.220 | **3.584** |

### SPEARMAN (bold = unseen)

| regime | test A | test B | test C | test D |
|---|---|---|---|---|
| A | +0.580 | **+0.218** | **+0.213** | **+0.135** |
| A+B | +0.584 | +0.755 | **+0.306** | **+0.771** |
| A+C | +0.576 | **+0.461** | +0.479 | **+0.510** |
| A+D | +0.580 | **+0.699** | **+0.311** | +0.847 |
| A+B+C | +0.577 | +0.754 | +0.477 | **+0.766** |

### Transfer T_MAE(regime -> config) = MAE(g_A) - MAE(g_regime); positive = improvement over IID-only

| regime | test A | test B | test C | test D |
|---|---|---|---|---|
| A+B | -0.013 [-0.05,+0.03] | +10.089 [+9.36,+10.81] | **+1.565 [+1.16,+2.00]** | **+13.228 [+12.43,+13.98]** |
| A+C | -0.112 [-0.17,-0.05] | **+7.741 [+7.09,+8.39]** | +6.467 [+5.87,+7.07] | **+10.613 [+9.98,+11.22]** |
| A+D | -0.012 [-0.05,+0.03] | **+9.876 [+9.07,+10.64]** | **+1.599 [+1.17,+2.06]** | +14.063 [+13.32,+14.79] |
| A+B+C | -0.096 [-0.16,-0.03] | +10.107 [+9.37,+10.83] | +6.389 [+5.81,+6.97] | **+13.185 [+12.40,+13.92]** |

### Rank stability of uncertainty features

| feature | rho_A | rho_B | rho_C | rho_D | mean|rho_OOD| | RS = Var_D[rho] |
|---|---|---|---|---|---|---|
| imagined_disp_spread | -0.455 | -0.096 | -0.179 | -0.096 | 0.124 | 0.0227 |
| internal_truncation_disp | +0.299 | +0.081 | +0.188 | +0.099 | 0.123 | 0.0083 |
| ensemble_disagreement_mean | -0.204 | -0.194 | -0.090 | -0.039 | 0.108 | 0.0054 |
| predictive_entropy_1step | -0.097 | -0.301 | -0.134 | -0.365 | 0.267 | 0.0126 |
| n_forward_planned | -0.085 | -0.204 | -0.101 | -0.111 | 0.139 | 0.0022 |
| ensemble_disagreement_1step | -0.079 | -0.295 | -0.127 | -0.367 | 0.263 | 0.0141 |
| latent_novelty | -0.064 | +0.104 | -0.139 | +0.103 | 0.115 | 0.0113 |
| latent_post_prior_kl | -0.050 | +0.009 | -0.345 | +0.064 | 0.153 | 0.0264 |
| internal_truncation_js | +0.049 | +0.192 | +0.080 | +0.056 | 0.109 | 0.0039 |
| latent_prior_std | +0.029 | -0.009 | +0.053 | -0.057 | 0.044 | 0.0020 |
| context_onestep_error | +0.016 | -0.211 | -0.150 | -0.333 | 0.231 | 0.0158 |
| free_ahead | +0.004 | +0.586 | +0.024 | +0.566 | 0.392 | 0.0791 |

### Criteria on completely held-out cells

  A+B+C->D       c1 3/3  c2 3/3  c3 3/3  -> restored on 3/3 seeds
  A+B->C         c1 3/3  c2 2/3  c3 3/3  -> restored on 2/3 seeds
  A+B->D         c1 3/3  c2 3/3  c3 3/3  -> restored on 3/3 seeds
  A+C->B         c1 3/3  c2 3/3  c3 3/3  -> restored on 3/3 seeds
  A+C->D         c1 3/3  c2 3/3  c3 3/3  -> restored on 3/3 seeds
  A+D->B         c1 3/3  c2 3/3  c3 3/3  -> restored on 3/3 seeds
  A+D->C         c1 3/3  c2 1/3  c3 3/3  -> restored on 1/3 seeds

  in-distribution preserved (A-test MAE within 0.5 of A-only): {'A': True, 'A+B': True, 'A+C': True, 'A+D': True, 'A+B+C': True}

### H6 verdict (main): **SUPPORTED**   per seed: ['SUPPORTED', 'SUPPORTED', 'SUPPORTED']
--- size-matched control: seed 0 ---
--- size-matched control: seed 1 ---
--- size-matched control: seed 2 ---

========================================================================
 H6 size-matched control  (mean over 3 seeds)
========================================================================

training anchors per regime: {'A': 22200, 'A+B': 22200, 'A+C': 22200, 'A+D': 22200, 'A+B+C': 22200}
test anchors per config: {'A': 11100, 'B': 2220, 'C': 2202, 'D': 2220}

### Predicted vs actual H* (bold = config unseen in training)

| regime | test A | test B | test C | test D |
|---|---|---|---|---|
| *actual* | 15.60 | 8.08 | 3.95 | 4.89 |
| A | 15.68 | **21.60** | **13.83** | **21.21** |
| A+B | 15.57 | 7.80 | **12.91** | **5.21** |
| A+C | 15.01 | **10.51** | 6.15 | **9.54** |
| A+D | 15.44 | **6.89** | **12.30** | 4.57 |
| A+B+C | 14.72 | 7.84 | 6.30 | **5.38** |

### MAE (bold = unseen)

| regime | test A | test B | test C | test D |
|---|---|---|---|---|
| A | 6.394 | **14.747** | **10.609** | **16.768** |
| A+B | 6.451 | 4.688 | **9.807** | **3.628** |
| A+C | 6.514 | **7.500** | 4.190 | **6.800** |
| A+D | 6.444 | **4.909** | **9.267** | 2.734 |
| A+B+C | 6.563 | 4.743 | 4.292 | **3.622** |

### SPEARMAN (bold = unseen)

| regime | test A | test B | test C | test D |
|---|---|---|---|---|
| A | +0.580 | **+0.218** | **+0.213** | **+0.135** |
| A+B | +0.575 | +0.747 | **+0.278** | **+0.759** |
| A+C | +0.573 | **+0.411** | +0.477 | **+0.463** |
| A+D | +0.575 | **+0.694** | **+0.287** | +0.844 |
| A+B+C | +0.569 | +0.745 | +0.466 | **+0.762** |

### Transfer T_MAE(regime -> config) = MAE(g_A) - MAE(g_regime); positive = improvement over IID-only

| regime | test A | test B | test C | test D |
|---|---|---|---|---|
| A+B | -0.057 [-0.10,-0.01] | +10.060 [+9.33,+10.78] | **+0.801 [+0.37,+1.25]** | **+13.140 [+12.35,+13.89]** |
| A+C | -0.120 [-0.17,-0.07] | **+7.247 [+6.58,+7.92]** | +6.419 [+5.83,+7.03] | **+9.968 [+9.32,+10.62]** |
| A+D | -0.050 [-0.09,-0.01] | **+9.838 [+9.04,+10.60]** | **+1.342 [+0.90,+1.82]** | +14.034 [+13.28,+14.76] |
| A+B+C | -0.170 [-0.23,-0.11] | +10.005 [+9.27,+10.72] | +6.317 [+5.74,+6.91] | **+13.146 [+12.36,+13.88]** |

### Rank stability of uncertainty features

| feature | rho_A | rho_B | rho_C | rho_D | mean|rho_OOD| | RS = Var_D[rho] |
|---|---|---|---|---|---|---|
| imagined_disp_spread | -0.455 | -0.096 | -0.179 | -0.096 | 0.124 | 0.0227 |
| internal_truncation_disp | +0.299 | +0.081 | +0.188 | +0.099 | 0.123 | 0.0083 |
| ensemble_disagreement_mean | -0.204 | -0.194 | -0.090 | -0.039 | 0.108 | 0.0054 |
| predictive_entropy_1step | -0.097 | -0.301 | -0.134 | -0.365 | 0.267 | 0.0126 |
| n_forward_planned | -0.085 | -0.204 | -0.101 | -0.111 | 0.139 | 0.0022 |
| ensemble_disagreement_1step | -0.079 | -0.295 | -0.127 | -0.367 | 0.263 | 0.0141 |
| latent_novelty | -0.064 | +0.104 | -0.139 | +0.103 | 0.115 | 0.0113 |
| latent_post_prior_kl | -0.050 | +0.009 | -0.345 | +0.064 | 0.153 | 0.0264 |
| internal_truncation_js | +0.049 | +0.192 | +0.080 | +0.056 | 0.109 | 0.0039 |
| latent_prior_std | +0.029 | -0.009 | +0.053 | -0.057 | 0.044 | 0.0020 |
| context_onestep_error | +0.016 | -0.211 | -0.150 | -0.333 | 0.231 | 0.0158 |
| free_ahead | +0.004 | +0.586 | +0.024 | +0.566 | 0.392 | 0.0791 |

### Criteria on completely held-out cells

  A+B+C->D       c1 3/3  c2 3/3  c3 3/3  -> restored on 3/3 seeds
  A+B->C         c1 3/3  c2 1/3  c3 2/3  -> restored on 1/3 seeds
  A+B->D         c1 3/3  c2 3/3  c3 3/3  -> restored on 3/3 seeds
  A+C->B         c1 3/3  c2 3/3  c3 3/3  -> restored on 3/3 seeds
  A+C->D         c1 3/3  c2 3/3  c3 3/3  -> restored on 3/3 seeds
  A+D->B         c1 3/3  c2 3/3  c3 3/3  -> restored on 3/3 seeds
  A+D->C         c1 3/3  c2 1/3  c3 3/3  -> restored on 1/3 seeds

  in-distribution preserved (A-test MAE within 0.5 of A-only): {'A': True, 'A+B': True, 'A+C': True, 'A+D': True, 'A+B+C': True}

### H6 verdict (size-matched control): **SUPPORTED**   per seed: ['SUPPORTED', 'SUPPORTED', 'SUPPORTED']


### FINAL: main = SUPPORTED | size-matched control = SUPPORTED
