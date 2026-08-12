device=cuda
--- seed 0 ---
  R0_baseline    A_test pred 15.65 MAE 6.436 | B pred 22.01 rho +0.230 C pred 14.69 rho +0.202 D pred 21.31 rho +0.157
  R1_ranknorm    A_test pred 15.87 MAE 6.625 | B pred 22.48 rho +0.234 C pred 16.18 rho +0.157 D pred 21.95 rho +0.049
  R2_gated       A_test pred 15.60 MAE 6.440 | B pred 20.92 rho +0.205 C pred 13.94 rho +0.218 D pred 20.41 rho +0.152
  R3_combined    A_test pred 15.82 MAE 6.629 | B pred 21.42 rho +0.200 C pred 15.40 rho +0.172 D pred 21.06 rho +0.040
  verdict(seed 0): NOT SUPPORTED
--- seed 1 ---
  R0_baseline    A_test pred 15.87 MAE 6.447 | B pred 22.80 rho +0.235 C pred 12.73 rho +0.294 D pred 22.27 rho +0.113
  R1_ranknorm    A_test pred 16.00 MAE 6.594 | B pred 21.78 rho +0.241 C pred 16.35 rho +0.136 D pred 22.32 rho +0.073
  R2_gated       A_test pred 15.82 MAE 6.452 | B pred 20.79 rho +0.183 C pred 12.43 rho +0.300 D pred 20.26 rho +0.081
  R3_combined    A_test pred 15.95 MAE 6.597 | B pred 19.94 rho +0.197 C pred 15.95 rho +0.144 D pred 20.37 rho +0.046
  verdict(seed 1): NOT SUPPORTED
--- seed 2 ---
  R0_baseline    A_test pred 15.54 MAE 6.298 | B pred 19.87 rho +0.131 C pred 13.34 rho +0.216 D pred 19.88 rho +0.115
  R1_ranknorm    A_test pred 15.74 MAE 6.402 | B pred 18.70 rho +0.189 C pred 15.71 rho +0.120 D pred 19.60 rho +0.113
  R2_gated       A_test pred 15.48 MAE 6.306 | B pred 19.05 rho +0.116 C pred 13.00 rho +0.221 D pred 19.17 rho +0.119
  R3_combined    A_test pred 15.69 MAE 6.412 | B pred 17.97 rho +0.173 C pred 15.32 rho +0.125 D pred 18.94 rho +0.110
  verdict(seed 2): NOT SUPPORTED

=== H5 diagnostic: uncertainty signals per split (mean over seeds) ===
| feature | A_test | B_test | C_test | D_test |
|---|---|---|---|---|
| dispspread_mean | 0.526 (rho -0.46) | 1.085 (rho -0.11) | 1.502 (rho -0.20) | 1.043 (rho -0.09) |
| js_moved_mean | 0.013 (rho -0.20) | 0.020 (rho -0.16) | 0.020 (rho -0.13) | 0.021 (rho -0.04) |
| ent_moved_mean | 0.209 (rho -0.40) | 0.149 (rho -0.16) | 0.213 (rho -0.11) | 0.153 (rho +0.04) |
| prior_std_anchor | 0.797 (rho +0.03) | 0.809 (rho +0.01) | 0.791 (rho +0.07) | 0.809 (rho -0.09) |
| trunc_h_disp | 24.383 (rho +0.30) | 19.750 (rho +0.10) | 16.633 (rho +0.20) | 19.955 (rho +0.09) |
| latent_nn_dist | 3.927 (rho -0.06) | 5.891 (rho +0.10) | 5.126 (rho -0.16) | 5.864 (rho +0.08) |

=== H5 arms (mean over seeds) ===
| arm | split | actual H* | predicted H* | MAE | Spearman |
|---|---|---|---|---|---|
| R0_baseline | A_test | 15.60 | 15.68 | 6.394 | +0.580 |
| R0_baseline | B_test | 7.75 | 21.56 | 14.855 | +0.199 |
| R0_baseline | C_test | 4.11 | 13.59 | 10.266 | +0.238 |
| R0_baseline | D_test | 5.11 | 21.15 | 16.451 | +0.128 |
| R1_ranknorm | A_test | 15.60 | 15.87 | 6.540 | +0.561 |
| R1_ranknorm | B_test | 7.75 | 20.99 | 14.305 | +0.221 |
| R1_ranknorm | C_test | 4.11 | 16.08 | 12.372 | +0.138 |
| R1_ranknorm | D_test | 5.11 | 21.29 | 16.743 | +0.078 |
| R2_gated | A_test | 15.60 | 15.63 | 6.399 | +0.579 |
| R2_gated | B_test | 7.75 | 20.25 | 13.867 | +0.168 |
| R2_gated | C_test | 4.11 | 13.12 | 9.839 | +0.246 |
| R2_gated | D_test | 5.11 | 19.95 | 15.350 | +0.117 |
| R3_combined | A_test | 15.60 | 15.82 | 6.546 | +0.561 |
| R3_combined | B_test | 7.75 | 19.78 | 13.433 | +0.190 |
| R3_combined | C_test | 4.11 | 15.56 | 11.886 | +0.147 |
| R3_combined | D_test | 5.11 | 20.12 | 15.695 | +0.066 |

=== per-seed criteria ===
  seed 0 R1_ranknorm    c1=False c2=False c3=True -> no
  seed 0 R2_gated       c1=False c2=False c3=True -> no
  seed 0 R3_combined    c1=False c2=False c3=True -> no
  seed 1 R1_ranknorm    c1=False c2=False c3=True -> no
  seed 1 R2_gated       c1=False c2=False c3=True -> no
  seed 1 R3_combined    c1=False c2=False c3=True -> no
  seed 2 R1_ranknorm    c1=False c2=False c3=True -> no
  seed 2 R2_gated       c1=False c2=False c3=True -> no
  seed 2 R3_combined    c1=False c2=False c3=True -> no

### H5 verdict: **NOT SUPPORTED**  (per seed: ['NOT SUPPORTED', 'NOT SUPPORTED', 'NOT SUPPORTED'])
