[warn] CUDA present but unusable (AcceleratorError: CUDA error: no kernel image is available for execution on the device
Search for `cudaErrorNoKernelImageForDevice' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
); falling back
device=cpu archs=['gru_large', 'gru_small', 'transformer'] seeds=[0] K=3
[15:53:18] collecting A_wmtrain (A, seeds 0-3000)
[15:54:46]   A_wmtrain: 3000 episodes, 360000 steps
[15:54:46] collecting A_gtrain (A, seeds 3000-4200)
[15:55:20]   A_gtrain: 1200 episodes, 144000 steps
[15:55:20] collecting A_val (A, seeds 4200-4800)
[15:55:38]   A_val: 600 episodes, 72000 steps
[15:55:38] collecting A_test (A, seeds 4800-5400)
[15:55:55]   A_test: 600 episodes, 72000 steps
[15:55:55] collecting B_test (B, seeds 100000-100600)
[15:56:17]   B_test: 600 episodes, 72000 steps
[15:56:17] collecting C_test (C, seeds 100000-100600)
[15:56:36]   C_test: 600 episodes, 71817 steps
[15:56:36] collecting D_test (D, seeds 100000-100600)
[15:57:18]   D_test: 600 episodes, 72000 steps
=== gru_large seed 0 ===
[15:58:39]   gru_large:A_gtrain: 22200 anchors, H* mean 15.74 sd 10.30 iqr 20.0 ceiling_frac 0.219 zero_frac 0.024
[15:59:19]   gru_large:A_val: 11100 anchors, H* mean 15.65 sd 10.26 iqr 21.0 ceiling_frac 0.215 zero_frac 0.022
[15:59:58]   gru_large:A_test: 11100 anchors, H* mean 15.59 sd 10.30 iqr 21.0 ceiling_frac 0.217 zero_frac 0.023
[16:00:39]   gru_large:B_test: 11100 anchors, H* mean 7.70 sd 8.67 iqr 11.0 ceiling_frac 0.052 zero_frac 0.228
[16:01:19]   gru_large:C_test: 11056 anchors, H* mean 4.10 sd 4.96 iqr 4.0 ceiling_frac 0.004 zero_frac 0.139
[16:02:00]   gru_large:D_test: 11100 anchors, H* mean 5.04 sd 6.54 iqr 7.0 ceiling_frac 0.012 zero_frac 0.293
=== gru_small seed 0 ===
[16:02:34]   gru_small:A_gtrain: 22200 anchors, H* mean 13.68 sd 9.78 iqr 17.0 ceiling_frac 0.143 zero_frac 0.032
[16:02:50]   gru_small:A_val: 11100 anchors, H* mean 13.58 sd 9.78 iqr 16.0 ceiling_frac 0.141 zero_frac 0.031
[16:03:05]   gru_small:A_test: 11100 anchors, H* mean 13.65 sd 9.84 iqr 17.0 ceiling_frac 0.148 zero_frac 0.032
[16:03:21]   gru_small:B_test: 11100 anchors, H* mean 7.47 sd 8.40 iqr 10.0 ceiling_frac 0.044 zero_frac 0.227
[16:03:37]   gru_small:C_test: 11056 anchors, H* mean 4.28 sd 4.97 iqr 5.0 ceiling_frac 0.003 zero_frac 0.133
[16:03:53]   gru_small:D_test: 11100 anchors, H* mean 5.13 sd 6.73 iqr 7.0 ceiling_frac 0.013 zero_frac 0.295
=== transformer seed 0 ===
[16:06:53]   transformer:A_gtrain: 22200 anchors, H* mean 16.68 sd 9.42 iqr 17.0 ceiling_frac 0.199 zero_frac 0.002
[16:08:11]   transformer:A_val: 11100 anchors, H* mean 16.73 sd 9.42 iqr 17.0 ceiling_frac 0.199 zero_frac 0.002
[16:09:26]   transformer:A_test: 11100 anchors, H* mean 16.58 sd 9.44 iqr 18.0 ceiling_frac 0.197 zero_frac 0.001
[16:10:43]   transformer:B_test: 11100 anchors, H* mean 11.40 sd 9.07 iqr 13.0 ceiling_frac 0.086 zero_frac 0.029
[16:11:58]   transformer:C_test: 11056 anchors, H* mean 6.79 sd 6.01 iqr 8.0 ceiling_frac 0.008 zero_frac 0.071
[16:13:15]   transformer:D_test: 11100 anchors, H* mean 8.55 sd 7.31 iqr 9.0 ceiling_frac 0.024 zero_frac 0.038

shared feature subspace: 74 columns, names identical across all 3 architectures

========================================================================
 H7 cross-architecture
========================================================================

### Within architecture (mean over seeds)

| arch | H* on A | A MAE | A rho | beats Claim-A ceiling | D rho (train A) | D rho (train A+B) |
|---|---|---|---|---|---|---|
| gru_large | 15.59 | 6.408 | +0.584 | yes (+1.355) | +0.201 | +0.724 |
| gru_small | 13.65 | 6.351 | +0.551 | yes (+1.050) | -0.003 | +0.745 |
| transformer | 16.58 | 6.432 | +0.513 | yes (+0.993) | +0.257 | +0.290 |

### Cross-architecture: g_phi fitted on SRC, evaluated on TGT's A-test

| fitted on \ evaluated on | gru_large | gru_small | transformer |
|---|---|---|---|
| gru_large | rho +0.584, MAE 6.41 | **rho +0.391, MAE 7.67** | **rho +0.207, MAE 8.69** |
| gru_small | **rho +0.356, MAE 8.69** | rho +0.551, MAE 6.35 | **rho +0.288, MAE 10.21** |
| transformer | **rho +0.363, MAE 7.98** | **rho +0.362, MAE 7.88** | rho +0.513, MAE 6.43 |

(bold = transferred across architectures; diagonal = same architecture)
