[warn] CUDA present but unusable (AcceleratorError: CUDA error: no kernel image is available for execution on the device
Search for `cudaErrorNoKernelImageForDevice' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
); falling back
device: cpu

=== stage 1: severity calibration (seed 0, disjoint layout seeds) ===
[17:36:42]   cal:obs:0.3: 925 anchors, H* mean 14.54 sd 10.46 iqr 20.0 ceiling_frac 0.189 zero_frac 0.056
  calibration obs p=0.3: H* 14.54 (target 7.75, |diff| 6.79)
[17:36:55]   cal:obs:0.5: 925 anchors, H* mean 13.76 sd 10.55 iqr 20.0 ceiling_frac 0.173 zero_frac 0.074
  calibration obs p=0.5: H* 13.76 (target 7.75, |diff| 6.01)
[17:37:07]   cal:obs:0.7: 925 anchors, H* mean 12.68 sd 10.35 iqr 17.0 ceiling_frac 0.152 zero_frac 0.092
  calibration obs p=0.7: H* 12.68 (target 7.75, |diff| 4.93)
  -> selected obs p=0.7 (H* 12.68)
[17:37:20]   cal:dropout:0.2: 925 anchors, H* mean 3.79 sd 4.09 iqr 4.0 ceiling_frac 0.001 zero_frac 0.147
  calibration dropout p=0.2: H* 3.79 (target 7.75, |diff| 3.96)
[17:37:32]   cal:dropout:0.35: 925 anchors, H* mean 2.33 sd 2.86 iqr 3.0 ceiling_frac 0.000 zero_frac 0.274
  calibration dropout p=0.35: H* 2.33 (target 7.75, |diff| 5.42)
[17:37:44]   cal:dropout:0.5: 925 anchors, H* mean 1.77 sd 2.28 iqr 2.0 ceiling_frac 0.000 zero_frac 0.305
  calibration dropout p=0.5: H* 1.77 (target 7.75, |diff| 5.98)
  -> selected dropout p=0.2 (H* 3.79)

=== stage 2: scored test sets ===
  F2 (observation noise p=0.7): 600 episodes
  F3 (action dropout p=0.2): 600 episodes
[17:39:28]   F2_test: 11100 anchors, H* mean 12.25 sd 10.25 iqr 17.0 ceiling_frac 0.142 zero_frac 0.093
[17:40:36]   F3_test: 11100 anchors, H* mean 3.70 sd 4.25 iqr 4.0 ceiling_frac 0.001 zero_frac 0.161
[17:46:33]   F2_test: 11100 anchors, H* mean 12.78 sd 10.33 iqr 17.0 ceiling_frac 0.155 zero_frac 0.077
[17:47:40]   F3_test: 11100 anchors, H* mean 3.62 sd 4.16 iqr 4.0 ceiling_frac 0.001 zero_frac 0.163
[17:53:31]   F2_test: 11100 anchors, H* mean 13.14 sd 10.22 iqr 18.0 ceiling_frac 0.154 zero_frac 0.065
[17:54:35]   F3_test: 11100 anchors, H* mean 3.69 sd 4.20 iqr 4.0 ceiling_frac 0.001 zero_frac 0.154

========================================================================
 H9 aleatoric-family transfer
========================================================================

selected severities: F2 obs p=0.7, F3 dropout p=0.2

### Validity gate (frozen: H* <= 9.3 AND IID-only transfer < 0.35)

| config | H* | A->cfg rho | passes gate |
|---|---|---|---|
| F2 | 12.72 | +0.359 | NO (invalid instrument) |
| F3 | 3.67 | +0.238 | YES |

### Spearman by regime (mean over seeds)

| regime | test A | test B | test C | test D | test F2 | test F3 |
|---|---|---|---|---|---|---|
| A | +0.580 | **+0.218** | **+0.213** | **+0.135** | **+0.359** | **+0.238** |
| A+B | +0.584 | +0.755 | **+0.306** | **+0.771** | **+0.207** | **+0.311** |
| A+C | +0.576 | **+0.461** | +0.479 | **+0.510** | **+0.286** | **+0.444** |
| A+F2 | +0.583 | **+0.226** | **+0.241** | **+0.087** | +0.545 | **+0.257** |
| A+F3 | +0.566 | **+0.388** | **+0.468** | **+0.539** | **+0.311** | +0.451 |

### Frozen predictions

| # | cell | threshold | rho | seeds | holds | instrument |
|---|---|---|---|---|---|---|
| Q1 | A+C -> F2 | rho > 0.50 | +0.286 | 0/3 | no | INVALID |
| Q2 | A+B -> F2 | rho < 0.35 | +0.207 | 3/3 | YES | INVALID |
| Q3 | A+F2 -> C | rho > 0.50 | +0.241 | 0/3 | no | valid |
| Q4 | A+C -> F3 | rho > 0.50 | +0.444 | 0/3 | no | valid |
| Q5 | A+B -> F3 | rho < 0.35 | +0.311 | 3/3 | YES | valid |

### Q6 `free_ahead` on new configs: F2 -0.015, F3 -0.031  -> holds: True

### H9 verdict: **NOT SUPPORTED (aleatoric shifts do not transfer to each other)**
