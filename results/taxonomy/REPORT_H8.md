[warn] CUDA present but unusable (AcceleratorError: CUDA error: no kernel image is available for execution on the device
Search for `cudaErrorNoKernelImageForDevice' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
); falling back
device: cpu
collecting E (MiniGrid-SimpleCrossingS11N5-v0, noise=0.0, slip=0.0)
  E: 600 episodes, 72000 steps
collecting F (MiniGrid-DoorKey-16x16-v0, noise=0.15, slip=0.0)
  F: 600 episodes, 72000 steps
--- seed 0: extracting E/F features ---
[16:40:08]   E_test: 11100 anchors, H* mean 2.18 sd 3.23 iqr 3.0 ceiling_frac 0.001 zero_frac 0.419
[16:41:13]   F_test: 11100 anchors, H* mean 15.23 sd 10.38 iqr 20.0 ceiling_frac 0.207 zero_frac 0.036
--- seed 1: extracting E/F features ---
[16:47:04]   E_test: 11100 anchors, H* mean 2.17 sd 3.23 iqr 3.0 ceiling_frac 0.001 zero_frac 0.419
[16:48:09]   F_test: 11100 anchors, H* mean 15.42 sd 10.40 iqr 20.2 ceiling_frac 0.213 zero_frac 0.034
--- seed 2: extracting E/F features ---
[16:54:06]   E_test: 11100 anchors, H* mean 2.18 sd 3.23 iqr 3.0 ceiling_frac 0.001 zero_frac 0.419
[16:55:13]   F_test: 11100 anchors, H* mean 15.56 sd 10.20 iqr 20.0 ceiling_frac 0.205 zero_frac 0.025

========================================================================
 H8 taxonomy test
========================================================================

### Degeneracy check on the new configs

| config | mean H* | sd | IQR | ceiling | zero | degenerate |
|---|---|---|---|---|---|---|
| E | 2.18 | 3.23 | 3.0 | 0.001 | 0.419 | False |
| F | 15.23 | 10.38 | 20.0 | 0.207 | 0.036 | False |

### Spearman by regime (mean over seeds)

| regime | test A | test B | test C | test D | test E | test F |
|---|---|---|---|---|---|---|
| A | +0.580 | **+0.218** | **+0.213** | **+0.135** | **+0.116** | **+0.472** |
| A+B | +0.584 | +0.755 | **+0.306** | **+0.771** | **+0.855** | **+0.373** |
| A+C | +0.576 | **+0.461** | +0.479 | **+0.510** | **+0.555** | **+0.417** |
| A+E | +0.581 | **+0.732** | **+0.313** | **+0.800** | +0.879 | **+0.423** |
| A+F | +0.587 | **+0.266** | **+0.231** | **+0.102** | **+0.110** | +0.569 |

### Registered predictions

| # | cell | threshold | rho (mean) | per-seed pass | holds |
|---|---|---|---|---|---|
| P1 | A+B -> E | rho > 0.50 | +0.855 | 3/3 | YES |
| P2 | A+B -> F | rho < 0.35 | +0.373 | 1/3 | NO |
| P3 | A+C -> F | rho > 0.50 | +0.417 | 0/3 | NO |
| P4 | A+E -> D | rho > 0.50 | +0.800 | 3/3 | YES |
| P5 | A+F -> C | rho > 0.50 | +0.231 | 0/3 | NO |

### P6: is `free_ahead` the epistemic carrier?

| A | B | C | D | E | F |
|---|---|---|---|---|---|
| +0.004 | +0.586 | +0.024 | +0.566 | +0.533 | +0.019 |

P6 holds: True  (|rho_E|=0.533 > 0.3, |rho_F|=0.019 < 0.15)

### H8 verdict: **NOT SUPPORTED**
