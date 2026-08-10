# Phase 0 — Literature Audit and Novelty Boundary

Audit performed before implementation. Purpose: locate the exact boundary between what is
already demonstrated and what this experiment would add.

## Novelty table

| Paper | Year | What it already demonstrates | How our hypothesis differs |
|---|---|---|---|
| [Certified World Models: Predictability Across Configuration, Horizon, and Resolution](https://arxiv.org/abs/2606.13092) | 2026 | A-priori horizon certificates from the predictor's **Jacobian/Lyapunov spectrum**: `T(ε) ~ log(1/ε)/λ`. No future observation needed. Tested on Lorenz, Hénon, L96, PushT, FetchPush. | Certificate is **orbit-wide and worst-case**. Authors state explicitly it "does not predict which specific future state will breach tolerance". We predict a **per-state** horizon and evaluate it as a prediction task. |
| [Conformal Orbit-Valid Trust Horizons for Equivariant World Models](https://arxiv.org/abs/2606.24946) | 2026 | Trust horizon from one-step latent residual + finite-time expansion estimate, split-conformal calibrated; transported across a symmetry orbit. Median certified/measured ratio 0.67. | Analytic + conformal, requires known group symmetry; distributional audit, not a learned per-state predictor. No predicted-vs-actual horizon correlation reported. |
| [Round-Trip Consistency: Bidirectional Diffusion Models Can Predict Their Own Rollout Errors](https://arxiv.org/abs/2608.00675) | 2026 | Self-supervised proxy for rollout error via forward-then-backward rollout; Spearman 0.91–0.98 vs realized error; per-depth calibrated error. | Requires **performing** a rollout (2× inference) and an invertible/bidirectional model. Domains are PDE/video, not an acting agent. Ours is a-priori from the belief state, and tested for planning utility. |
| [Biased Dreams: Limitations to Epistemic Uncertainty Quantification in Latent Space Models](https://arxiv.org/abs/2604.25416) | 2026 | **Negative result:** ensemble disagreement in RSSM latents captures local but *not* global compounding error; latent rollouts drift to attractor regions where uncertainty falsely collapses. | Directly predicts our experiment may fail. We pre-register this as the main falsification risk and test whether a *learned* predictor beats the disagreement scalar it critiques. |
| MBPO / MACURA / adaptive-rollout family ([e.g. 2206.02380](https://arxiv.org/abs/2206.02380)) | 2020–2025 | Rollout length adapted from model uncertainty; truncate when uncertainty crosses a threshold. Improves model-based RL sample efficiency. | **Online truncation** observes the rollout as it unfolds. We ask whether the horizon is knowable *before* imagining, and we include online truncation as a baseline (`trunc_h_disp`, `trunc_h_js`). |
| [ELVIS: Ensemble-Calibrated Latent Imagination for Long-Horizon Visual MPC](https://arxiv.org/abs/2605.04709) | 2026 | Adaptive-horizon MPC with real-time confidence thresholds; compute–performance trade-off. | Threshold-crossing during rollout, not a predicted scalar horizon; no predicted-vs-actual horizon evaluation. |
| [Uncertainty-Aware Robotic World Model (offline MBRL on real robots)](https://arxiv.org/abs/2504.16680) | 2025 | Epistemic uncertainty estimates track realized rollout error over long horizons. | This is Claim A (correlation). We treat it as the weak baseline our result must beat. |
| [Understanding Rollout Error in Graph World Models](https://arxiv.org/abs/2606.27780) | 2026 | Topology-dependent error amplification (spectral radius × weight norms); explains *why* rollouts degrade. | Post-hoc explanatory analysis at the model/topology level; no per-state prospective horizon. |
| [Hallucination in World Models: Predictable & Preventable](https://arxiv.org/abs/2606.27326) | 2026 | Three signals predicting *where* a 350M visual world model will fail; coverage-aware data fixes. | Predicts failure *location//likelihood*, not a per-state horizon in steps; no planning-compute claim. |

## Conclusion of the audit

The hypothesis as originally stated — "uncertainty at time `t` correlates with future
imagination error" — **is already demonstrated** (Claim A) and would be incremental.

Three things are *not* jointly established anywhere found:

1. a **per-state, learned, a-priori** estimate of the horizon (existing a-priori methods are
   analytic and explicitly worst-case/orbit-wide);
2. evaluated as a **prediction task against ground-truth `H*`**, with the decisive control
   being that it must beat the *best monotone function of any single uncertainty scalar*;
3. without requiring the rollout to be performed first (unlike round-trip consistency) or a
   known symmetry group (unlike the conformal trust-horizon work).

The hypothesis was therefore sharpened into H1–H4 in `PREREGISTRATION.md`. The literature
also supplies a concrete **falsification risk**: *Biased Dreams* predicts that latent
ensemble disagreement should fail exactly this task. That makes the experiment genuinely
two-sided rather than a confirmation exercise.
