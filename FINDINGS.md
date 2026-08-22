# Eikonal vs. zero-centered gradient penalties in RpGAN — findings

**Benchmark:** `examples/100gaussians.py` recipe (RpGAN-logistic, ParticlePrior, Fourier-2 D,
EMA eval, Adam β1=0, delayed cosine anneal), 7 000 steps. **Grid:** arms A–F × coeff
{0.005, 0.02, 0.1, 1.0} × 5 seeds, plus A@0.3, LR-sensitivity (×0.5/×2) and lazy-vs-every-step
stages — 185 runs, 0 failures, 0 collapse events in any penalized arm. Quality = exact W1/W2
(POT `emd2`, 4 096 pts; finite-sample floor for this mixture: **W1 ≈ 0.14**) plus sliced-W1
(100 k pts / 512 proj), mode recall, hist-KL/JS, NLL; sharpness = the repo's `hq` (fraction
within 3σ of a center). Spectral radius of the local game Jacobian via float64 Arnoldi on
finite-difference JVPs of the alternating GD update map (Adam state excluded by design; the
FD smoothing bias is identical across arms). Full numbers: `results/TABLE.md`,
`results/bootstrap.md`, plots in `results/plots/`.

## Verdicts

**H1 (quality) — FALSIFIED on transport metrics, CONFIRMED on sharpness.** On final exact W1
every eikonal (arm, coeff) loses to R1/R2 at γ=1.0 with bootstrap CI95 excluding zero
(A@1.0: **0.183 ± 0.022**, recall 1.000, best NLL; best eikonal: B@1.0 0.368 ± 0.028). But
A@1.0 is *blurry* — hq 0.598, never passes the repo bar (100 modes + hq ≥ 0.9). The two
families sit on different ends of a W1-vs-sharpness frontier that neither can cross: sweeping
A's γ over {0.005 … 1.0} its sharpness peaks at hq 0.943 (γ=0.1) and the A@0.3 control
confirms the frontier is smooth (W1 0.229, hq 0.868), while the slope-preserving arms reach
hq 0.962–0.986 (study best: B@1.0 at 2×LR — W1 0.290, **hq 0.986**, recall 0.998) — a
sharpness regime no tested γ of A reaches. Zero-centering buys transport accuracy by blurring;
slope-saturation buys sharpness at a ~2× W1 cost.

**H2 (mechanism) — CONFIRMED, textbook.** Mid-training median ‖∇D‖ at samples: 0.045/0.078
(real/fake) under A@1.0 vs 1.04/1.20 under C@0.1 (`plots/gradnorm_evolution.png`). Bonus
consistent with theory: after step ~5 000 (near distribution match) C's norms sag below 1 —
the data term overpowers the eikonal target exactly when the ideal critic wants to flatten.

**H3 (stability) — zero-centering is NOT load-bearing for damping.** Unregularized game:
dominant modulus 1.055 ± 0.052 (max 1.151), rotational content |Im λ| ≈ 0.037, 47/100 modes.
*Every* penalty arm, any centering, any coeff ≥ 0.02: modulus ≤ 1.002, |Im λ| ≤ 0.0015, zero
collapses across all 175 penalized runs. Damping comes from penalizing the gradient-norm
*curvature* at the samples, not from the zero center (zero-centering does damp rotation ~4×
harder at matched coeff: |Im| 2.8e-4 vs 1.25e-3 at 0.1 — a gradient, not a cliff). The
predicted eikonal failure mode ("no stationary point with p_g = p_data and ‖∇D‖ = 1") is real
but shows up as a **residual W1 floor** (~0.18–0.25 sliced vs 0.08 for A@1.0, coeff-1.0 panel
of `plots/w1_curves.png`), not as oscillation or divergence — the LR anneal + EMA read-out
absorb the residual drift.

**H4 (speed / LR headroom) — PARTIALLY CONFIRMED.** At base LR the eikonal arms reach the
repo convergence bar fastest of the whole study: C@0.1 3 840 ± 242 and B@1.0 3 860 ± 233 steps
vs 4 500 ± 200 for the best bar-reaching baseline A@0.1 (bootstrap CI95 of the difference
[−940, −400], significant). At 2× LR eikonal arms don't collapse and their *endpoints improve*
to the study's best (B/D hq 0.986), whereas A stays on its own frontier — but everyone crosses
the bar *later* at 2×LR (wider orbit until the anneal engages), so "higher LR tolerance" holds
and "fewer steps at higher LR" does not. The directive's steps-to-2×-best-A-W1 metric is
degenerate here: no eikonal arm ever reaches it (see H1).

**Pitfall checks.** Lazy-every-16 with 16× rescale ≡ every-step at the same coeff
(C@0.1: 0.446 vs 0.452) — centering does *not* interact with lazy application; only the
time-averaged weight matters (every-step at λ/16 is simply weaker and worse, 1.013).
Arm E (interpolates) ≈ arm C (data points) at matched coeff (0.465 vs 0.452 at 0.1) —
evaluation location is a second-order effect here; centering is the live variable.

## Bottom line

The directive's premise imports Kantorovich duality into a non-Wasserstein objective, and the
data falsify its strong form: re-centering the penalty at 1 does not improve distributional
distance — it strictly loses W1/W2/NLL to a well-tuned zero-centered penalty, because the
eikonal constraint is incompatible with the flat-critic equilibrium and leaves a residual
transport floor. What survives, and is worth keeping: (1) the damping that stabilizes RpGAN
comes from *any* gradient penalty at the samples, not from zero-centering — H3's negative
prediction failed; (2) the one-sided cap (B, ‖∇D‖ ≤ 1, penalty-free for a flat critic — so it
keeps the correct equilibrium while still bounding steepness) is the best variant tested:
sharpest samples in the study, mild LR-robustness gains, and ~15 % faster to this repo's
convergence bar. If your quality metric rewards sharpness (this repo's does), cap the critic's
slope instead of zeroing it; if it rewards transport (FID-like), keep R1/R2 and crank γ higher
than you think (γ=1.0 here, 50× the tuned default).

## Reproduce

`lib/grad_regularizers.py` (arms; unit/FD tests in `tests/test_regularizers.py`),
`lib/game_jacobian.py` (spectral), `experiments/train_arm.py` (single run),
`experiments/gen_configs.py` + `configs/` (185 YAMLs), `experiments/run_grid.py` (parallel
runner, resume-safe), `experiments/analyze.py` (tables/bootstrap/plots). Every figure and
table regenerates from `results/runs/*/{summary.json,metrics.jsonl}` via
`.venv/bin/python experiments/analyze.py`.
