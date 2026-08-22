# Gradient-penalty centering in RpGAN — findings (round 2, incl. crossover & audit)

**Benchmark:** `examples/100gaussians.py` recipe (RpGAN, ParticlePrior, Fourier-2 D, EMA eval,
Adam β1=0, delayed cosine anneal), 7 000 steps, 5 seeds per cell. **351 runs total, 0 failures**:
main grid (arms A–F × coeff {0.005…1.0}), A@0.3 control, LR ×{0.5,2}, lazy-vs-every-step,
per-mode variance audit (60 deterministic reruns — all 48 checked reproduce the original
summaries bit-identically), κ-anneal, dual-norm (L1/L∞) caps, OptimisticAdam, and a
Wasserstein-objective crossover. Exact W1/W2 via POT (n=4096; sampling floor W1 ≈ 0.14);
sharpness = `hq` (mass within 3σ of a center) + per-mode σ ratio (audited); stability = Arnoldi
spectral radius of the alternating-GD game Jacobian (comparative only; Adam/OAdam state
excluded). Tables: `results/TABLE.md`, `results/bootstrap.md`, `results/LEADERBOARD.md`.

## The organizing result: compatibility vs. curvature

Two independent properties of a gradient penalty, previously conflated, separate cleanly here:

- **Curvature (whether you converge) is center-agnostic.** Unregularized game: spectral radius
  1.055 ± 0.052, rotational content |Im λ| ≈ 0.037, 47/100 modes. *Any* sample-point penalty,
  any centering: radius ≤ 1.002, |Im λ| ≤ 0.0015, zero collapses in all 285 penalized logistic
  runs. This is Mescheder et al.'s Lemma 3.3 algebra made empirical — the damping comes from
  the penalty's PSD curvature block, not the center (their own footnote notes WGAN-GP/DRAGAN
  are not zero-centered). New measurement on top of the known theory: centering *modulates*
  the damping — zero-centering suppresses rotation ~4× harder at matched coeff (|Im λ|
  2.8e-4 vs 1.25e-3 at 0.1) via the cross-term curvature that survives at a flat critic.
- **Compatibility (where you can converge) is loss-family-relative.** The logistic-RpGAN
  optimal critic is density-ratio-shaped: flat at distribution match. A center-at-1 penalty
  forbids that, and the pathology Mescheder proved divergent on the Dirac-GAN shows up here in
  its soft, finite-training form: a **residual W1 floor** (~0.18–0.25 sliced vs 0.08 for
  R1/R2@γ=1.0; coeff-1.0 panel of `plots/w1_curves.png`), masked by anneal+EMA, with the
  mid-training ‖∇D‖≈1 target visibly sagging after step ~5 000 as the data term fights the
  constraint (`plots/gradnorm_evolution.png`). Under the logistic objective every strict-eikonal
  cell loses W1 to tuned R1/R2 with bootstrap CI excluding 0.

**The crossover is the kill test, and it convicts the mismatch, not the geometry.** Rerunning
the grid under `loss_type: wasserstein` (IPM — where the unit-slope critic is the *correct*
optimum): every centering variant now **matches** zero-centering on W1 (all CI95 include 0;
A@1.0 0.154, B@1.0 0.164, C@0.1 0.177, E@1.0 0.199), the one-centered arms take the better
hq/σ cells (E@1.0 hq 0.978, B@0.1 σ-ratio 1.49 — the best distributional fidelity in the whole
study, near the 0.14 W1 sampling floor), and no-penalty WGAN detonates (collapse rate 0.8,
spectral radius ~1e9). Same penalties, same coefficients: a 2× W1 penalty under f-div, a wash
under IPM. **Centering must match the loss family's optimal critic: margin device for f-div
losses, transport device for IPM losses.**

## Hypothesis verdicts (updated)

- **H1** — falsified as stated for f-div RpGAN (transport metrics, CI-backed), with the
  sharpness half now audit-hardened: the per-mode variance audit found **no variance collapse
  anywhere** (σ ratios 2.1–5.4, all over-dispersed, center bias ≤ 0.08 = 2.7σ); B's hq 0.986
  is genuine concentration, so the W1-vs-sharpness frontier dissociation stands as the novel
  empirical content.
- **H2** — confirmed (0.045 vs 1.04 med ‖∇D‖ at mid-training), plus the late-training sag.
- **H3** — the interesting inversion: zero-centering is *not* load-bearing for damping
  (curvature is center-agnostic), but it *is* load-bearing for the equilibrium class under
  f-div losses (compatibility). Neither half of the original directive's framing survives intact.
- **H4** — partial: at base LR the cap arms are fastest to the repo bar (B@1.0 3 860 ± 233 vs
  A@0.1 4 500 ± 200; CI95 of the difference [−900, −380]); at 2× LR they don't collapse and
  their endpoints improve to study-best, but bar-crossing happens later (wider orbit until the
  anneal), so LR *tolerance* holds, LR *speed* does not.

## Round-2 arms

- **Dual-norm caps (arXiv:1910.06922 replication):** the L1-gradient cap (L∞ margin — their
  best CIFAR cell) replicates directionally here: `b_cap` L1@1.0 beats L2 at both LRs on hq
  and σ ratio (hq 0.992, σ 2.14 at 2×LR — study-best fidelity numbers) and is within noise of
  the L2 champion on W1. Our B-beats-C result is an independent replication of their
  inequality-vs-equality finding in a new objective family.
- **κ-anneal (target 1→0):** prediction half-wrong. Annealing the center slides B *along* the
  frontier toward A (linear: W1 0.368→0.345, hq 0.967→0.929) instead of capturing both ends;
  delayed ≈ no-op. The margin pressure that buys sharpness leaves with the constraint.
- **OptimisticAdam** (correct preconditioned-step variant, vendored from
  ParticleGAN-WorldModel): no help. On top of penalties it's neutral-to-harmful (b_cap: same
  W1, hq −0.03, σ ratio 2.9→5.3); alone it does *not* rescue the unregularized game (radius
  1.036 vs 1.055, 64/100 modes) — this game's instability isn't the pure rotation optimism
  cancels, and the penalties already remove what rotation there is.

## Leaderboard (promotion track — see `results/LEADERBOARD.md`)

**CHAMPION: `b_cap` L2, coeff 1.0, 2×LR** (W1 0.290, hq 0.986, recall 0.998, bar 5/5,
audit-clean). Runner-up within noise: the L1-norm variant (hq 0.992, best σ ratio).
Off-board note for a bigger decision: the WGAN-objective cells dominate the global fidelity
frontier (B@1.0: W1 0.164 *and* hq 0.951 *and* σ 1.74) — promoting a penalty inside the
logistic recipe is the conservative move; switching the toy benchmark's objective to
Wasserstein+cap is the aggressive one and needs its own bake-off (different objective, not
ranked on the boards).

## Caveats (unchanged ones flagged before, plus scope)

The "crank γ" observation applies to the **symmetric R1+R2** tested here — R3GAN's ablations
show R1-alone diverging even at γ=100; γ remains the frontier dial, not an escape (γ=1.0 fails
the repo bar at hq 0.60). ~30 CI cells invite multiplicity inflation; the headline claims rest
on the largest, replicated effects. Spectral radii are comparative (GD map, no optimizer
state). All of this is one 2-D toy with small MLPs — scale smoke test required before any
external writeup. Repo-bar speed claims embed the bar's own hq threshold (circularity noted;
the W1 and σ audits are the independent checks).

## Reproduce

Arms/norm/anneal: `lib/grad_regularizers.py` (FD-gradchecked incl. L1/L∞ in
`tests/test_regularizers.py`); optimizer: `lib/oadam.py` (vendored verbatim — do not replace
with third-party copies); trainer `experiments/train_arm.py` (writes `final_samples.npy` +
`ckpt.pt` per run now); stages via `experiments/gen_configs.py --stage
{main,lr_sens,lazy,audit,anneal,wgan,dualnorm,oadam}`; `experiments/run_grid.py` (resume-safe);
`experiments/analyze.py` regenerates every table/plot; `experiments/leaderboard.py` the boards.
Raw runs in `results/runs*/` (gitignored, on disk).
