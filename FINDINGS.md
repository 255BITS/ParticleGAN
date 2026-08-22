# Gradient-penalty centering in RpGAN — findings (round 3: provenance, bake-off, corrected audit)

**Benchmark:** `examples/100gaussians.py` recipe (RpGAN, ParticlePrior, Fourier-2 D, EMA eval,
Adam β1=0, delayed cosine anneal), 7 000 steps, 5 seeds per cell. **420 runs, 0 failures**;
training is deterministic (48/48 audit reruns bit-identical). Metrics: exact W1/W2 via POT
(n=4096; sampling floor W1 ≈ 0.14), `hq` (mass within 3σ_true = 0.09 of a center), per-mode
width measured two ways after round 3 — raw mean-centered σ ratio (tail-inflated; kept for
context) and **median-centered core σ ratio** (the honest number; see
`results/metric_recon.md`), plus tail mass beyond 10σ_true. Stability: Arnoldi spectral radius
of the alternating-GD update map (comparative only). Tables: `results/TABLE.md`,
`results/bootstrap.md`, `results/LEADERBOARD.md`, `results/provenance.md`.

## The organizing result: compatibility vs. curvature

- **Curvature (whether you converge) is center-agnostic.** Unregularized: radius 1.055 ± 0.052,
  |Im λ| ≈ 0.037, 47/100 modes. Any sample-point penalty, any centering: radius ≤ 1.002,
  |Im λ| ≤ 0.0015, zero collapses in all penalized logistic runs. (Mescheder Lemma 3.3 made
  empirical; new: zero-centering damps rotation ~4× harder at matched coeff — 2.8e-4 vs 1.25e-3.)
- **Compatibility (where you can converge) is loss-family-relative.** Under logistic RpGAN the
  optimal critic flattens at match; center-at-1 forbids that and pays a residual W1 floor
  (~2.5×) — the Dirac-GAN incompatibility in its soft, finite-training form. Under the
  **Wasserstein objective the same penalties at the same coefficients are a statistical wash on
  W1** (all CI95 include 0) and the capped arms take the fidelity cells. Centering must match
  the loss family's optimal critic: margin device for f-div, transport device for IPM.

## Round 3 verdicts

**Provenance of the cap's stability — trajectory regularizer, not endpoint regularizer.**
The blocking question (hinge slack at a flat critic ⇒ where does B's radius ≤ 1.002 come
from?) resolves to *inherited from the state, earned by the trajectory*: at B's endpoints
≤ 0.4% of samples clear the cap and masking the penalty out moves the dominant modulus by
−0.0004 (adding a zero-centered probe penalty there: also nothing), while the identical masked
measurement on `f_none`'s own checkpoints reads 1.068 — the state differs, not the
measurement. The hinge *was* engaged for 56–75% of training (q90 ‖∇D‖ above the cap until
step ~5 200–5 900) and went slack only when the LR anneal collapsed the critic to near-flat.
**Deployment consequence:** the cap supplies no standing damping at convergence. Safe where
the equilibrium is self-stable; in a game that keeps injecting rotation (e.g. recurrent
world-model training), standing curvature (R1/R2) is the defensible choice.

**Corrected variance audit — retraction and a third strike against strict eikonal.** Round 2's
"no variance collapse anywhere (σ ratios 2.1–5.4)" was an artifact of the mean-centered second
moment: the far tail inflates the moment *and* drags the per-mode mean. Median-centered core
widths: champion `b_cap` **0.87–0.92** (≈ true width; 0.8% tail mass beyond 10σ), `a_r1r2@0.1`
0.91–0.96, `a_r1r2@1.0` **2.16** (the blur is real; hq 0.60 is honest), and
`c_eikonal@0.1` **core 0.55 ± 0.23 — variance-compressed** — hidden under a raw ratio of 4.80
by 3.15% of mass stranded whole grid cells away (99th-pct nearest-center distance 2.01 vs
0.47 for b_cap): the expected signature of a critic pinned to unit slope between modes. The
leaderboard's compression guard now reads the core ratio; core width predicts measured hq to
within 0.05, so the two metrics no longer conflict.

**Wasserstein + cap bake-off — won on the stated metrics, then demoted by the corrected
audit.** On W1/hq/tails the caps look dominant: every cap cell matches A on W1 (0.159–0.172,
CIs include 0), hq 0.94–0.975 vs 0.87–0.92, bar-pass 5/5 in all six cap cells vs 0/5 for
A@1.0, and a broad coefficient plateau (0.3/1/3). But the core-ratio audit shows **every
bar-passing WGAN cell — cap AND R1/R2@0.1 — is variance-compressed** (cores 0.38–0.73;
raw ratios of 1.5–2.2 hid it), while the only core-honest WGAN cells, A@1.0 (core 1.02–1.14),
never pass the bar. Under IPM on this benchmark, sharpness is purchased substantially with
core compression. What survives: `wgan_a_r1r2@1.0` is the best *calibration* cell in the
study (W1 0.154, core 1.02, recall 1.000, tail 2.1% — soft on hq, which is the conservative
direction), and unpenalized WGAN detonates (collapse 0.8, radius ~1e9), so a penalty — any
penalty — is load-bearing there.

**Curriculum (B→A hard switch) — prediction failed, coherently.** The switch keeps B's
bar-crossing speed (3 860) but the endpoint re-equilibrates to the destination arm's frontier
position within the remaining 2 800 steps (→A@0.1: W1 0.359/hq 0.929 ≈ B's own endpoint;
→A@1.0: hq collapses to 0.65). No two-knob escape: consistent with provenance — the
trajectory bequeaths the *basin*; the W1/hq operating point is set by the penalty active at
the end. Together with round 2's κ-anneal null: neither soft nor hard schedules cross the
frontier; the frontier is a property of the *final* penalty regime.

## Standing verdicts (rounds 1–2, updated wording)

- **H1:** falsified for f-div on transport metrics (CI-backed); the sharpness half survives
  the *corrected* audit for the cap family only (core ≈ 0.9), not for strict eikonal
  (core-compressed). The W1-vs-concentration dissociation stands.
- **H2:** confirmed (0.045 vs 1.04 med ‖∇D‖ mid-training; late sag as the data term wins).
- **H3:** inverted into the decomposition above — zero-centering is not needed for damping,
  but is doubly right under f-div: compatible center *and* strongest rotation damper per unit
  coefficient, *and* (round 3) the only family with standing curvature at the endpoint.
  R3GAN chose correctly; these results spell out why.
- **H4:** partial — caps fastest to the sharp bar at base LR (CI95 [−900, −380] steps);
  2× LR improves endpoints, not bar speed. OAdam: clean null (rotation is already gone;
  residual instability is symmetric-part, which optimism cannot fix). Lazy ≡ every-step at
  matched time-average.

## Leaderboard & promotion (see `results/LEADERBOARD.md`)

Logistic promotion track CHAMPION: **`b_cap` L2, coeff 1.0, 2×LR** (W1 0.290, hq 0.986,
core σ 0.866 — clears the corrected guard; runner-up L1 variants 0.82–0.85 also honest;
the L∞ variant, d_asym, e_interp, and all measurable c_eikonal cells are now flagged
variance-compressed). The logistic cap cells and the curriculum→A@0.1 cell (core 0.911,
W1 0.283, hq 0.938) are the only sharp-and-core-honest configurations in the study.
**Bake-off outcome:** the WGAN+cap recommendation is withdrawn — its dominance was partly
core compression. If the objective may change and the deployment metric is *calibration*,
`wasserstein + a_r1r2@1.0` is the best honest cell (W1 0.154, core 1.02); if the metric is
*sharpness*, stay logistic and promote the champion. Provenance caveat travels with the cap
either way.

## Caveats

Symmetric R1+R2 only (R1-alone diverges per R3GAN's ablations); γ is a frontier dial, not an
escape. ~40 CI cells → multiplicity; headline claims rest on the largest, replicated,
cross-validated effects. Spectral radii comparative (GD map, no optimizer state; FD smoothing
bias identical across arms). One 2-D toy, small MLPs; deployment validation happens on the
target domain (world-model repo — see `docs/eikonal-branch-notes.md` there), with this suite
retained as a CI canary rather than a benchmark.

## Reproduce

`lib/grad_regularizers.py` (arms, norms, anneal; FD-gradchecked), `lib/oadam.py` (vendored —
do not substitute third-party copies), `lib/game_jacobian.py`, `lib/toy_metrics.py` (incl.
core-ratio estimator), `experiments/train_arm.py` (per-run `ckpt.pt` + `final_samples.npy`),
`experiments/gen_configs.py --stage {main,lr_sens,lazy,audit,anneal,wgan,dualnorm,oadam,
bakeoff,curriculum}`, `experiments/run_grid.py`, `experiments/analyze.py`,
`experiments/leaderboard.py`, `experiments/provenance.py`, `experiments/metric_recon.py`.
Raw runs in `results/runs*/` (gitignored, on disk).
