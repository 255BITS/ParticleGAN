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

## Draft: fixed-sigma MoG prior — Stage 0 gate (2026-09-16)

Stage 0 only, three seeds (1–3), 7k steps, 200k final EMA samples per run and
matched real references. Full [report and deviations](results/mog/STAGE0.md),
[results CSV](results/mog/results.csv), and [traces plot](results/mog/stage0_traces.png).
No small-N sweep has run; predictions 1–7 are **inconclusive** pending their
specified comparisons.

C0 regression passes **3/3**: all 70 original log entries per seed and every final
EMA generator tensor plus the particle table are bit-identical to pre-change
`af1843a`. The measured code is `5db6d3a`; the new prior uses sigma=0 and no
standardization for this control.

| Reference | Pass rate | HQ / real | Core width / real | HQ-only KL |
|---|---:|---:|---:|---:|
| C0: 20k learned atoms | 0/3 | 0.99978 ± 0.00107 | 0.84343 ± 0.04312 | 0.03478 ± 0.00195 |
| C1: fresh Gaussian | 0/3 | 0.06069 ± 0.00310 | 10.18916 ± 0.04109 | 0.39484 ± 0.02837 |

C0 covers all 100 modes and crosses the old coverage bar at step 5500 in every
seed, but fails the stricter width and balance criteria. Evaluating each of its
20k atoms exactly once gives KL **0.03684 / 0.03361 / 0.03303**: the imbalance
is in the learned distribution, not eval sampling noise. Fair-share-normalized
mode shares range from **0.465–0.526** at the minimum to **1.650–1.798** at the maximum.
Historical `hist_kl` includes bridge samples; applying that estimator directly
to the reproduced pre-change 20k-sample outputs gives **0.03498 / 0.03221 / 0.03240**.
These are current-recipe reproductions, not recovered historical study scores.

The simulated allocation-null KL means are **0.56891 / 0.28368 / 0.13208** for
N=100/200/400 (1,000 draws each); mean empty-mode counts are 36.557/13.526/1.829.
The historical core-width estimator is reused exactly (median radius around
per-mode coordinate medians, no HQ truncation), with same-size real normalization.

Recommendation: **go to the Stage 1 optimizer pilot**, retaining the stricter
pass criterion; do not equate matching C0's HQ with passing. Its balance and width
leave room for a useful result. This is a recommendation only: later stages are
paused for the owner's decision. The six Stage 0 runs took 3.4 minutes on two
A6000s; C0 averaged 70.2 seconds/run including the new metric suite.

## Fixed-sigma MoG prior — Stage 1 optimizer pilot (2026-09-16)

The owner authorized Stage 1 and revised the criterion to **original or better**.
Before pilot results, we froze the observed three-seed C0 envelope: modes=100,
HQ/real ≥ **0.99883228**, width/real **0.79091586–1.20908414**, and HQ-only
KL ≤ **0.03750414**. All C0 references pass this rule. The old design criterion
remains recorded separately; Stage 0 history is unchanged.

**No match at nominal r=1/8 within 7k steps:** 36 unique runs, **0/36 passes**.
All 36 fail HQ, width, and balance individually; this is not a marginal threshold
failure. We completed the 30-run LR sweep and six new momentum runs, reusing the
six identical beta=0 comparisons. Tests: 35 passed plus 13 subtests. All 36 runs
are certified complete; all 2,520 metric-trace rows and frozen configs validated.
Training sources: `33b3144`.

Selected by pass rate, then HQ, then width distance from real:

| N | Particle LR multiplier | Particle β1 | Passes | HQ/real | Width/real | KL |
|---|---:|---:|---:|---:|---:|---:|
| 100 | 10× (initial LR 0.06) | 0 | 0/3 | 0.29137 | 4.400 | 0.37679 |
| 400 | 10× (initial LR 0.06) | 0.5 | 0/3 | 0.38016 | 3.373 | 0.15626 |

Higher LR improves HQ but does not approach C0. The momentum preference is small:
N=100 beta=0.5 has HQ/real 0.28825; N=400 beta=0 has 0.37668. The selected N=400
momentum setting has worse balance than beta=0 (0.15626 vs 0.13829); it wins only
through the preregistered HQ tie-break after both pass rates are zero. Neither is
an established optimum; both LR winners are at the tested upper boundary.

**The Gaussian centers fit much better than their noisy neighborhoods.** In the
selected N=400 cell, **99.4%** of component centers map within a data mode's 3σ
radius, but **37.6%** of noisy samples do. HQ-conditioned purity is effectively
1.0, with no empty majority allocations. N=100 has 91.0% HQ component centers,
28.8% HQ noisy samples, purity 0.9803, and 13.3 empty majority allocations. These
center-only measurements are explicit diagnostics; all primary evaluation keeps
noise on. Excessive output spread, plus N=100 allocation failures, explains the
poor quality better than an evaluation-only pass-rule issue.

**Clumping complicates the proposed noise/separation ratio.** Selected N=400
r_eff averages **4.63**, despite nominal r=0.125; 91.2% of evaluable components'
nearest neighbors have the same majority output mode. This is not the nominal
r=2 Gaussian-collapse control. All three selected runs at each N also exceed the
2× raw-scale drift threshold. The report retains those flags and separately
records distances to neighbors with different majority modes; the prescribed
r_eff and ranking are unchanged.

Recommendation: before the full Stage 2 grid, test N=400 with the selected
optimizer at **r=0, 1/32, and 1/16**. The atoms control isolates the small-table
limitation; smaller noise tests whether width can recover. r=1/32 would be an
explicit addition to the original design. **No follow-up runs launched.** The
seven original predictions remain inconclusive for their full stated comparisons;
the pilot's supporting and contrary observations are enumerated in the report.

[Full report](results/mog/STAGE1.md) · [per-run results](results/mog/stage1_results.csv)
· [leaderboard](results/mog/stage1_leaderboard.csv)
· [optimizer plots](results/mog/stage1_optimizer.png)
· [runbook](results/mog/STAGE1_RUNBOOK.md).
The timed first run took 68 seconds; the remaining LR batch took 10.2 minutes
and the momentum round 2.5 minutes with two workers per A6000.
