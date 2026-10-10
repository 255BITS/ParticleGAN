# Hydraulic v1 on the selected BCAP winner

Research diagnostic. It grants no ordinary qualification and changes no default.
Earlier hydraulic rounds keep their original identities: [round 1](../README.md),
[round 2](../round2/README.md), [round 3](../round3/README.md).

## Preregistered plan (frozen before execution)

**Winner (control).** The develop-selected `bcap` preset, direction-blend BCAP
(`bcap-default-baseline-direction-v1`). It is zero-momentum DualNorm with smoothing
.001, G/E .012, D x1.5, prior x2.5, non-saturating loss, cap/coefficient/lazy 1/1/1,
per-offset convolution and constant rates. It is the only BCAP recipe with a
complete ordinary measurement: 6/6 Tier 1 and 9/21 Tier 2 at
[source d378734f](../../../bcap-default-baseline/README.md). Experiment memory
says to base future BCAP research on it. #373 (direction blend plus local-v2
transport) is a 7-task research diagnostic with no ordinary measurement. It is
an open PR and does not replace the winner.

**Control reuse, not a rerun.** The winner's archived ordinary results are the
control. Scientific source has not changed between d378734f and this branch's
base (`git diff d378734f..origin/develop -- particlegan experiments benchmarks lib
configs/forge/tasks` is empty). The only code delta is an opt-in Recipe flag
that defaults to off. With the flag off, the trainer calls the identical `opt_g.step()`.
The recipe serialization omits both new defaults, so archived recipe identities
are unchanged. [`tests/test_hydraulic_travel.py`](../../../../../tests/test_hydraulic_travel.py)
checks disabled-flag tensor parity.

**Binding check (AGENTS.md).** v1 ran against the #355 search winner. On every
public-GANTrainer task, the current winner differs from that recipe only by
direction blend. Direction blend has zero activation on those tasks: GANTrainer
protects only `loss_gan`, which is also its whole objective. The ordinary
receipts show `blended_steps=0` and byte-identical tensors. The v1 control
signatures reproduce exactly (Gaussian stationary 2/72, grid100 precision
.24072). So v1's archived evidence binds to this winner on those hosts.
Prior, initializer, budget and sampling law remain task-owned and unchanged.

**Scope.** The bound needs the trainer-owned unconditional sample-space update.
It covers the consumed real batch, replayed latents and prior rows. Ten
revision-8 cells run in caller-owned component hosts and do not consume it:
two_pole, unused_token_hold, ae_gan_hold, both five-word tasks, trajectory,
residual_student, unipolar, cover_leftover and mid_scale_identity. Preflight
declares them unsupported (no silent inactivity), as Forge already does for
sample-space transport. Under the ordinary tier veto, this would block Tier 2
entirely. The measured scope is therefore a separate diagnostic view with all
**17** public-trainer tasks of the revision-8 suite. These are Tier 1 Gaussian
smoke and ring acquisition, plus 15 Tier 2 tasks. Original gates and own
checkpoint dependencies remain; Gaussian stability restores its own passing
smoke state.

**Arms.** One global trainer configuration per arm. The trainer delta from the
winner is explicit:

| Arm | Delta vs winner | Mechanism question |
| --- | --- | --- |
| A0 winner | none (archived ordinary evidence) | control |
| A1 travel | `hydraulic_travel_fraction=1, radius=real_spacing` (exact PR360 v1) | Does v1's retention/precision signal hold on the winner, and what does it cost on the 14 cells v1 never measured? |
| A2 gap | `hydraulic_travel_fraction=1, radius=gap_adaptive` | The v1 Gaussian smoke confirmed at 834, versus 375 for the control. That matches a pure speed limit: travel about 3 units at the accepted .0036 RMS per update is about 830 updates. Does the bound still retain the target if the radius is max(real spacing, median generated-to-real nearest distance)? That radius equals v1 at stationarity and opens only while the generator is far from the data. |

Bound-scale-only arms are excluded. A tighter bound (.5) would predictably miss
the 1,000-update smoke, because acquisition is travel-limited. A looser fixed
bound is the uncontrolled version of A2. No seed-only repeats.

**Protocol.** Protocol seed 0 and public deterministic initializer. Each task
keeps its architecture, data law, batch sequence, prior, sampling, update
budget and evaluation cadence. Named constructor, data, training-noise and
evaluation streams are checkpointed; probes draw no RNG (tested). The hardware
is CUDA on RTX A6000 with Python 3.12.13 and Torch 2.14.0. The reservation is
31,620 seconds per arm and the campaign ceiling is 64,000 seconds. There is one
run per arm with zero scientific retries, no tuning and no rescue checkpoints.

**Predictions** (A1 numbers repeat the archived v1 values, because the binding is exact):

| Cell | A0 winner (observed) | A1 travel (predicted) | A2 gap (predicted) |
| --- | --- | --- | --- |
| gaussian1d_smoke | PASS, confirm 375 | PASS, confirm 834 | PASS, confirm <=500 |
| gaussian1d_stability | FAIL, stationary 2/72, shift 0/24 | FAIL, 71/72, 23/24 | stationary >=70/72, shift >=22/24, final KS <=.05 |
| grid100 precision | .24072 FAIL | .49586 (>=.48) FAIL | >=.40 FAIL |
| rotated/staggered precision | .256/.302 | >=.40 each, FAIL | >=.35 each, FAIL |
| vector_two_broad | PASS | PASS | PASS |
| ring16_acquisition | PASS (confirm 1250/1600) | at risk (speed-limited) | PASS |
| mode_hold | FAIL (suffix 3/5) | suffix >=3 | suffix >=3 |
| other vector/image cells | spiral, stripes2 PASS; others FAIL | no new pass; spiral/stripes2 at risk | no new pass; passes kept |

The machine-checked study prediction is grid100 precision >=.48 for A1. For A2
it is a Gaussian stability final KS <=.05.

**Decision rule.**

- **Promote** an arm to an ordinary full-suite comparison only if it repairs
  `gaussian1d_stability` (full PASS) and keeps all five winner passes in scope:
  smoke, ring, broad, spiral and stripes2.
- **Continue** when Gaussian retention improves (stationary >=60/72) with no
  lost winner pass but the gate still fails.
- **Stop** that revision when it loses any winner pass, or when retention does
  not improve.

## Results

**Verdict: stop both exact revisions; neither is promoted.** Both arms repair no
gate and each loses one winner pass in scope: A1 loses Tier 1 ring acquisition,
and A2 loses vector_spiral. The bound reliably improves Gaussian retention and
native placement, but the remaining Gaussian failure is a slow mean drift that
a per-update speed limit does not remove.

Compact leaderboard: [LEADERBOARD.md](LEADERBOARD.md). [results.json](results.json)
holds all cells, hydraulic counters, attempt IDs and source digests.
[media/](media/index.json) has 34 actual-training GIFs rendered from saved
observations, with no added updates or samples.

| Key metric | A0 winner | A1 travel (v1) | A2 gap-adaptive |
| --- | --- | --- | --- |
| Passes in the 17-cell scope | **5** | 4 | 4 |
| Lost winner passes | - | ring16_acquisition (Tier 1) | vector_spiral |
| Gaussian smoke first confirmation | 375 | 834 | **459** |
| Gaussian stationary / shift hold / reacquisition | 2/72, 0/24, FAIL | **71/72, 23/24**, PASS | 68/72, 22/24, PASS |
| Gaussian final KS / std ratio | .3206 / .662 | **.0152** / .984 | .0225 / .943 |
| grid100 precision / centre RMS sigma / cov trace bias | .2407 / 1.520 / **+.371** | .4959 / .379 / +.895 | **.6343 / .183** / +.806 |
| rotated100 / staggered100 precision | .2555 / .3017 | .4894 / .5276 | **.6251 / .6461** |
| vector_two_broad component cov error (all PASS) | .3856 | **.2309** | .2476 |
| unequal_mass min mass ratio / comp cov error | .2075 / 3.692 | **.9357 / .656** (still FAIL) | .7023 / 1.337 |
| unequal_width comp cov error / mass TV | 6.288 / .2915 | **5.177 / .1091** | 8.868 / .2947 |
| anisotropic mass TV | **.196** | .333 (one component empty) | .333 (one component empty) |
| mode_hold modes / hq / suffix | **8 / .989 / 3** | 4 / .683 / 0 | 7 / .760 / 0 |
| Paid seconds (17 cells) | archived | 2,419 | 2,741 |

The ten caller-owned cells do not consume the bound. Their winner results are
unchanged by construction: two-pole, unused token, AE hold, both word tasks,
trajectory, residual and three identity hosts. This includes the conditional
identity repairs (trajectory and residual) and word hold.

**Predictions.**

- **A1 reproduced archived v1 exactly.** Smoke confirmed at 834. Stationary was
  71/72 and shift hold 23/24. grid100 precision was .49586, so the study
  prediction (>=.48) was observed. Broad cov error was .230946. This confirms the
  binding: direction blend is inactive on these hosts, so v1 evidence transfers.
  The new cells show the cost of the speed limit. Ring acquisition misses its
  terminal component-covariance gate: one of 16 components has error 9.12, and
  the terminal suffix is 0. Mode hold collapses to 4 modes. Rotated/staggered
  precision roughly doubles (.49/.53).
- **A2.** Smoke confirmed at 459, inside the predicted 500. Shift hold was 22/24
  (predicted >=22). Final KS was .0225, so the study prediction (<=.05) was
  observed. grid100 precision was .634 (predicted >=.40), and ring passed.
  Stationary was 68/72, missing the >=70 prediction. vector_spiral, predicted to
  be kept, lost its terminal suffix (1/5).

**Mechanism readout** (saved counters; no new measurements):

- **The bound is a hard speed limit almost everywhere.** A1 limits 99-100% of
  Gaussian, native, ring and vector updates. The mean accepted scale is .04-.06
  on Gaussian and native and .27-.57 on vector. The radius is the real-batch
  spacing (.0051 Gaussian, .0119 native). There are zero rejections, and the
  maximum accepted/radius ratio is <=1.
- **Images are essentially untouched.** Limited fractions are 0-.23 and the
  mean scale is >=.90. bars4, blobs4 and stripes2 therefore end with numerically
  identical metrics in all three arms. Only intensity2 under A1 (limited 23%)
  changes, losing one mode.
- **The A2 radius opens as designed.** It is wider than the spacing in 53-56% of
  Gaussian updates and 100% of native updates (mean native radius .048 vs .012).
  It roughly halves acquisition time and raises native precision to .63-.65,
  comparable to #370 Armijo (.66). It does so while keeping 68/72 retention.
- **The remaining Gaussian misses are mean drift, not jitter.** Every stationary
  or shift-hold miss in both arms has width ratio 1.00-1.06 and mean error
  .10-.16 target sigma, with KS .050-.068. The large-KS states at
  4042-4459 are the post-shift reacquisition window, and A2 clears it faster. A
  tighter bound slows the drift but cannot remove the location bias. Expanding
  the bound (A2) adds three such misses.
- **The density gates still fail.** Native covariance trace bias stays
  +.8 to +.9, versus +.37 for the winner, and radial KS stays .26-.33 against a
  .04 bound. Anisotropic loses a component in both arms. Bounded travel improves
  placement, not within-component shape. This is the same tradeoff as #371 role
  balance and #370 Armijo.

**Recommendation.** Keep the winner. Retain the hydraulic flag as an opt-in
research flag; it defaults to off and archived identities are unchanged. Do not
run bound-scale sweeps or seed repeats. If this line continues, the next single
mechanism should target the coherent mean-location bias that survives the bound,
for example the shared-mean component of the G/prior update. It must keep the A2
radius, measure shared versus differential displacement, and use the same
17-cell scope plus explicit ring, spiral and mode-hold checks. Gaussian retention
remains the open failure.

## Provenance

- Execution source: commit 2087cbf29, scientific digest
  `3983c4fa7a97bc7b94029861cd11dca37e4d022ab5c03d3ec3e10104d6e04222`.
  Both arms share the protocol (sha256 `7b975bab…`) and runtime (`640b9254…`).
  Protocol seed 0, Python 3.12.13, Torch 2.14.0, two RTX A6000s, two workers
  per GPU.
- Requests: travel `2782783c5e834d00892c1a9b`, gap `7981e71ff235563021dc89d2`.
  34 complete cells, zero retries, 5,159.6 paid seconds, against a 63,240-second
  reservation within the 64,000-second ceiling.
- Control: archived `bcap-default-baseline-direction-v1` results (source
  d378734f, digest `6a225fcd…`), not rerun. There is no scientific source drift
  from d378734f to the execution base. Develop at 0e9c7daf0 adds only
  `benchmarks/toy_audit/ring16_restart.py`.
- Software: `tests/test_hydraulic_travel.py` has 11 checks. They cover the
  bound, gap radius, opt-in identity, checkpoint replay, probes consuming no RNG
  stream, disabled-flag tensor parity and caller-owned preflight blockers.
  217 related recipe/boundary/technique/trainer tests pass, and
  `forge validate` passes.
- Raw logs, JSONL, checkpoints and attempt receipts are kept outside Git under
  `/mnt/ml7tb/ParticleGAN-forge/bcap-hydraulic-winner-20261010`.

Reproduce or inspect:

```sh
python reports/forge/bcap-physics/hydraulic/winner/workflow.py prepare
python reports/forge/bcap-physics/hydraulic/winner/workflow.py run --gpus 0,1 --workers-per-gpu 2
tail -f /mnt/ml7tb/ParticleGAN-forge/bcap-hydraulic-winner-20261010/logs/driver.log
python reports/forge/bcap-physics/hydraulic/winner/publish.py --media   # saved-only
```
