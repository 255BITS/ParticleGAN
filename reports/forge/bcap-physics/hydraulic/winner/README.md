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

_Pending execution._ Tail the central log:

```sh
tail -f /mnt/ml7tb/ParticleGAN-forge/bcap-hydraulic-winner-20261010/logs/driver.log
```
