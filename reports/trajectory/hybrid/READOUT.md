# Hybrid D improves held-out path fidelity, not within-route diversity

All four runs completed: two 1k viability scouts, then two matched 10k runs.
Both RTX A6000 GPUs used, one worker each. Total training across all runs:
**4.14 GPU minutes**. No warnings or failures. No more experiments queued.

[Interactive comparison](confirm_10k/index.html) ·
[Hybrid animation](confirm_10k/hybrid_continuous/futures.gif) ·
[Hybrid config](../../../configs/trajectory/hybrid/confirm_10k/hybrid_continuous.yaml) ·
[Motion completion handoff](MOTION_HANDOFF.md)

## 10k leaderboard

Both use continuous geometry, learned particles, Gaussian noise, the same
four-step DDGAN/UCD formulation, constant LR and sample exposure. Variance is
median within-route coefficient variance relative to real, for valid outputs;
target1. All12 interpolation and16 train-reference variance groups are present.

| D | Test SW1 ↓ | Route TV ↓ | Interp. valid ↑ | Extrap. valid ↑ | Test collisions ↓ | Interp. variance ratio | Train s |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Hybrid** | **.0320** | **.0483** | **99.8%** | **91.9%** | 2.0% | .36 | 126.4 |
| MLP control | .0385 | .0667 | 99.6% | 76.7% | **.2%** | .34 | **98.6** |

Real-vs-real floor: SW1 .0070, route TV .0156, validity100%, collisions0%.
Both cover all16 test context/route pairs. Each10k run sees1.28M examples per
optimizer phase /2.56M real draws. Endpoint-only512 samples/context evaluation.

The hybrid reduces test SW1 by16.9%, improves route calibration and extrapolation
validity by15.2 percentage points, but costs28.3% more training time. It uses
202,280 D parameters versus204,040 MLP (-0.9%); matched parameter count does not
imply matched compute. G102,402 and prior640,000 parameters remain unchanged.

## The diversity hypothesis did not succeed

The intent was to combine MLP shape fidelity with the previous temporal D's
richer spread. The hybrid's variance ratio is only .36 versus MLP .34; both
remain substantially underdispersed. The earlier temporal/continuous value .72
is historical context, not a third matched-source arm of this round. We gained
held-out fidelity, not the hoped-for recovery of continuous variation.

The test SW1 advantage mostly comes from the extrapolation scene: hybrid .0520
versusMLP .0772. Interpolation SW1 is almost tied (.0253 vs .0256). On original
training-reference geometries, hybrid SW1 is slightly worse (.0393 vs .0367),
although validity is higher (99.7% vs99.2%). No universal winner claim.

## Why validity can improve while collisions get worse

Validity requires collision freedom, boundary accuracy and proximity to the
known analytic path family. In the extrapolation scene, MLP's23.3% invalidity
is explained by failing the support-distance threshold; only0.4% collide.
Hybrid support failures fall to1.9%, but collisions rise to7.8%, leaving8.1%
invalid overall. These categories overlap. Neither model fails the boundary
threshold in the evaluated test samples. Across the full test set, collisions
are2.0% hybrid versus0.2% MLP.

See [validity audit](validity_audit.json), recomputed from saved test arrays
with unchanged Routes.diagnose. The hybrid better reproduces the target path
family on average but still generates some paths through the obstacle. Its
best pooled SW1 is not a basis for calling it suitable for real robot control.

## Recommendation and stopping point

Keep hybrid/continuous as the strongest observed toy candidate for held-out
fidelity, with MLP/continuous as the cheaper low-collision comparison. Preserve
both configs; no no-argument defaults changed. The original diversity question
remains open, and neither configuration solves distribution matching.

Stop this toy round here and move to planning real motion completion after
compact, as requested. The [handoff](MOTION_HANDOFF.md) captures the retained
formulation, dataset/conditioning decisions and evaluation requirements. No
real-motion dataset has been chosen or downloaded; no motion job is queued.

## Design, validation and limitations

- [Plan and reproduction commands](PLAN.md). Hybrid combines a two-layer
  width192 MLP of the full candidate/noisy future/context with three temporal
  convolution stages32/64/64. Their pooled features feed one96-unit fusion
  layer and the existing eight UCD heads. No extra scores, losses, normalized
  coordinates for bcap, or modifications to G or the shared sampler.
- YAML selects d_architecture=hybrid, d_width=192, d_temporal_width=32;
  geometry_mode=continuous. All other recipe settings match the MLP control.
- Ten unit tests pass, including both hybrid branches' nonzero finite bcap
  parameter gradients, input gradients, double backward, UCD label exclusion,
  and hybrid concat/one-shot compatibility. Full loss/schedule/prior and
  evaluation implementations are unchanged.
- All four completion certificates and current source fingerprints verified.
  Both GPUs used CUDA. Sources stayed fixed throughout the two stages.
- Both endpoint route plots visually inspected. Report gallery references
  checked; exported GIFs decode all64 frames. Viewer code unchanged; no new
  browser rendering test performed. Raw samples and EMA inference checkpoints
  are under ignored results/trajectory/hybrid/.
- 1k SW1 favored MLP (.0486 vs .0547); neither arm was dropped based on scout
  rank. Both reached the same prescribed10k budget with no checkpoint selection.
- Single-run configuration comparison, no seed sweeps or significance claim.
  GPU execution is not forced deterministic. The control here (.0385 SW1,
  76.7% extrapolation validity) is close to the prior continuous control (.0390,
  80.3%), but earlier discrete controls showed greater validity variation.
  Extrapolation still means just one geometry with two preference classes.
- Figures and analytic diagnostics apply to this synthetic two-route,
  three-coefficient task. Real human motion needs new data splits, metrics,
  physical diagnostics and conditioning choices; toy thresholds do not carry over.

[All four runs](all_runs.json) · [10k machine leaderboard](confirm_10k/leaderboard.json)
