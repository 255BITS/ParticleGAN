# Human motion transfers quickly, but the futures still jitter

All four runs completed on both RTX A6000 GPUs: MLP and hybrid at1k, then both
at10k. Total training summed across runs: **3.92 GPU minutes**. Endpoint-only
evaluation; no training result inspected before its entire pair finished.

[Interactive comparison](confirm_10k/index.html) ·
[Hybrid animation](confirm_10k/hybrid/futures.gif) ·
[Hybrid config](../../../configs/motion/confirm_10k/hybrid.yaml) ·
[Plan / commands](../PLAN.md) · [All runs](all_runs.json)

## Recommendation

Keep **hybrid as the starting baseline**. It has the better validation
class-marginal SW1 (.1828 vs .1902), lower validation ADE, less bone distortion,
and substantially lower acceleration. This does not make it a universal winner:
MLP is cheaper, more varied, and narrowly better on test single-draw ADE and SW1.
No strong significance claim from this small subject holdout.

**Neither model beats holding the last pose on single-sample test prediction
error.** Hybrid best-of-eight ADE only barely improves on that deterministic
reference, with an eight-draw selection advantage. We have a working generative
transfer baseline, not convincing multimodal forecasting yet. No learned-particle
versus Gaussian-latent ablation was performed, so this round cannot attribute a
benefit to particles or quantify coverage of plausible alternative futures.

## 10k comparison

Errors/distances use source coordinates divided by the training-only scalar
.2894682. They are not centimeters. Bone error is mean absolute relative change
from the last observed skeleton; lower is better. Diversity is mean pairwise
joint distance among8 samples, with no known target value. Acceleration includes
the prefix/future boundary and uses per-frame units.

| Model | Val SW1 ↓ | Test SW1 ↓ | Single ADE ↓ | Best-of-8 ADE ↓ | Diversity | Bone error ↓ | Accel. ↓* | Train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Hybrid D** | **.1828** | .1487 | .4479 | **.3721** | .2714 | **10.7%** | **.1462** |119.7|
| MLP D | .1902 | **.1467** | **.4447** | .3794 | .3174 |11.4%| .2068 |**93.3**|
| Held pose | .1936 | .1572 | **.3784** | — |0|0%| .0044 |—|
| Constant velocity | .3385 | .2867 | .5917 | — |0|23.1%| ~0 |—|

*Lower acceleration alone is not better motion. Recorded acceleration is .0523
and speed .0676; stationary/constant-velocity outputs artificially minimize it.
Hybrid acceleration is2.8x recorded and MLP4.0x; hybrid speed .1220 and MLP .1619
also exceed recorded. These support excessive temporal variation, rather than
interpreting all MLP diversity as useful multimodality.

Hybrid reduces mean acceleration by29.3% versus MLP, but costs28.4% more training
time. Both improve substantially in pose consistency from1k, where mean bone
errors were20.4%. At10k,13.8% of hybrid bone/frame observations deviate by more
than20% from the observed length, versus15.8% for MLP. Recorded bone error is
near zero in these source arrays; passing this diagnostic alone would still not
establish realistic joint angles or contact.

Boundary acceleration is .2545 hybrid / .2781 MLP / .0550 recorded. The first
future frame moves .2422 hybrid / .2664 MLP from the last observed pose, compared
with recorded motion on this split (see summary reference metrics). There is a
large remaining continuity problem. The models receive prefix context but do
not clamp or overwrite their first future frame.

Class SW1 uses only one random sample per prefix and is averaged over actions.
It compares action marginals across different prefixes, not a known conditional
future distribution. Validation has just one held-out subject and4–16 evaluated
clips/action. Test contains two subjects and8–16 evaluated clips/action. Small
SW1 differences should be interpreted cautiously. Recorded-vs-itself SW1=0 is
a sanity check, not an independently estimated real-vs-real noise floor.

## What the animations show

The viewer presents a recorded skeleton followed by four random futures sharing
the same observed prefix. All panels share camera and scale, with a gray recorded
reference behind generated blue poses. Action selection and camera rotation are
interactive. GIFs show walk and boxing with three samples each. These are fixed
first evaluated clips and first draws, not examples chosen for appearance.

Generated outputs have recognizable body structure and vary across samples.
Some limbs differ in length or pose from the reference; numerical diagnostics
also identify excess temporal jitter. Both10k viewers were inspected in Chromium,
including an intermediate future frame. All four exported GIFs decode24 frames.
No full perceptual study or guarantee of physical plausibility is implied.

With within-action prefixes shuffled and all sampling randomness fixed, generated
joint displacement averages .6836 hybrid / .6616 MLP. G responds to observation
context. This intervention does not prove that the response is correct.

## Data, formulation and reproducibility

- Official HumanAct12 release:1,191 clips,1,103 long enough,88 excluded (<24frames).
  Eligible clips:786 train,81 validation,236 test. [Data manifest](data_manifest.json)
  records all used/excluded filenames, hashes, subjects and action labels.
- Custom subject split P01–P09 / P10 / P11–P12 before windowing. All12 actions
  available in each split, no exact duplicate arrays detected. No claim of
  matching a published benchmark protocol. Keep the raw arrays local.
- Train samples action uniformly, clip uniformly within action, start uniformly
  within clip. Evaluate192 train /77 validation /174 test midpoint prefixes,
  eight futures each. Held-pose/CV use exactly the same prefixes.
- One translation anchors the last observed root. A scalar RMS calculated only
  from training poses normalizes all splits. Subsequent root motion and source
  orientation remain. Native frame order; approximate15fps playback.
- Shared U-Net G154,504 parameters; D hybrid883,412 vs MLP881,456 (+.22%);
  learned prior640,000 parameters. Shared G and equal1.28M examples per optimizer
  phase at10k (2.56M real draws). Constant LR, four reverse steps,48 joint UCD
  heads, Gaussian step noise, unchanged Rp logistic / bcap / VICReg.
- Checkpoints save EMA inference weights, particle table, noise source, config,
  normalization scale and dataset fingerprint; optimizer resume is not supported.
- 34 tests passed across motion, trajectory and runner. Tests cover train-only
  normalization, split leakage, future-independent context, shape/sampling,
  joint UCD exclusion, particle/input gradients and bcap double backward.
  Existing optimizer loop matches master verbatim; legacy model initialization
  and state dictionaries match bitwise after parameterizing input dimensions.
- Both GPUs used successfully (hybridGPU0, MLPGPU1). All four completion
  certificates/source archives verified;10k fingerprints match current training
  sources. No training warnings or failures. Google Drive download retried a
  transient error and completed with all files verified.
- The only source change between1k and10k is a display-axis correction:
  released HumanAct12 coordinates have positive Y downward. Training arrays,
  models, losses, normalization and metrics are unchanged. Report GIFs/viewers
  regenerated from saved arrays with a separately recorded renderer fingerprint.
- No-argument `experiments/train_motion.py` uses
  [configs/motion/default.yaml](../../../configs/motion/default.yaml):10k hybrid.
  Original trajectory and CIFAR defaults remain unchanged. Historical completed
  runs use their archived sources; current-source runner reuse remains strict.

## Next experiment

Target **G's observation-to-output path**, keeping the loss and sampler fixed.
Currently the576-coordinate prefix is compressed into a64-wide conditioning
embedding. A direct last-observed-pose skip and a learned motion residual could
make preserving a person's skeleton and starting near the boundary easier.
Compare that architectural change against this hybrid baseline at equal sample
exposure. Judge boundary continuity, bone consistency, prediction error and
motion/diversity together; do not reward a collapse to frozen poses.

After a credible motion baseline, compare learned particles with Gaussian latent
draws. Do not extend training or add loss penalties solely to improve one metric.
No further experiments are queued.
