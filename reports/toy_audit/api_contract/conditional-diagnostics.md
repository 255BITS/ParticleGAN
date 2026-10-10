# Conditional, behavior and native diagnostic API reform v1

The [completed common-runner campaign](RUN_REPORT.md) supplies full-budget
training receipts and goal GIFs. The software-validation results below remain
separate from that campaign's learned-model outcomes.

These are **42 runnable new variants for 31 retained historical questions**:
23 conditional variants and 19 native/control/reconstruction variants. Historical
names, ratings, source receipts and scientific statuses remain unchanged. This
adds implementations and calibrated binary gates, **not new trained qualification**.
The root API runner records actual update-boundary observations, goal-reference/
output GIFs and source/runtime/recipe bindings separately.

## Executed API and scope

Every variant advances an actual current public optimizer or policy. The direct
conditional hosts use `Recipe.make_optimizers`, `Recipe.make_loss` (RpGAN),
`Recipe.make_critic_penalty` (KA2) and the public full-horizon LR scheduler.
Stochastic hosts have a 128-component, 8D learned MoG with relative width .25,
without standardized reads, plus a 64-wide conditional MLP. Deterministic maps
have no sampled latent; source and heldout correspondence remain caller-owned.
The unused-slot host retains the shared-vector plus per-slot correction rather
than treating its two outputs as unrelated problems. Sparse serving declares a
hard top-three coordinate adapter and argmax symbol; joint mode/symbol, exact
support and continuous within-mode spread are all scored.

`--recipe auto` honors each named case's declared KA2 or native E22 recipe.
Independent Atlas row evidence/birth-death has no binding in the stochastic
conditional host and is explicitly refused. No unused Atlas controller is
presented as an executed mechanism. Native routed wrappers apply the shipped
E22Policy/RoutedRows lifecycle, sampler, bank, guard and optimizer controls.
All these protocols are CPU-only; source/recipe/config files are unchanged.

The direct conditional recipe uses zero generator-output and critic-input noise;
the paired map game compares Gaussian noise with noise plus the correct row's
error, at fixed sigma .10. Its KA2 input is the joint output/context coordinate
vector; real/fake rows share identical contexts. This is a new declared caller
host, not a claim to reproduce an archived critic/architecture. Checkpoints
include actual optimizer/model/gradient/mode/RNG owners and fixed sampler inputs.
External `max_steps` limits execution without shortening the declared schedule.

## Necessary scope distinctions

- **PR196 variants train with oracle complete labels.** They are supervised
  conditional-imputer baselines, not evidence that a MisGAN learns the complete
  distribution from incomplete data alone. That original question remains
  unverified. Each actual mask mechanism trains with its original incomplete
  context draw; evaluation deliberately oversamples a supported single-sensor
  mask on 16 independent heldout rows. Exact Bayes posterior mode weights,
  missing-coordinate means/variance and orthogonal 8D lift noise are scored.
  The 2D projection shown in the GIF cannot hide failure in the other coordinates.
- **PR153 is fully observed six-state dynamics plus the exact source renderer.**
  It retains ID and ceiling-bounce OOD starts and free-running 5/20/50-step dream
  gates, but does not establish image-only latent world-model learning. Predicted
  states are fed back; no oracle next state enters the learned rollout.
- Source00/01 train a direct clean conditional sparse/symbol kernel. Source02/03
  train a direct conditional posterior on four frozen observation/time panels
  per class. These do not execute or qualify an iterative DDGAN chain.
- Source04/05 and 06/07 preserve their exact conditional samplers and distinct
  discrete/continuous training-context exposure. Complete 64-point obstacle
  trajectories and joint 6D local transitions remain different questions.
- Source08/09 preserve the exact affine/swirl targets and paired heldout gate in
  a new direct conditional model. Source13 isolates previous-command dependence
  through `tanh(-2.2*previous)`; it does not replay the archived 250-step AE
  pretraining or 400-step latent-joint game.
- PR227 receives a **fresh** relative edit-error bound plus a signed beneficial
  code-ablation gate under this arm's trained critic and common heldout noise.
  The original four-cross-trained-judge campaign and frozen source cards are not
  reused. A negative zero-minus-live delta fails, irrespective of its magnitude.
- PR231 receives a **fresh** recipient-error bound relative to zero recipient
  output. Its host has no outer source identity: using unchanged source-host
  error as denominator would falsely pass zero output. Whole additive FiLM
  erasure remains different from PR227 H/b neutralization. Historical trained
  evidence remains NO_FROZEN_GATE; a new bypass quality PASS would not prove
  a causal damping repair.
- PR224 is a constructed native optimizer causal unit. Its full 96-update gate
  fails the stiff native release and passes cancellation plus safe geometry.
  It uses one terminal observation. Learned gates otherwise require the root
  runner's five post-update observations and the full named budget/count.
- Source14 has four retained API questions: paired fit, two-site support, moving
  paired edit and complete activation replay. Moving requires three actual
  orientation endpoints and one terminal observation at update 1500; earlier
  observations cannot pass an unexecuted shift. Replay uses explicit public API
  initialization with the zero-context-harm guard, and compares full optimizer/
  controller/model/RNG state and gradients against a separate actual native
  baseline. A software replay PASS alone cannot fill a learned-quality cell.

## Gate calibration and observer validation

The focused suite has 93 tests. It runs **two exact API updates on each of 42
variant/baseline pairs**, with an observation inserted only on one trajectory,
then compares all retained state. Each of 23 conditional gates also passes an
exact-law or exact-pair witness at its declared evaluation count. These are
software/evaluator controls, never learned-model receipts. Negative controls
reject collapsed posterior means/widths, wrong mode-symbol coherence, missing
route mass, shuffled actions/trajectory identities, neutral or unused-slot
motion, half-strength identity swaps, backward circle control, persistent sprite
states, off-plane imputation error, omitted live adversarial forces and harmful
code direction. Stiff 96-update native/cancel/safe controls and complete routed
replay execute the actual public paths.

Final focused validation: **93 passed in 12.77s**. The common runner's metadata
validator separately accepts all 42 definitions and their 31 historical IDs.
These results validate software execution and evaluator controls, not full-budget
learned quality. The raw test log is outside Git at
`/ml2/hypergan/toy-api-conditionals-tests.log`.

Posterior and imputation variants declare 4096 draws per panel/query; these
counts prevent a 1024-draw max-over-rows gate from rejecting genuine Bayes draws
through finite-sample TV/variance variation. Bounds were not weakened and no
model/seed search occurred. Other stochastic conditional defaults use 1024 draws
per heldout context. Exact finite paired panels, control episodes and native
report grids declare their actual counts instead; a caller's sample-count option
does not redraw those fixed panels. All numerical observations are flat and finite. Genuine circle
successor overflow stops only the evaluation prefix, explicitly fails completed
horizon/nonfinite bounds and shows actual finite states; it is never clipped,
reset to the oracle, interpolated or counted as completed 1024-step playback.
Native checkpoint equality preserves exact tensor bytes, including unchanged
NaN sentinel fields in noise-tester blocks; NaN numerical scoring cannot pass.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/pr155-e22-venv/bin/python -m pytest -q \
  tests/test_toy_api_conditionals.py tests/test_toy_api_diagnostics.py
```

After integration with the common runner, full declared commands include:

```sh
python -m benchmarks.toy_audit.api_run --case api-circle-controller \
  --device cpu --recipe auto --output /tmp/toy-api-circle-full
python -m benchmarks.toy_audit.api_run --case api-mask-mcar-p20 \
  --device cpu --recipe auto --output /tmp/toy-api-p20-full
python -m benchmarks.toy_audit.api_run --case api-routed-moving \
  --device cpu --recipe auto --output /tmp/toy-api-moving-full
```

An explicit `--steps 16 --eval-samples 128` acquisition is a shorter separate
API/media cohort and cannot qualify the declared full protocol. Causal/moving
GIF acquisition must reach the declared release/shift event to illustrate it.
Raw arrays, checkpoints, run logs and per-step streams stay outside Git; compact
source-bound receipts and actual observation GIFs are published by the root
runner separately. No new trained result is asserted by this definition report.

## Pinned pure excerpts from unmerged sources

The new API path includes pure sampler/oracle excerpts, not imported old trainers.
Their originals remain unmerged library sources; no deletion or library port is
claimed. Excerpts deliberately exclude unused training/metric imports.

| New audit sampler | Original proposal source | Original Git blob |
|---|---|---|
| `api_sources/circle.py` | PR22 c85af40f1e30339702dcb6143830fca15d5334f2:`lib/circle_transition.py` |1d662989605d9e5aa856e51dd174685f179fb481|
| `api_sources/sprite.py` | PR153 3972788a2395340596fd6c9f3859978b3d072d24:`lib/sprite_animation.py` |c76e9f8069a97d100f6b49f3aa2f37ece14b0741|
| `api_sources/misgan.py` | PR196 5efc80a3dfb2baffc83f54c61cab07ccfc9826f5:`lib/misgan.py` |b92f379bac7e0f21016242414c6ab300e4b3c277|

All additional shipped targets and native callers come from current develop
664ce464e3add5c65d06c8b324c6f9892e644eec. Actual import/file hashes are bound by
each new root-runner receipt, not inferred from an older campaign's package hash.
