# Frozen initial prior: Gaussian continuous-learning diagnostic

Freezing the initial prior does not solve Gaussian acquisition and retention at the selected constant BCAP rates. Past extrapolation gives a passing final shifted snapshot but cannot retain full quality. All three fixed-prior ring arms also fail.

This is an explicit fixed-prior control cohort. It is based on the public
extrapolation implementation in [PR317](https://github.com/255BITS/ParticleGAN/pull/317),
which is a prerequisite for this PR; PR317 remains open. The user requested three
independent investigations of the 1D Gaussian. This PR isolates prior learning.

## Question and unchanged conditions

Can G/D acquire, retain and adapt to the Gaussian when prior motion is removed?
Compare the unchanged alternating, simultaneous and extrapolation-from-the-past
update timing with a MoG whose initial location table stays fixed. Timing remains
an explicit comparison; prior learnability is a separate task-control cohort.
The learned-prior controls are the original [PR317 readout](../bcap-past-extrapolation/README.md)
and source-bound receipts listed in [protocol.json](protocol.json). They were
referenced without rerunning or relabeling them. No ordinary task, qualification,
selection or generated leaderboard is replaced by this diagnostic.

Both tasks retain 256 uniform MoG locations, sigma 0.1, initialization scale 1,
no standardization, batch 128, seed 0 and public deterministic initialization.
The Gaussian is N(2,.5²), z_dim 2, G/D width 32/depth 2, critic Fourier 2.
Ring16 retains z_dim 4 and width 64/depth 2. The selected BCAP recipe uses RpGAN
and cap/coefficient 1, full dualnorm, zero momentum, constant G/D rates .012/.018;
nominal prior rate .03 stays in the recipe but has no optimizer group in this
fixed-prior cohort. Both LR floors are 1; no annealing, EMA, prior regularizer or
training noise. One whole recipe and control law is used across both tasks.

The public initializer intentionally retains buffers. The diagnostic temporarily
exposes the fresh initial prior tensor as a trainable parameter, invokes
`FormulationContext.initialize(prior, component="prior")`, and converts it back
to a buffer before optimizer construction. A process-local factory reuses this
initialized prior during public `GANTrainer` construction. No fitted checkpoint,
extra constructor draw or hidden initialization-stream reset is used. Exact initial
model and named-stream proofs match the original learned-prior step 0 states.
Metadata explicitly records the initializable view and final fixed buffer.
`learned_locations` is removed only from the explicit diagnostic task/candidate
requirements; every other capability requirement is retained.

Every scientific update executes through public `GANTrainer`; the orchestration,
scorer and cadence are reused from the unchanged prior study. The past-lookahead
cache contains G/D only because the prior is a buffer. [Gidel et al. equations 20–21](https://arxiv.org/pdf/1802.10551)
are applied to the existing normalized network field, as in PR317. This diagnostic
does not test raw-gradient SGD, Adam or a stronger prior spring.

## Frozen gates and cost

Stationary updates run through 4000. Gaussian acquisition at 1000 requires five
terminal full passes, then all 72 scheduled checks through 4000 must pass.
Ring acquisition at 1600 similarly requires five terminal full passes and all 144
later checks. Cadence remains 24 per original 1000/400-update block respectively.
Gaussian full bounds: 4096 samples, finite fraction 1, mean error<=.2 target sigma,
std ratio [.8,1.2], analytic target CDF KS<=.05. Ring retains all original coverage,
precision, balance and component covariance bounds; no moments-only substitution.

Each Gaussian checkpoint continues from 4000 to 6000 after target mean2→3 with
sigma 0.5 unchanged. There is no history reset. Reacquisition at 5000 requires five
terminal full passes; all 24 later checks must pass. A separate pre-shift checkpoint
copy receives no updates and uses identical evaluation draws. A continuous learner
pass requires successful stationary acquisition/hold plus shift acquisition/hold.
Shift arms execute even after stationary failure to retain their diagnostic value.

Reservation: nine new trials, 30000 updates, 4500 seconds; six stationary trials at
600 seconds each and three 2000-update shifts at 300 seconds each. Zero scientific
retries, seeds, tuning or additional continuation. Training, model sampling,
initialization and neural fixtures use CUDA with no CPU fallback. The exact
archived CPU target-generation law, numerical scoring and media rendering remain
unchanged. Their use is recorded separately from GPU neural execution.

## Results

Every declared combined acquisition/strict-hold gate fails. The fixed-prior past arm improves scalar shape, but no stationary scalar arm ever achieves five consecutive full passes. All ring arms have 0/240 full checks.

Completed cost: 30,000 new updates, 337.193 loop seconds, 3,840,000 real training examples, nine trials and zero retries. Loop time includes scheduled evaluations and excludes setup, restore, serialization and publication.

### Stationary Gaussian

| Cohort | Timing | Acquisition | Hold | Longest streak | Final mean error / sigma | Final std ratio | Final KS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Learned (original) | Alternating | FAIL | 1/72 | 1 | 0.27536 | 1.74796 | 0.09594 |
| Learned (original) | Simultaneous | FAIL | 0/72 | 0 | 0.63305 | 2.45104 | 0.29329 |
| Learned (original) | Past extrapolation | FAIL | 3/72 | 2 | 0.63036 | 0.95865 | 0.26799 |
| Frozen initial | Alternating | FAIL | 3/72 | 2 | 0.25358 | 1.05720 | 0.11345 |
| Frozen initial | Simultaneous | FAIL | 0/72 | 0 | 0.08858 | 1.47128 | 0.14677 |
| Frozen initial | Past extrapolation | FAIL | 7/72 | 1 | 0.16522 | 0.93468 | 0.07294 |

### Stationary ring16

| Cohort | Timing | Acquisition | Hold | Final covariance error | Final min eigen ratio | Final HQ |
| --- | --- | --- | --- | --- | --- | --- |
| Learned (original) | Alternating | PASS | 144/144 | 0.56385 | 0.27828 | 0.97070 |
| Learned (original) | Simultaneous | PASS | 75/144 | 0.54616 | 0.18197 | 0.97754 |
| Learned (original) | Past extrapolation | FAIL | 115/144 | 0.25533 | 0.65523 | 0.94824 |
| Frozen initial | Alternating | FAIL | 0/144 | 3.63940 | 0.30582 | 0.73047 |
| Frozen initial | Simultaneous | FAIL | 0/144 | 3.36449 | 0.55049 | 0.70703 |
| Frozen initial | Past extrapolation | FAIL | 0/144 | 5.27132 | 2.76880 | 0.68115 |

Each fixed-prior ring endpoint covers all 16 modes and keeps mode mass approximately balanced, but outputs are too diffuse: covariance error exceeds .85 and HQ is below .85. This control loses the learned prior’s cluster precision.

### Gaussian mean shift, 4000→6000

| Frozen timing | Reacquisition | Hold | Full checks | Longest streak | Final mean error / sigma | Final std ratio | Final KS | No-update final KS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Alternating | FAIL | 0/24 | 2/48 | 2 | 0.35539 | 1.12578 | 0.17680 | 0.72798 |
| Simultaneous | FAIL | 0/24 | 0/48 | 0 | 1.57269 | 1.62144 | 0.47287 | 0.61616 |
| Past extrapolation | FAIL | 1/24 | 4/48 | 2 | 0.03461 | 0.96094 | 0.03256 | 0.73699 |

The fixed-prior past arm’s final shift endpoint passes every full bound (KS .03256), and the no-update copy fails all 48 checks. It adapts through G/D learning, but has only one passing check in the strict retention interval and no five-pass window. A successful final snapshot does not pass the continuous-learning contract. Original learned-prior shift controls pass 0/48 checks under their own archived identity.

## Interpretation

Freezing the prior alone does not repair the selected constant-rate Gaussian recipe. Prior motion is not the only obstacle under this recipe: the residual G/D game still drifts with every timing variant. The prior can still contribute to instability, and these controls do not identify a unique cause.

Past extrapolation with a frozen prior improves the final stationary KS from .26799 (learned) to .07294 (fixed), and gives a passing shifted endpoint. This is useful evidence of an adaptable G/D game, but the strict acquisition/retention failures reject this exact recipe as a continuous learner. The next comparison should examine the independently investigated magnitude-sensitive G/D update rule; compare it with the learned-prior magnitude change before deciding whether to combine mechanisms. A stronger position spring is not supported by this role control. No additional round is launched here.

The same-network affine capacity control with this original initial MoG already
[passes 24/24 scalar checks without training](../gaussian1d-diagnosis/README.md).
The fixed-prior experiment therefore tests optimization stability rather than
whether these tensors can represent the Gaussian. It does not isolate G from D,
and failure of this exact fixed-prior recipe does not exclude other fixed-prior
optimizers. Retain the original passing learned-prior ring recipe.

## Verification and artifacts

Eight software checks pass, including all three timings on both tasks: exact
initial tensors/streams, absence of prior optimizer/cache state, immutable prior,
and bit-exact CUDA continuation. Scoped host binding restores after errors and
CPU execution refuses before creating output. The [verification](verification.json)
recomputes every saved numerical observation and gate, while
[restore-proof.json](restore-proof.json) checks every final CUDA context, constant
rates, prior immobility and the no-update control. These add no scientific updates
or model sampling. Existing source-bound [scorer controls](../bcap-past-extrapolation/scorer-controls.json)
retain their original identity and unchanged scorer/target/gate bindings.

The [compact metrics](results.json), [protocol](protocol.json),
[provenance](provenance.json), [stationary curves](stationary.svg) and
[shift curves](shift.svg) report live results, including failures. Full curves,
stdout, checkpoint tensors and saved outputs remain in the ignored raw directory
and immutable archive. These tables are diagnostic readouts; the single current
technique leaderboard remains [technique-inventory.md](../technique-inventory.md).

Actual-training GIFs use nine saved live GPU observations each, fixed axes, target
samples, step labels and full numerical pass/fail captions:

| Timing | Gaussian stationary | Ring stationary | Gaussian shift |
| --- | --- | --- | --- |
| Alternating | [GIF](alternating-gaussian1d_acquisition-stationary.gif) | [GIF](alternating-ring16_acquisition-stationary.gif) | [GIF](alternating-gaussian1d_acquisition-shift.gif) |
| Simultaneous | [GIF](simultaneous-gaussian1d_acquisition-stationary.gif) | [GIF](simultaneous-ring16_acquisition-stationary.gif) | [GIF](simultaneous-gaussian1d_acquisition-shift.gif) |
| Past extrapolation | [GIF](extrapolation_from_past-gaussian1d_acquisition-stationary.gif) | [GIF](extrapolation_from_past-ring16_acquisition-stationary.gif) | [GIF](extrapolation_from_past-gaussian1d_acquisition-shift.gif) |

## Reproduction

Scientific freeze: `33ca3db28cd3a01336c8f183987184be21379220`.
Protocol SHA256: `6e5601f349c8e62307ff94d141b5dd7a896d407fdbd2e9a95fdb4b5da4c21073`.
Hydrate original source-bound artifacts listed in the protocol before executing.
New output paths are exclusive; this command launches the complete declared budget.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/usr/bin/python -u -m benchmarks.toy_audit.gaussian_frozen_prior run \
  --device cuda:0 --output runs/api/gaussian-frozen-prior-v1 \
  > runs/api/gaussian-frozen-prior.run.log 2>&1
tail -f runs/api/gaussian-frozen-prior.run.log
```

Per-cell logs live beside checkpoint directories and can also be tailed.
Saved-state reproduction and publication launch no training:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/usr/bin/python reports/forge/gaussian-frozen-prior/verify.py \
  --device cuda:0 --raw runs/api/gaussian-frozen-prior-v1
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/gaussian-frozen-prior/publish.py \
  --raw runs/api/gaussian-frozen-prior-v1
```
