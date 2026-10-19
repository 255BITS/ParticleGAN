# AE-DDGAN composed public-API toys

This track tests whether the API's deterministic particle AE can compose with a
trained DDGAN: held-out clean inputs must reconstruct through four actual reverse
steps, while independently sampled reverse chains and conditional transitions
retain the target distribution. Correct reconstructions alone do not establish
independent generation, and a correct generated marginal does not establish
correct input/output pairs.

The proposal and numerical gates below were approved before training. Full-budget
training waits for the shared source revision to freeze. The 14 focused
software/scorer tests pass; their updates are short integration checks and grant
no trained-quality or tier qualification.

| Case | Initial placement | Frozen budget | Wall cap |
|---|---|---:|---:|
| `api-model-ae-ddgan-gaussian-chain` | Tier 1: basic composition | 2,500 updates | 180 s |
| `api-model-ae-ddgan-bimodal-chain` | Tier 2: multimodal conditional and full-chain composition | 5,000 updates | 300 s |

Tier 2 has a gate dependency on the Gaussian composition; it starts from its own
deterministic initialization and consumes no parent checkpoint. Placement is
based on the question and cost before results. These are provisional tests, not
calibrated scientific screens or recommendations of a new default.

## Frozen task and recipe

The Gaussian target is `N([.6,-.35], diag([.45²,.30²]))`. The bimodal target has
equal weights at `[-.75,0]` and `[.75,0]`, each with covariance `.18² I`.
Both use public `DDGAN` with alpha-bar `[1,.9,.5,.05,.0001]`, 128 learned MoG
centers in four latent dimensions, absolute kernel sigma `.25`, no
standardization, and batches of 64. The G/D networks are the same two-layer,
width-32 LeakyReLU networks as the separate DDGAN tasks. A width-32, two-layer
encoder maps each clean input into a four-dimensional query and raw offset.

All networks and the initial prior use public `init.deterministic_orthogonal_`,
with protocol seed 0, isolated model constructor state, and the public R2Normal
prior initializer. Separate checkpointed streams own critic/generator target
draws, timestep draws, forward noise, prior indices, kernel noise and reverse
noise. Evaluation uses disposable independent streams and preserves all model,
optimizer, gradient, policy, mode and training-stream state. Exact checkpoint
resume is tested through a subsequent real update.

The complete selected `bispf-optimized-quintic-v1` recipe is shared globally.
Only task-owned model/encoding/conditioning and resource fields are adapted;
the declared reconstruction weight remains 1. No optimizer, learning rate,
prior-update or regularizer adjustment is introduced. The recipe retains its
learned-prior policy and SBNP regularizer, applied once per generator update.
Encoder-selected prior rows are unioned with independently sampled generator
rows before the joint optimizer update. A frozen-prior recipe control verifies
that task construction cannot silently enable prior updates.

The shared DDGAN host uses the actual public `UpdatePolicy` lifecycle and
checkpoint, public adversarial loss, critic penalty, optimizer and protected
reconstruction objective. Adaptive row/serving controllers, input/output noise,
kinetic transport and strict-progress policies are rejected explicitly because
this conditional host has no frozen contract for them. The selected baseline
uses none of those controls. This study does not make a capability claim for
every API recipe or the large default `ae_ddgan` architecture.

## Training and separate sampling cohorts

Training samples independent critic and generator clean batches and uniform
per-row timesteps. Public `DDGAN.forward_pair` supplies real `(x_(t-1),x_t)`;
G predicts clean data from `(x_t,z,t)` and public `DDGAN.reverse` constructs the
fake transition. The ordinary adversarial game uses an independently sampled
prior code. A separate reconstruction objective encodes the matching clean
input through `Recipe.encode` and uses
`ParticleEncoding.reconstruction_loss(G(x_t,code,t), clean)`.

Independent generation starts from `N(0,I)` and takes four reverse steps,
redrawing independent MoG indices, local Gaussian kernel noise and reverse
Gaussian noise at every step. Eight held-out conditional panels retain the
separate DDGAN task's transition-distribution gates.

Paired reconstruction encodes each held-out clean input once and holds that
deterministic AE code through steps `4,3,2,1`, from an independent Gaussian
start with fresh reverse noise at each step. This is explicitly a different
sampling law from independent generation. The shuffled-code counterexample
rolls code rows while reusing exactly the same Gaussian start and reverse noise;
only input-code identity changes. No clean target enters G except through the
declared encoder code.

## Gates and controls

Each case uses 2,048 samples per evaluation panel, 24 fixed post-update
observations and five passing terminal observations. Every independent-chain
and conditional-panel gate from the matching DDGAN task remains required:
projection CDF error <= `.12`, normalized mean error <= `.25`, and whitened
covariance eigenvalues in `[.65,1.4]`. Bimodal generation additionally requires
HQ >= `.95`, mode-mass TV <= `.10`, worst local mean error <= `.5` sigma, and
local covariance eigenvalues in `[.5,1.5]`.

Paired residuals are whitened by the exact analytic target covariance. Required
paired NMSE is <= `.10`; the worst coordinate normalized RMSE is <= `.25`;
shuffled-code NMSE divided by paired NMSE is >= `3`; and the reconstruction
must execute exactly four actual reverse steps. The worst-axis gate is stricter
than aggregate NMSE and catches a neglected input coordinate.

Scorer controls accept exact paired oracle outputs. They reject correct
marginals with shuffled input/output pairs, input-independent collapse and
nonfinite arrays. A clean-code oracle passed through the actual four-step
public reverse implementation reproduces the exact input. Shared DDGAN controls
separately validate analytic final/transition oracles and reject mean collapse,
wrong conditions, missing modes, missing within-mode variance and nonfinite
output. Tests also verify shared paired/shuffled noise, fixed code through the
paired chain, and freshly redrawn independent-generation codes.

## Reproduction and publication

```sh
python -m pytest tests/test_toy_api_model_ae_ddgan.py -q
python -m benchmarks.toy_audit.api_run \
  --case api-model-ae-ddgan-gaussian-chain --device cuda:0 \
  --output /nvme/dev/particlegan/artifacts/api-model-toys/ae-ddgan/gaussian
```

The root Forge study owns the bounded full runs, actual-training GIFs, compact
receipts and existing leaderboard update. Tail each run's local stdout/JSONL
path from that study's run manifest. Raw traces, checkpoints and observation
arrays stay under the external artifact root, outside Git. Do not create a
second generated leaderboard for these tasks.

Historical [denoising toy research](../../denoising-toy/README.md) remains under
its original source, initializer, seed, architecture and sampling law. These
new small analytic cases and the explicit learned AE composition neither
regrade those results nor claim the historical 100-Gaussian question is solved.
