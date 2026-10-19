# AE-DDGAN composed public-API toys

This track tests whether the API's deterministic particle AE can compose with a
trained DDGAN: held-out clean inputs must reconstruct through four actual reverse
steps, while independently sampled reverse chains and conditional transitions
retain the target distribution. Correct reconstructions alone do not establish
independent generation, and a correct generated marginal does not establish
correct input/output pairs.

The Gaussian composition completed 2,500 updates and **FAILS distribution
quality**, although its separately scored paired reconstruction meets every
paired bound throughout the final five observations. The bimodal full run is
**BLOCKED**; its declared three-update software prefix is **INCOMPLETE** against
the 5,000-update task. The 14 focused software/scorer tests pass and the actual
public-API chains execute, but component correctness supplies no trained-quality
or tier qualification. The proposal and gates were frozen before training.

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

## Frozen research results

The original receipts bind source
[`bee9f06ed21695e518c1cb702e12a385ab9389e8`](https://github.com/255BITS/ParticleGAN/tree/bee9f06ed21695e518c1cb702e12a385ab9389e8),
the one global `bispf-optimized-quintic-v1` recipe, protocol seed 0, public
deterministic initialization, clean live serving, CPU and one Torch thread.
Source remained unchanged throughout execution. The [shared frozen protocol](protocol.json)
and [compact publication](../../toy_audit/api_contract/model_modes/publication.json)
retain the complete binding, results and saved artifact identities. The later
report-only export repair used the original saved receipts and GIFs; it made
no new model updates or observations.

**Gaussian composition: FAIL**, 2,500/2,500 updates, 24 post-update checks,
2,048 draws per panel, 24.14754085500317 seconds. Actual updates and endpoint
metrics remained finite, and paired reconstruction executes exactly four
reverse steps. No check passed the complete joint gate (0/24), including every
one of the final five.

| Endpoint metric | Observed | Gate |
| --- | ---: | ---: |
| Independent final-chain projection CDF error | 0.18311283346834928 | <=0.12 |
| Independent final-chain normalized mean error | 0.5973940376636387 | <=0.25 |
| Independent final-chain minimum covariance eigenvalue | 1.175797625707041 | >=0.65 |
| Independent final-chain maximum covariance eigenvalue | 1.525788308358236 | <=1.40 |
| Worst conditional projection CDF, `t1_q1` | 0.2711668099394876 | <=0.12 |
| Worst conditional normalized mean, `t1_q1` | 0.5111426660065264 | <=0.25 |
| Minimum conditional covariance eigenvalue, `t1_q1` | 0.44320869089488796 | >=0.65 |
| Maximum conditional covariance eigenvalue, `t1_q0` | 1.2578323270254619 | <=1.40 |
| Paired normalized MSE | 0.030386758489284807 | <=0.10 |
| Paired worst-axis normalized RMSE | 0.1877374734714363 | <=0.25 |
| Shuffled-code NMSE / paired NMSE | 65.31286360856024 | >=3 |
| Paired reverse steps | 4 | ==4 |

The final five paired-only checks all pass: worst NMSE is
`0.04409267234218497`, worst-axis RMSE at worst is `0.2134214730917655`,
minimum shuffled gain is `45.17452524795841`, and every reconstruction has four
reverse steps. These observations support input-code identity preservation in
this declared reconstruction cohort. They cannot qualify independent generation.

The exact endpoint failed bounds are `final_covariance_eigen_max`,
`final_normalized_mean`, `final_projection_cdf`, `t1_q0_covariance_eigen_min`,
`t1_q0_normalized_mean`, `t1_q0_projection_cdf`,
`t1_q1_covariance_eigen_min`, `t1_q1_normalized_mean`,
`t1_q1_projection_cdf`, `t2_q0_normalized_mean`, and `t2_q0_projection_cdf`.
The full verdict also fails the final-five complete-observation requirement.

[Actual Gaussian AE-DDGAN training GIF](../../toy_audit/api_contract/model_modes/media/api-model-ae-ddgan-gaussian-chain.gif)
contains nine saved training states, with independent generation, conditional
transitions and paired reconstructions shown under their distinct laws.

**Bimodal composition: full run BLOCKED; prefix INCOMPLETE**, 3/5,000 updates,
three post-update checks, 2,048 draws per panel, 0.6358917680045124 seconds.
At this software-only prefix endpoint, final-chain CDF error is
`0.4994666313419944`, normalized mean `0.16790898077722458`, covariance
eigenvalues `[0.02122680601807642,0.07935052914163286]`, HQ `0.15576171875`,
mode-mass TV `0.2314453125`, local mean error `3.8781101609372763` sigma and
local covariance eigenvalues `[0.021165410644067385,0.3121805873395517]`.
Worst conditional CDF is `0.9954839518035866`, normalized mean
`4.476929674178463`, minimum covariance eigenvalue `0.0890677536788668`, and
maximum `1.057986852881`. Paired NMSE is `0.9872169089736147`, worst-axis
RMSE `0.9946277601080216`, shuffled gain `1.0070487133206956`, with four
actual reverse steps.

Prefix failed bounds are `final_covariance_eigen_min`, `final_hq`,
`final_local_covariance_min`, `final_local_mean_sigma`, `final_mode_mass_tv`,
`final_projection_cdf`, `paired_nmse`, `paired_shuffled_gain`,
`paired_worst_axis_nrmse`; `covariance_eigen_min`, `normalized_mean` and
`projection_cdf` for each of `{t1_q0,t1_q1,t2_q0,t2_q1}`; and
`covariance_eigen_min` for `{t3_q0,t3_q1}`. The raw API FAIL also records that
the full default budget/cadence and final-five passing observations are
incomplete. It is published as INCOMPLETE rather than a 5,000-update quality
failure. The [actual three-update prefix GIF](../../toy_audit/api_contract/model_modes/media/api-model-ae-ddgan-bimodal-chain.gif)
contains four saved states including initialization.

All six basic campaign tasks completed, with three PASS and three FAIL. The
predeclared global basic-task gate and the failed Gaussian parent block both
full bimodal runs. Initial Tier 1/Tier 2 placements and gates remain unchanged;
no task is moved after failure. These CPU results are separate from historical
CUDA/noisy-serving or larger-budget evidence.

The current evidence shows a useful split: paired input reconstruction works
under its fixed-code law, while independent-prior and low-noise conditional
sampling still fail. A useful next analysis is to compare saved encoded-code
and independently sampled prior-code moments, decoder sensitivity to each, and
the endpoint low-noise conditional residuals. This is a saved-state diagnostic
recommendation, not an explanation proven by the run and not a request for
another training arm. Preserve both the successful paired submetrics and the
failed full gate in the single existing [Forge leaderboard](../technique-inventory.md).

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
