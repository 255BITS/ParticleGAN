# AE coverage: completed four-mode toy readout

Both AE cases **PASS** their full frozen 1,200-update protocols using the
selected global `bispf-optimized-quintic-v1` recipe. Each passes all 24
post-update scoring checks, including the required final five at updates 1000,
1050, 1100, 1150 and 1200. Source is
`bee9f06ed21695e518c1cb702e12a385ab9389e8`, protocol seed 0; source bindings
remained unchanged throughout execution. Both remain required Tier 1 questions
in the existing `discriminator_stability` view. Scientific tier calibration
remains provisional and this bounded research is no public-default promotion.

| Completed task | Final reconstruction MSE | Final paired mode agreement | Terminal passing suffix | Actual training |
|---|---:|---:|---:|---|
| Reconstruction-only AE control | .004427 | 1.000000 | 24/24 checks | [goal GIF](../../toy_audit/api_contract/model_modes/media/api-model-ae-reconstruction.gif) |
| AE-GAN: reconstruction + independent generation | .009270 | .997070 | 24/24 checks | [goal GIF](../../toy_audit/api_contract/model_modes/media/api-model-ae-gan-multimodal.gif) |

The AE-GAN endpoint independently generates all four modes with precision
`.996826`, mass TV `.117676`, component eigenvalue ratios `.276335` to
`1.048481`, and component centroid RMS `.635961` target sigma. Every declared
gate passes. Across the final five checks, the worst precision is `.985107`,
maximum mass TV `.117676`, minimum eigenvalue ratio `.276335`, maximum
eigenvalue ratio `1.601173`, maximum centroid RMS `.963244`, maximum
reconstruction MSE `.009270`, and minimum paired mode agreement `.997070`.
The centroid gate has the smallest final-window margin; these results establish
the declared tolerances rather than exact distribution recovery.

The reconstruction control reaches lower reconstruction error but its independent
prior output has precision `.632813`, maximum eigenvalue ratio `21.423988`, and
centroid RMS `3.681531` target sigma. Those generation diagnostics are outside
the stronger AE-GAN bounds and do not belong to the control's narrower pass
criterion. The control therefore demonstrates why reconstruction quality alone
cannot establish generative quality. Keep both tasks at Tier 1: the same selected
baseline passes their unchanged declarations, and they test distinct API behavior.

The [compact publication](../../toy_audit/api_contract/model_modes/publication.json)
retains metrics, raw-receipt identities, guards, source manifests and artifacts.
Both runs use the identical resolved recipe SHA256
`29b9ff4ba509a6a934dbd9994f8f11de8709dbc28e0e7b4ea22cdc5ba518aa57`:
learned prior locations, SBNP prior regularization `.01`, DualNorm with quintic
convolution rule, learning rate `.012`, prior multiplier `2.5`, critic multiplier
`1.5`, and reconstruction weight `1`. The task-owned horizon is 1,200; no
model-specific rate or regularizer tuning occurred. These are clean, live
four-Gaussian vector results with an independent MoG prior, not image-model,
noisy-serving, VAE, or DDGAN evidence.

Execution used CPU, Python 3.12.13 and PyTorch 2.14.0+cu130, one Torch thread,
deterministic algorithms and serial autograd. Recorded costs are 6.075 seconds
for the control and 12.716 seconds for AE-GAN (18.791 total CPU seconds).
Finite-state and RNG-isolation guards pass with zero unintended RNG deviations.
G, E and learned prior each receive exactly 1,200 optimizer updates per task;
D receives zero for the explicit reconstruction control and 1,200 for AE-GAN.
The publication marks these bounded research receipts `qualification_input=false`;
they supply no automatic whole-view qualification or default promotion. The
existing single Forge leaderboard remains the comparison authority. Retain this
coverage and stop these completed protocols; further scope would require a new
falsifiable task or trainer hypothesis rather than unchanged reruns.

| Case | Falsifiable question | Frozen numerical gate |
|---|---|---|
| `api-model-ae-reconstruction` | Can the deterministic particle AE reconstruct all four noisy modes? | Reconstruction MSE ≤ .025; paired nearest-mode agreement ≥ .98. |
| `api-model-ae-gan-multimodal` | Can the same AE also produce the correct independent prior distribution? | Both reconstruction gates, four populated modes, precision ≥ .90, mass TV ≤ .15, component minimum eigenvalue ratio ≥ .10, maximum eigenvalue ratio ≤ 4, component centroid RMS ≤ one target sigma. |

The target is the equal mixture of four isotropic two-dimensional Gaussians at
`(±1, ±1)` with sigma `.08`. G and E are two-hidden-layer width-64 MLPs; D
has the same width/depth plus two Fourier frequencies. The initial prior is 64
uniform MoG locations in two dimensions, public R2 normal initialization with
standard deviation `.5`, fixed latent sigma `.025`, and no standardization.
The public deterministic initializer uses component seeds G=0, D=1, E=2 and
prior=3 within protocol seed 0. These are separate component streams, not seed
experiments.

Each task has 1,200 updates, batch size 128, 4,096 evaluation draws, 24 frozen
post-update scoring checks and a required final five-check pass. The wall cap is
600 seconds per task; the two-task maximum is 2,400 updates and 1,200 seconds.
No continuation, tuning or seed study is authorized by this proposal. Scoring
uses clean live decoder outputs from the full sampled prior, with no EMA, output
noise or best-checkpoint selection. The GIF shows held-out reconstructions and
independent prior generation against the desired target on fixed axes.

The reconstruction-only case is an explicit scoped objective control: it omits
adversarial D/G updates and still applies the same recipe-owned prior update
policy, prior regularizer and G/E optimizer. Independent generation is measured
but cannot fail this narrower gate. The AE-GAN case uses public
`Recipe.make_optimizers`, `Recipe.encode`, `ParticleEncoding.reconstruction_loss`,
`Recipe.make_critic_penalty`, `Recipe.prior_regularization` and `UpdatePolicy`.
Task adaptation sets only model/encoder mode, reconstruction weight and frozen
resources. Recipe prior learning/freezing, regularizer, learning rates,
optimizer and critic formulation are preserved. Scheduled trainers bind the
task horizon of 1,200 updates; an intentionally schedule-free trainer retains
its absent horizon. Resolved Forge recipes with conflicting task-owned fields
are rejected before model construction.

Oracle samples pass the scorer. Point-mass collapse fails the eigenvalue gate;
unequal populations fail mass balance; a mode-permuted reconstruction fails
paired agreement even with perfect marginal samples; nonfinite outputs fail.
The same checks also demonstrate why reconstruction cannot substitute for an
independent-generation gate.

Constructor, D-data, G-data, latent-index, latent-kernel, training input/output
noise and evaluation streams are isolated. Checkpoints carry the public policy,
models, optimizers, prior metadata, regularizer state, both consumed data streams,
all noise streams, the consumed prior-constructor stream and the batch-sequence
digest. Only rows actually used by a generator objective gain optimizer ownership:
reconstruction registers its encoded rows, and the AE-GAN objective also registers
its independent prior draws. The reconstruction control does not grant row
ownership to unused independent prior draws. Software tests check exact
resume in the same and fresh fixtures, evaluation-state isolation, the explicit
execution cap, matched initial networks/locations and matched batches under a
recipe-owned frozen-prior control. A public `UpdatePolicy` MoG metadata
checkpoint defect found by these tests is fixed separately in the shared API.

Historical evidence is preserved under its original scope:
[the 250-update two-anchor Forge task](../../../configs/forge/tasks/ae_gan_hold.json),
[the stricter two-anchor public caller](../../../benchmarks/toy_audit/api_diagnostics.py)
and its [published API audit](../../toy_audit/api_contract/RUN_REPORT.md).
Neither historical result is regraded or replaced by these four-mode tasks.

Software acceptance before scientific execution: 19 targeted tests passed.
These short software tests supply no trained-quality qualification. The reported
full training ran after the combined source and scientific contracts were
committed. Raw stdout, JSONL, observations and checkpoints remain outside Git:
`/nvme/dev/particlegan/artifacts/api-model-toys/research-v1/model_ae_reconstruction`
and the sibling `model_ae_gan_multimodal` directory. Raw API receipt SHA256s are
`4774b62c23ad031cc093499d0c0435f27ac9233e95f65e78c11762658b346b10`
and `f999fd367ba77e78b11a67602f3ab82963f7390471f4b91f3087770cf6c54b3e`,
respectively; their original numerical observations and GIFs were preserved when
the report exporter was repaired to rehydrate serialized recipes.

Declaration validation compares task gates, target, prior, architecture, resources,
budget and cadence against executable constants; changing task JSON cannot change
its own expected scorer. Focused regression tests reject those contract changes
and incompatible injected recipes. No gates, budgets, protocols or training
sources were changed during this readout.
