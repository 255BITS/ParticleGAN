# AE coverage: frozen proposal and software acceptance

This adds two prospective public-API tasks using the selected global
`bispf-optimized-quintic-v1` recipe unchanged. Both enter the existing
`discriminator_stability` view as required Tier 1 questions. Tier calibration
remains provisional; failure must remain visible and cannot justify weakening a
gate or silently moving a task out of the counted requirements.

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
These tests run at most two updates per fixture and supply no trained-quality
qualification. Full training will run only after the combined source and task
contracts are committed. Raw stdout/JSONL/checkpoints belong under
`/nvme/dev/particlegan/artifacts/api-model-toys/ae`; final compact receipts and
actual-training GIFs will be linked here after execution. The existing single
Forge leaderboard remains the comparison authority.

Declaration validation compares task gates, target, prior, architecture, resources,
budget and cadence against executable constants; changing task JSON cannot change
its own expected scorer. Focused regression tests reject those contract changes
and incompatible injected recipes. No full scientific training has run yet.
