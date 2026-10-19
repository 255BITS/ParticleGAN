# Public-API particle VAE toys

This new seed-0 cohort tests the API's hard constant-KL VAE and genuine soft
categorical VAE on two balanced Gaussian anchors. Both are proposed Tier 1
requirements. Their gates remain provisional: scorer controls do not calibrate
a scientific screen or qualify any public default.

Historical [12-arm scout](../../mog-vae/README.md) and
[five-arm stability study](../../mog-vae/stability/README.md) found useful
stochastic encodings but unresolved prior coverage, mode width and late
stability on 100 modes. Those experiments retain their original optimizer,
seed, initialization, budget and evidence identity. This two-mode cohort is a
narrow, reproducible public-API regression question, not a historical rerun.

The hard posterior selects one component, adds the prior's fixed local noise,
and retains `KL=log(16)` in the reported negative ELBO. Its discrete gradient is
the API's biased straight-through approximation. Soft categorical inference
uses `KL=log(16)-H(q)` and the public leave-one-out score estimator. Removing
this variable KL would create a stochastic AE; no such model is called a VAE.

The frozen setup uses balanced means `(-1.5,0)` and `(1.5,0)`, isotropic data
standard deviation .05, a 16-by-2 learned MoG with fixed local sigma .025, and
width-32 two-layer LeakyReLU encoder/decoder/critic networks. The Gaussian
decoder observation sigma is .03. The task allows 1,200 updates of 64 inputs,
24 fixed scoring observations, 4,096 held-out inputs per observation, and five
passing terminal checks. Training uses two posterior draws; evaluation uses
eight. Live decoder means and actual Gaussian likelihood samples have separate
quality metrics and gates.

The baseline is the exact declared `bispf-optimized-quintic-v1` recipe. The task
changes encoder mode, observation model and fixed resources only; learning
rates, optimizer law, prior learning and SBNP regularization remain recipe-owned.
Both modes share the same target batches, public deterministic initializer,
network sizes, initial prior, observation cadence and allowance. The Gaussian
negative ELBO is multiplied by `.03²`, making its 2D reconstruction term MSE;
the variable categorical KL remains in the objective with its required weight.
The adversarial and recipe prior objectives are also retained.

Training executes through public `Recipe.encode`, `ParticleEncoding`, loss,
optimizer, critic-penalty and prior-regularizer components, with `UpdatePolicy`
owning schedules and update lifecycle. `GANTrainer` explicitly excludes encoder
hosts. Constructor, critic data, generator data, prior identity, prior noise,
posterior noise, input/output noise, evaluation and penalty streams are isolated
and checkpointed. Evaluation uses local generators and cannot change training
state. Encoder, generator, critic, prior, policy and optimizer state are saved
for exact continuation.

The identity diagnostic promotes the same observed predictions, targets and
posterior values to float64, calls the public negative ELBO, and compares it
with an independent Gaussian log-density calculation. It adds no model forward
or sample and leaves training arithmetic unchanged. This prevents float32
cancellation before training from rejecting a valid bound. Injected recipes
must already match the task's model, encoder, resources, observation model and
1,200-update schedule horizon; incompatible injections fail explicitly.

The gates require reconstruction MSE ≤.05; shuffled-code reconstruction ratio
≥5; posterior mutual information ≥.3 nats; local-noise second-moment ratio
between .8 and 1.2; KL identity error ≤1e-5; and Gaussian negative-ELBO identity
error ≤1e-4. Both decoder means and actual likelihood samples require at least
.85 within radius .2 of an anchor, quality mass TV ≤.15, and both modes.
Nonfinite metrics fail. Oracle samples pass; shuffled reconstruction, collapsed
posterior, missing local noise, missing KL and collapsed prior/likelihood samples
are destructive controls and must fail their relevant bounds.

Only compact final metrics, source receipts and actual-training GIFs will be
published. Bulk stdout, checkpoints and observations are kept under
`/nvme/dev/particlegan/artifacts/api-model-toys/vae`. Root maintains the one
current goal leaderboard; this report does not create a second leaderboard.

Status: implementation and software controls prepared; full scientific runs
await the shared frozen-source commit. No trained-quality claim is made here.
