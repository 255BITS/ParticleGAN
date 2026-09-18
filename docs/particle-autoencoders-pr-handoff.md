# Particle autoencoders: PR and release handoff

> Historical experiment/planning handoff. The 0.5.0 release API and final naming
> are documented in [the particle autoencoder guide](particle-autoencoders.md).
> `vae_gan` is the constant-KL default; reconstruction never adds KL.

## User's selected next step

The user says there is enough experimental evidence for a PR covering **particle
AE-GAN, particle VAE-GAN, and AE-DDGAN**. Prepare for compaction now; after
compaction, document the three families, prepare the PR, and bump a release.
This supersedes all older suggestions to run more toy/image experiments,
including the proposed top-four uniform posterior. Do not resume those sweeps
by default. No new experiment, PR, version bump, commit, push, tag, or release
was performed in this preparation turn.

## Workspace and preservation

- Worktree: `/home/martyn/dev/ParticleGAN-mog-autoencoder`.
- Branch: `feature/mog-autoencoder`; HEAD `b06f01f`.
- Main worktree `/home/martyn/dev/ParticleGAN`, branch `master`: tracked source
  remains unchanged. Continue in the feature worktree.
- Prior toy and first CIFAR AE/variation work is committed. Latest DDGAN and
  both genuine-VAE rounds remain uncommitted/untracked, alongside updated
  `.gitignore` and handoff. Do not lose or overwrite them when compacting.
- No experiment jobs remain running or queued. Raw checkpoint/source archives
  and logs under ignored `runs/`; portable results under `reports/`.
- Python: `/home/martyn/dev/ParticleGAN/.venv/bin/python`; two RTX A6000 GPUs.
- AGENTS.md: no seed-only experiments; be token efficient; easy-to-tail logs;
  summarize metrics, leaderboard, explanations and recommendations.
- No delegation requested for this stage. Earlier explicitly requested rename
  subagent completed; it is not an ongoing delegation task.

## Three-family documentation scope

| User-facing family | Definition | Evidence currently available |
|---|---|---|
| Particle AE-GAN | `E(X)->(k,u); z=p[k]+sigma*u; G(z)->X_hat`. Deterministic bounded offset, reconstruction+GAN, no variational objective. | Toy and CIFAR32; numerical conditional-jitter audit on frozen image models. |
| Particle VAE-GAN | An explicit posterior and likelihood plus GAN. Main soft categorical posterior samples k and uses prior-matching local noise; hard variant has constant joint KL. | Toy only. Do not present CIFAR AE results as image VAE validation. |
| AE-DDGAN / DDGAN + particle AE | `E(X)->z_X; G(z_X,X_t,t)->X_hat`. E sees clean X; independent prior latent codes still used for adversarial training and reverse-chain generation. | CIFAR32; matched 10k controls, per-t one-step reconstruction and fixed-context numerical variation. |

Keep AE-DDGAN explicitly deterministic. We have not tested a genuine VAE-DDGAN.
Stochastic categorical AE-GAN without KL is an ablation, not a fourth flagship
variational model. Post-training jitter alone does not make a VAE.

Explain using simple `E(X) -> ...` notation as the user prefers. State which
parts are learned, what remains fixed, and which latent distribution is used
for reconstruction versus unconditional generation. Keep caller-owned networks,
training loops, devices and optimizers consistent with the package's existing
composable design.

### Variational details that must remain precise

- Prior: uniform k over K; `z=p[k]+sigma*eps`, eps standard Gaussian.
- Soft posterior: `E(X)->q(k|X); k~q; z=p[k]+sigma*eps`. Local posterior equals
  prior conditional, so continuous KL is zero; categorical KL is
  `log(K)-H(q(k|X))`. This is generally variable, so omitting it changes ELBO
  optimization. Existing soft VAE uses an unbiased two-draw score estimator.
- Hard posterior: `E(X)->one k; z=p[k]+sigma*eps`. Joint KL over `(k,z)` is
  `log(K)`, constant and omitted from optimization but retained in ELBO metrics.
  Hard routing uses a **biased straight-through query gradient**; a valid bound
  does not imply an unbiased optimizer. This is related to the established
  VQ-VAE constant-KL argument, not a novelty claim.
- Fixed sigma fits this construction naturally. The mathematical cancellation
  requires **matching local posterior and prior**, not static sigma alone.
  A jointly learned shared sigma could also cancel. Unrestricted learned local
  offset/variance generally introduces a nonconstant continuous KL.
- Constant joint KL does not imply zero KL, uniform aggregate usage, calibrated
  diversity, or the same value for KL to the marginal overlapping MoG p(z).
- Likelihood: `p(X|z)=N(G(z),tau^2 I)` in toy. Two-dimensional negative ELBO:
  `MSE/tau^2 + KL + log(2*pi*tau^2)`. GAN/spread are additional losses, not part
  of the ELBO itself. Standard benchmark samples use decoder means G(z); actual
  likelihood samples include observation noise and have separate metrics.
- Hard VAE randomness is inside a single selected particle, with modest measured
  output variation. Do not promise broad semantic diversity or multimodal
  uncertainty over particle identity for this family.

## Evidence and source map

Read these portable reports instead of rerunning experiments:

1. [Latest five-arm toy round](../reports/mog-vae/stability/README.md),
   [leaderboard](../reports/mog-vae/stability/LEADERBOARD.md),
   [protocol](../reports/mog-vae/stability/PROTOCOL.md).
   At 6k: AE97 modes/89.71%HQ/MSE.002341; stochastic noKL96/92.92%/.007220;
   GAN96/75.49%; hard VAE88/85.21%/.003264; soft VAE86/82.50%/.003884.
   Hard pairRMS.01066, soft.02946; both retain input mode on all65,536 draws.
   Hard KL5.9915, hard/soft negativeELBO4.443/3.916. AE is best reconstruction;
   GAN best SW1.1685; all output mode widths too narrow. Stability remains mixed.
2. [Original 12-arm VAE scout](../reports/mog-vae/README.md),
   [leaderboard](../reports/mog-vae/LEADERBOARD.md). Includes local posterior and
   no-GAN controls and matched-count late-regression audit. Do not mix settings
   or cherry-pick earlier checkpoints into final tables.
3. [Matched direct/DDGAN CIFAR comparison](../reports/cifar-particle-ddgan/README.md),
   [leaderboard](../reports/cifar-particle-ddgan/LEADERBOARD.md).
   10k updates/FID50k: direct GAN19.483, particle AE-GAN20.054,
   DDGAN+particle AE43.233, DDGAN49.475. AE improves DDGAN FID12.6% at16.9%
   extra training cost; direct AE reconstructionMSE.078996. These use half the
   preceding CIFAR scout learning rates. One matched trajectory, not a universal
   superiority claim. Historical conditional DDGAN results are not matched.
   DD reconstruction has noisy-image side information, not latent-only or full
   encoded reverse-chain reconstruction. Held-fixed context audits confirm code
   use; post-training noise injection is not a learned variational posterior.
4. [Earlier CIFAR AE round](../reports/cifar-particle-ae/README.md) and its
   variation report: older higher-LR results, including late regression; preserve
   as history, not headline evidence of robust FID superiority.
5. [Full chronological handoff](mog-autoencoder-handoff.md) for older toy
   routing/oracle experiments. Oracle routing is not the intended release path.

Implementation locations:

- Direct AE: `lib/image_particle_autoencoder.py`,
  `experiments/train_cifar_particle_ae.py`; toy `train_mog_autoencoder.py`.
- DD AE: `lib/image_particle_ddgan.py`,
  `experiments/train_cifar_particle_ddgan.py`; analyze/variation scripts alongside.
- Genuine VAE: `experiments/train_mog_vae.py`,
  `experiments/train_mog_vae_stability.py`, matching analyzers and tests.
  Second trainer intentionally preserves the first trainer's source fingerprint.
- Configs: `configs/cifar_particle_ae/`, `configs/cifar_particle_ddgan/`,
  `configs/mog_vae/`; existing queue `follow_grid.py` / `run_grid.py` unchanged.
  Explicit `--workers_per_gpu 1` is essential; default is five.

## Work remaining for a release-ready PR

1. Inspect branch changes against current target branch and settle the public
   API scope. Currently new encoders/adapters live in `lib/` and `experiments/`.
   `pyproject.toml` includes **only particlegan and particlegan.* in the wheel**.
   Merely documenting checkout-only imports will not expose release features.
   Extract the minimal reusable routing/posterior/loss primitives if presenting
   these as package features, with caller-provided encoders/decoders and clean
   deterministic/variational semantics. Do not publish toy-specific spatial
   skips or image architectures as generic APIs by accident. Update public
   exports and any justified recipes/examples; keep core dependency torch-only.
2. Write coherent user documentation for all three families, runnable simple
   examples, objective/gradient caveats, fixed-sigma semantics, and evidence
   links. Update README, docs/api.md, docs/reproducing.md as appropriate.
   Condense exploratory reports in the PR description; preserve raw provenance.
3. Add meaningful tests for any extracted public primitives and installed-wheel
   imports/use. Run the full repository suite, relevant opt-in CUDA integration
   checks, build the wheel/sdist, Twine strict checks, and wheel smoke tests
   outside the checkout. Existing experiment checks are not release validation.
4. Review files to stage: include implementation/configs/tests/portable reports,
   exclude ignored raw runs/checkpoints/data. Preserve prior run source archives
   if refactoring invalidates current-source completion matching; do not rerun
   old successful experiments merely because library fingerprints changed.
5. Prepare commits and a focused PR describing actual final API/behavior and
   measured limits. User has selected PR/release work; no further experiment
   approval question is needed to begin this authorized preparation.
6. Set release version after checking current target branch/tags/registry state
   during release execution. Local pyproject currently says0.4.0; 0.5.0 is a
   plausible minor feature bump, not a version already selected or published.
   Update pyproject, CHANGELOG, README changelog/install text and stale release
   instructions together.

## Existing validation and release process

- Latest VAE work:24 tests plus13 subtests; five pilots and five full runs passed;
  five certificates,15checkpoint hashes and saved arrays verified.
- First VAE scout:12 full runs passed;36checkpoint hashes verified.
- CIFAR direct/DDGAN round:28 tests plus13 subtests at that stage; all four
  pilots/full runs and numerical variation audits passed.
- These are targeted suites and experimental checks, **not a newly completed
  full package/release test run**. No training remains queued.
- Latest dry-runs: stability5done/0to run; originalVAE12done/0to run.
- Main tracked source unchanged; feature `git diff --check` passes.
- Read [release guide](releasing.md), `.github/workflows/tests.yml`, and
  `.github/workflows/release.yml` when implementing the release. The guide has
  historical statements about first publication and prepared0.4.0; verify current
  external state rather than treating them as current facts.
- CI tests Python3.10–3.12 with CPU torch, full pytest, and wheel smoke tests
  outside the checkout; then builds distributions and runs `twine check --strict`.
  CUDA checks use `RUN_CUDA_IMAGE_TESTS=1` locally as documented.
- Publishing a GitHub release with tag matching `v{pyproject version}` triggers
  tests/build and PyPI trusted publishing through environment `pypi`. A draft
  release or tag alone does not publish. No release action occurred this turn.
