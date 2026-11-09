# Reproducing and checking experiments

Use Python 3.11 or newer, as required by [the package metadata](../pyproject.toml).
The CPU test workflow checks the full non-CUDA suite in four shards on each of
Python 3.11 and 3.12, and verifies a complete nonoverlapping collection for each
interpreter. A separate installed-wheel job checks the base public API and exact
checkpoint continuation on Python 3.11 with Torch 2.6.0 and NumPy 1.26.0. That job
installs no optional extras; it does not claim minimum-version support for a
Torchvision/experiment dependency combination it has not tested.
Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[experiments,dev]'
python -m pip check
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

The `experiments` extra includes the dependencies needed by the experiment scripts,
including PyYAML and POT (`import ot`) for exact transport evaluation. A CUDA
installation of PyTorch is useful for full studies. For CPU-only work, install
PyTorch from its CPU wheel index before installing the project:

```bash
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[experiments,dev]'
```

The tests exercise numerical gradient correctness, evaluation isolation, sparse
gradient penalties, particle identity bookkeeping, experiment failure/resume
behavior, and small training runs. They do not reproduce the reported full
studies or establish convergence for a new dataset. The exact-transport test
uses a distribution with a known answer so a missing solver is caught before
an expensive training run reaches its final evaluation.

## Record the environment

Run `python scripts/check_lint.py` for the reviewed Ruff baseline used by CI.
Install the `dev` extra to get its pinned Ruff version. The baseline retains
existing fixture imports and exports; new findings and resolved exemptions both
fail the check. Undefined names (`F821`) are strict: even `--update-baseline`
refuses them. Three reviewed static control-flow false positives in frozen
reproduction sources have exact path/message/source-SHA exceptions in the lint
driver; changing their bytes or introducing another undefined name fails the
strict gate. The retained `F811` entries were reviewed as imported pytest fixture
names shadowed by injected fixture arguments, rather than active implementation
redefinitions. Their existing counts stay explicit; new redefinitions still fail.
After reviewing a change, refresh the compact baseline with
`python scripts/check_lint.py --update-baseline`. Historical source snapshots
under `reports/` are outside this lint scope.

The manifest defines supported dependency ranges, not a bit-for-bit numerical
environment. Save the resolved environment and commit with every full study:

```bash
mkdir -p results/reproduction
python -m pip freeze > results/reproduction/requirements.txt
git rev-parse HEAD > results/reproduction/commit.txt
python -c 'import torch; print(torch.__version__); print(torch.version.cuda)' > results/reproduction/torch.txt
```

Also retain the expanded run configurations, hardware description, summaries,
and samples. Matching a seed is insufficient for exact reproduction across
different PyTorch/CUDA versions and hardware. Historical tables were produced
before the evaluation-randomness fix and should not be expected to match new
trajectories bit for bit.

Software maintenance can change a source digest while leaving the original
experiment results intact. Check retained publication integrity with
`python ops/refresh_forge_readouts.py --check --reports-only`; use strict
`--check` separately to inspect current-source evidence gaps. Preserve original
receipt identities and qualification results when that scope reports
`NEEDS_REFRESH`. A source change alone does not justify rerunning an unchanged
study or relabelling its results as evidence for the new source.

## Historical controlled prior comparison

The earlier VICReg comparison used the same generator, discriminator, loss,
penalty, learning rates, and training budget for three priors:

* `particles`: a learned finite table, with its prior optimizer and VICReg term;
* `frozen_gaussian`: a fixed table initialized from a Gaussian;
* `fresh_gaussian`: newly sampled Gaussian noise for ordinary draws.

The legacy configuration name `gaussian` continues to mean the frozen table.
It must not be interpreted as fresh continuous Gaussian sampling. The learned
table's optimizer and regularizer are part of the learned-prior intervention;
further ablations are needed to distinguish their individual effects.

The completed comparison and its protocol are in
[the prior-comparison report](../reports/prior-comparison/README.md); its
driver is in the repository history. The newer
[2026-10-25 matched prior ablation](../reports/prior-ablation-2026-10-25/README.md)
uses the BISPF recipe and its SBNP regularizer under its own declared protocol.
Keep the two studies' recipes, initializers, budgets and sampling laws separate.
The older GIFs retain their original recipes and are not evidence for either
matched comparison unless its report binds them explicitly.

## Interpreting shape and diversity

For a deterministic generator over M fixed particles, ordinary sampling can
emit at most M distinct outputs. Drawing 100,000 samples from 20,000 particles
does not create 100,000 independent locations in output space. Mode coverage,
sample plausibility, continuous diversity, and distributional fidelity are
separate properties.

The robust core ratio estimates a radial median using an isotropic-Gaussian
conversion. A two-point cloud in each mode can pass that check while having
zero variance along one axis. The covariance eigenvalue ratios now expose
this failure: each eigenvalue is divided by the true variance, and the minimum
and maximum are averaged over sufficiently populated modes. These are
tail-sensitive population moments; read them alongside core spread, tail
mass, and the number of audited modes. Even correct covariance does not prove
Gaussian shape, so inspect radial and angular structure for stronger claims.

Generated-sample NLL under the target rewards concentrating at mode centers.
It is a plausibility score, not a calibration test. Likewise, a confidence
interval containing zero is an inconclusive difference test, not proof of
equivalence. Establish an equivalence margin and adequate replication before
claiming two recipes perform the same.

For future scaling studies, vary the particle count and separately test noise
around particles. Adding noise changes the model distribution and requires a
fresh comparison; the finite-support limitation is not silently removed here.

## Particle autoencoder studies (0.5.0)

The [API guide](particle-autoencoders.md) maps public primitives to the tested
encodings. Original trainers remain intact to preserve archived source hashes.
Public API additions change the runner's source fingerprint: use a fresh output
directory for a new run, rather than replacing successful historical artifacts.

- [Matched CIFAR direct/DDGAN round](../reports/cifar-particle-ddgan/README.md):
  configs under `configs/cifar_particle_ddgan/`.
- [Toy VAE scout](../reports/mog-vae/README.md) and
  [lower-LR/constant-KL comparison](../reports/mog-vae/stability/README.md):
  configs under `configs/mog_vae/`.
- [Earlier AE studies](../reports/mog-autoencoder/README.md) and
  [CIFAR AE baseline](../reports/cifar-particle-ae/README.md).

Use the [queue runner](experiment-runner.md) with one worker per GPU for image
jobs and the reports' commands for tailing combined logs. Reports include
configs, completion certificates, metrics and numerical audits; raw checkpoints
and source archives stay under ignored `runs/`. No new training was required
for the API release. A recipe alone does not specify the study's architecture.
