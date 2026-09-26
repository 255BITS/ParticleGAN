# Frozen vector preparation support

This is a **prepare-only scaffold**, not a training runner. During implementation the supervisor stopped further C6-specific runtime integration after C6's stationary run showed departures. No task has been trained or scored by this support directory. There is intentionally no execution flag.

`prepare.py` accepts an explicit candidate checkout, verifies every package module against the audited immutable C6 source ZIP, verifies pinned host sources, and emits six complete task declarations plus a source archive and preparation receipt. It imports neither PyTorch nor the candidate package and constructs no models. `host-lock.json` pins the frozen plan **and** the four promoted discriminator cards, host samplers/scorers, initialization code, observation/noise conventions and image template. The archive includes candidate source, harness, the pinned fixture files and conservatively resolved local Python dependencies, including imports in unused historical host functions. These historical files are provenance; this scaffold never constructs a LegacyRecipe or invokes a legacy training loop.

```bash
/tmp/pr38-default-env/bin/python prepare.py \
  --repo /path/to/unchanged/C6/repo \
  --output /path/to/new/preparation-directory
```

The default candidate declaration and ZIP are `experiments/constant_game/C6-single/{declaration.json,source.zip}` inside `--repo`. Alternate locations can be passed with `--candidate-declaration` and `--candidate-zip`; their content must still match the audited C6 archive. A future candidate needs an explicitly reviewed candidate verifier change, while the frozen host lock/declarations remain reusable.

The emitted recipe is the full exact C6 recipe with only `num_particles=256`, `z_dim=4`, and `batch_size=128` adapted. Evaluator budgets stay outside the learner: five tasks use 1200 accepted updates, spiral uses 1600, and all retain 24 observations and five final passing observations. The unrelated historical learner fields in `original_job`/`frozen_host_spec` are provenance only; they must not override the emitted `recipe`.

The model contract preserves the explicit prior first (`init_std=.5`, its own seed0), then G and D. Data, latent and penalty streams use seeds0/1/2. Four discriminators come from `leading_profile.json`, including the batch-distance head for unequal mass; plan-only construction is incorrect. Generator dimensions and target distributions remain frozen.

Vector observation latent sampling uses seed990; scoring target/projections use991/992. Paired output noise uses **402+1901=2303, fixed across observation steps**, matching `toy100_compatibility.py::run_vector`. The validated image template uses402+step+1901 instead; copying that image convention would change the vector measurement. Live and EMA measurements receive identical evaluation draws and must preserve all learner and caller data-stream state. Noise magnitudes come exclusively from C6's fixed360/720 startup; host budget schedules and external noise wrappers must not be installed.

## Unresolved runtime semantics

The original vector host is written for CPU-default model initialization and explicit CPU generators. Its published CUDA device policy instead changes default tensor construction and device-less generators, including initialization and data/latent draws. C6's ring/image template initializes networks on CPU before moving them to CUDA. These are different fixture contracts, despite using identical seed numbers. A later runtime adapter must explicitly select and pin the intended historical backend, initialization order/backend, training stream backend and evaluation scorer backend. No silent conversion is supplied here.

Current `GANTrainer` supports these six unconditional one-D/one-G hosts in principle; no confirmed learner API gap was found. Its stream validation requires latent generators on the model device, so retaining exact CPU latent draws while training on CUDA cannot simply be achieved by passing the original CPU generator. A claim of GPU parity needs the reviewed backend contract and actual validation. The four discriminator cards and all target/scoring settings are already resolved.

Before adding a run adapter, construct `GANTrainer(..., serial_backward=True)` and preserve the full public transaction through its `step()` method; optimizer factories alone omit the secant update. A fresh generator-real batch must be supplied once per accepted step from the existing data stream. Snapshot actual runtime and emit the rows described by `receipt-contract.json`. All outputs here remain **NOT_RUN**; C5 scores, C6 ring results or preparation success cannot fill a vector quality gate.

Validation performed here is limited to Python syntax, immutable-source hashes, dependency snapshot construction and six-task fixture declarations. It does not establish model initialization parity, CUDA behavior, checkpoint continuation or task quality.
