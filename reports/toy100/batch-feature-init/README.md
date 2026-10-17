# Batch-feature initialization: research and package integration

> **API update:** the recipe no longer initializes weights and `initialize_` is gone; this
> construction is now the explicit `particlegan.init.deterministic_orthogonal_(module, seed=k)`
> ([API](../../../docs/api.md#initialization)), and the research registry lives in `benchmarks/init_research`.

The initialization-only search selected **QR weights + patterned biases + R2
particles + zero explicit batch-distance readout coefficients**. The frozen
research run passed all **22/22 fixed benchmark gates**, plus long hold.
Shifted-target recovery remains **FAIL** (pre-shift STAY passes 120/120).
No seed, optimizer, schedule, architecture, sampling, budget, or threshold sweep
was performed. The original QR control passed 21/22; zeroing the entire critic
head regressed to 19/22. See [the leaderboard](batch-feature-leaderboard.md),
[mechanism and results](BATCH_FEATURE_REPORT.md), and
[math/architecture guide](../../../docs/initialization.md).

## Public entry points

`get_recipe()` now defaults to `initialization="batch_feature_zero"`.
`GANTrainer` and `recipe.make_optimizers` use the direct, RNG-neutral
`particlegan.initialize_(network, key=0)` API. Standard layer declarations
replace the historical hook's captured declarations; keys are explicit
(G=0, D=1, E=2) instead of depending on process-wide optimizer count.
`initialization=None` preserves supplied weights. Recipe-created learnable
priors get R2 initialization before MoG calibration; supplied priors are kept.

The separately registered `--init batch_feature_zero` retains the historical
Adam hook for exact research replay. It takes precedence over recipe defaults.
The older registry name `qr_pb_pq` refers to a different implementation and
has not been silently remapped.

The public API and replay hook agree tensor-for-tensor on a standard G/D
witness. Extensions to unknown/custom parameter declarations, new parameter
orders, current public trainers, transformers, or LoRA need separate training
qualification. Transformer/LoRA coverage here consists of construction and
algebra checks, not measured training convergence. The mathematical guarantee
is zero initial contribution from the explicit batch branch to scores/input
derivatives, not convergence of the entire sampled Adam game.

## Package replay and integration validation

The package hook was rerun on all 22 tasks plus both stress checks. Result:
**22/22 + long hold PASS; shifted-target recovery FAIL**, exactly as before.
[The parity receipt](package-parity-audit.json) confirms all 19 small-task
initial tensors and final learned/optimizer/RNG states are identical to the
qualified candidate. All three native final/holdout arrays and both recorded
stress trajectories are identical; the stress drivers do not save final
checkpoints. Both source manifests are unchanged (1,630/1,629 files).

The public direct API has tests for repeatability, unchanged RNG streams,
standard-network tensor parity with the research hook, matrix Gram/RMS,
attention/embedding and frozen LoRA construction, neutral batch readout,
EMA initialization, repeated optimizer construction, and legacy checkpoint
continuation. The current public trainer also completed a two-step A6000
smoke run; this is an integration check, not a 22/22 qualification of that
trainer. Installed-wheel AdamW, AE/VAE, and GAN examples pass. The Lunar smoke
pipeline exports its GIFs/page; its smoke budget does not establish a flight
success gate. Full CPU suite status is recorded in the PR checks.

## Evidence retained here

- `batch-feature-collected.json`, `batch-feature-audit.json`: the original
  30-run follow-up (six mechanism screens, 22 tasks, two stress checks).
- `batch-force-diagnostic.json`, `BATCH_FEATURE_MATH.md`: the contracting-force
  diagnostic on unequal mass and its explicit assumptions.
- `verify_initialization_math.py`, `initialization-math-checks.json`: portable
  checks of QR Gram/RMS identities and the convolution/residual/LoRA equations.
- `audit_package_parity.py`: compares all 19 initial/final small-task states,
  three native final/holdout sample arrays, recorded stress trajectories,
  audited RNG digests, and both frozen source manifests.

The original checkpoint/runtime roots remain under
`/ml2/hypergan/pr194-init-search-20260926/batch-feature-full-suite`; the package
replay is under `package-init-parity-suite` in the same directory. Only the
initializer and its numerical helper were added to the frozen runtime. A first
attempt to use the entire current package aborted before training because the
old drivers import removed legacy APIs; `package-full-suite` preserves those
logs, and those aborted jobs are excluded from all training scores.

To repeat the compact audits with the retained artifacts:

```bash
python reports/toy100/batch-feature-init/verify_initialization_math.py
python reports/toy100/batch-feature-init/audit_package_parity.py \
  /path/to/package-init-parity-suite /path/to/batch-feature-full-suite \
  /tmp/package-parity-audit.json
```

The four historical K3P driver files were restored byte-for-byte to their
recorded source hashes. New CLI activation uses the registry launcher, so the
source provenance of previous qualification results remains intact. The benchmark-only
`LegacyRecipe` keeps random initialization by default and omits that neutral
field from historical receipts; it does not inherit the new public default.
