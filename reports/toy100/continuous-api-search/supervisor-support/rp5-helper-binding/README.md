# RP5 public LR-helper binding — scratch proposal

Prepared only; not applied to the active worker or research checkout. The patch adds keyword-only `precision=` to `learning_rate_scales` and `scale_learning_rates`, plus the public root export `ReversiblePrecision`. A continuous recipe requires that controller type and a matching variant. Helpers return its current scales without advancing precision or reading a horizon; a caller NetworkLRTransition is rejected for this mode. Finite recipes retain their existing schedule arithmetic and reject an accidentally supplied continuous controller.

```python
from particlegan import learning_rate_scales, scale_learning_rates

network, prior_scale = learning_rate_scales(
    trainer.completed_steps, trainer.recipe, precision=trainer.precision)
scale_learning_rates(
    trainer.completed_steps, trainer.recipe,
    (trainer.opt_g, trainer.opt_d), trainer.initial_lrs, trainer.prior,
    precision=trainer.precision)
```

Variant matching verifies that controller policy matches the recipe; the helpers cannot infer critic/optimizer ownership. Callers should pass the controller belonging to their update transaction. Calling these helpers alone does not implement RP5's precision observations or secant transaction.

`rp5-helper-binding.patch` changes only recipes.py, the root export and one new contract-test file. GANTrainer, ReversiblePrecision, game_update, KA2 and optimizer arithmetic are byte-identical to immutable RP5. `baseline.json` records source ZIP and original hashes; `results.json` records proposed hashes and the exact test command; `snapshot/` is the isolated runnable package.

Patch integrity: `git apply --check` passes against a separate pristine file snapshot. The patch was not applied there or to the worker. Active package hashes still match the immutable baseline.

Validation: **16 passed** on CPU, including an existing finite NetworkLRTransition resume contract. Tests cover controller scales at ages0–1e9 without state mutation, irrelevant finite schedule fields, open/closed role scaling without compounding, reopening, missing/type/variant errors before LR mutation, conflicting schedules, and old finite closed-form behavior. The tests construct tiny CPU modules/optimizers but run no training forward, backward, optimizer step, GPU, or quality experiment. This is API coverage preparation and makes no qualification claim.

Reproduce from `snapshot/`:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -m pytest -q tests/test_precision_lr_helpers.py tests/test_k3p_trainer.py::test_caller_marked_network_transition_keeps_prior_schedule_and_resumes --junitxml=../cpu-contract-results.xml
```

Before applying elsewhere, compare that package to the recorded immutable source, review the patch, and run its normal relevant integration checks. No worker source or shared manifests were changed here.
