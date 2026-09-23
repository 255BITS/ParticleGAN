# Develop API quality review

Initial review of `master` (`d162e4c`) plus `particle-finetune/base`
(`510e005`), integrated in PR #44. The intended boundary is a small library of
generic GAN primitives with stable defaults. This is a release review, not a
claim that the integration is ready for release.

## Scope

The integration changes 1,115 files, with 715,448 added lines, but the installed
Python package grows from 9 to 12 modules and 1,580 to 2,088 lines. Top-level
exports grow from 17 to 24. The wheel is 47,425 bytes and contains only
`particlegan` and distribution metadata. PyTorch remains the sole required
dependency. Research repository growth has not become a large runtime dependency
tree; public contracts and default promotion are the main concerns.

## Prioritized findings

1. **High: keyword recipe selection silently changes behavior.**
   `particlegan/recipes.py:get_recipe` no longer accepts positional presets, and
   `get_recipe(name="ddgan")`, `get_recipe(name="mog")`, and
   `get_recipe(name="ae_gan")` now all create an ordinary particle GAN without
   an encoder. Previously those names selected different components. Positional
   calls fail visibly; keyword calls can train the wrong model. Current tests
   explicitly require the new behavior, so passing them does not establish
   compatibility with 0.5.0. Define a migration policy before release: preserve
   old calls temporarily with warnings, or reject ambiguous legacy selectors
   with actionable migration errors. Keep complete saved-recipe restoration.

2. **High: a measured GAN winner becomes defaults for unverified model families.**
   `Recipe` raises generator LR from .0006 to .00425, changes Adam beta2 from
   .999 to .99, changes cap coefficient/target from 1/1 to 6/1.25, and lowers
   particle spread weight from 1 to .05. Removing domain presets also changes
   prior optimizers and model-specific settings. The existing verification
   report explicitly says MoG/DDGAN/application convergence under the common
   defaults is unestablished. Separate API simplification from default
   promotion. Preserve established behavior through a migration period or
   supply matched evidence for each supported family before calling these
   stable defaults. A smoke test establishes execution, not training quality.

3. **Medium: application-specific policy is now a top-level library API.**
   `particlegan/locked_shared.py` contains Music/slider terminology, cover
   posture, a 12-particle cloud, named application deviations and aliases.
   Its factories accept only the exact frozen stamp and duplicate generic
   `GANLoss`/`GradientPenalty` construction. Four associated symbols are
   exported from `particlegan`. Move the stamp and application vocabulary to
   benchmark/application code; migrate callers explicitly if compatibility is
   needed. Generic loss and penalty primitives belong in the public package.

4. **Medium: the batch-dependent critic needs an explicit penalty contract.**
   `BatchDistanceDiscriminator` couples scores across rows, while
   `grad_regularizers._score_scalar` sums the scores before differentiation.
   Consequently the gradient for row i is the derivative of the sum of all
   scores, not just its own score. A deterministic three-point Jacobian check
   with only distance-head weights active confirms these differ. Existing
   tests establish parity with the research implementation, not equivalence
   to a per-sample gradient cap. Document the batch-level definition, including
   finite-difference semantics, or keep this critic experimental until the
   supported combinations have a deliberate contract. Do not silently change
   the formula: that would invalidate the recorded winner.

5. **Medium: the repository test gate is red.**
   `test_particle_native_2d.py` expects fixed-arm EMA action MSE <= .18 but
   produces .248968631. The source branch's installed-default verification
   already reports this failure. Diagnose the benchmark or explicitly separate
   research acceptance gates from deterministic API checks with honest reporting;
   do not weaken its threshold merely to obtain green CI.

6. **Low: trainer scope and documentation need clearer boundaries.**
   `GANTrainer` is optional and reasonably small, but supports only scalar,
   unconditional GANs with the exact `ParticlePrior` type, and owns Adam,
   scheduling, train/eval modes, EMA, RNG restoration and update cadence.
   Keep that scope explicit; avoid extending it into an application framework.
   `CHANGELOG.md` and `locked_shared.py` still describe old recipe defaults as
   unchanged, contradicting the current implementation. Reconcile these claims.

## Recommended boundary

Keep learned priors, losses, gradient penalties, conditioning, diffusion and
encoding operations composable and caller-controlled. Keep one small inspectable
configuration layer with a documented compatibility policy. Treat the optional
trainer and reference discriminators as conveniences with explicit scope;
application stamps, benchmark selection, controllers and domain-specific tuning
belong outside the stable top-level API. Do not delete research evidence merely
because it is outside the runtime API.

Review remaining PRs individually against this boundary. Several are application
experiments or superseded by commits already in the integration; their open
status alone is not a reason to merge them into `develop`.

## Verification

- CPU suite: **686 passed, 5 skipped, 1 failed**, plus 27 passing subtests,
  in 77.23 seconds. Four real-data CUDA tests and one trainer CUDA test skipped.
- Wheel built and installed into an isolated target outside the checkout.
  Two-update GAN quickstart and AE/hard-VAE reconstruction examples passed.
- Direct compatibility probes reproduced all three silent selector changes.
- No seed experiments or new training search. The test suite ran its existing
  numerical gates; the 19-task training leaderboard was not rerun.
- Added `develop` to the existing CI push trigger so branch updates run checks.

Local logs: `/tmp/particlegan-develop-qc/{pytest,wheel,install,smoke}.log`.
Use `tail -F /tmp/particlegan-develop-qc/pytest.log` to follow the suite.

## Existing evidence leaderboard

These are archived development-suite results, not new measurements from this
review. Recipe versions below are not package versions or supported selectors.

| Recipe | Live passes with documented D choices | Reference D profile |
| --- | ---: | ---: |
| v3 | 19/19 | 15/19 |
| v2 | 8/19 | 8/19 |
| v1 | 5/19 | 5/19 |

The installed v3 replay reports EMA success on 7/10 measured tasks. The full
19/19 uses task-specific discriminator choices, resource sizes and some custom
loops; it is not universal default convergence evidence. See the
[verification report](../reports/transfer_suite/single_default_verification/README.md)
and [version comparison](gan-v3.md).

Recommended next work: settle recipe compatibility first, remove application
policy from the public namespace second, then verify defaults across the
supported model families with matched configurations rather than seed sweeps.
