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

1. **Resolved: named component selection and recipe ownership.**
   The integration initially removed positional names and silently interpreted
   `get_recipe(name="ae_gan")` as ordinary GAN metadata. Model-family names
   (`gan`, `mog`, `ddgan`, `ddgan_mog`, `ae_gan`, `vae_gan`, `ae_ddgan`) now
   select component configurations. Explicit keyword overrides remain supported.
   `Recipe.make_trainer` is removed; callers explicitly import and construct
   `GANTrainer(recipe, G, D)`. UCD, encoding and diffusion compose through
   caller-owned loops. The recipe owns settings and small object factories.

2. **Accepted defaults; remaining evidence gap across model families.**
   The new shared hyperparameters are retained by design: G/D LR .00425,
   particle LR .0085, Adam (0,.99), cap coefficient/target 6/1.25 and particle
   spread weight .05. No legacy numerical preset selectors or compatibility
   tables are added. Named families select components, dimensions and routing,
   while inheriting current optimizer/loss/schedule settings. GAN remains a
   discrete particle prior; a MoG default is not promoted in this change.
   The existing verification report does not establish MoG/DDGAN/AE/VAE
   convergence under these common settings. Evaluate those families before
   describing the defaults as their individually vetted best settings; passing
   component and execution checks alone does not establish that claim.

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

5. **Tracked research failure, with independently tested gate logic.**
   The native-2D toy expects fixed-arm EMA action MSE <= .18 but produces
   .248968631. Replaying exact original source `c7e8a73` gives the same result;
   the integration's GitHub matrix reproduces it on Python 3.10, 3.11 and 3.12.
   The historical convergence assertion is now a strict expected failure,
   separate from protocol, finite-value, threshold-boundary and CLI tests.
   Thresholds are unchanged and the benchmark CLI still fails this numerical
   gate. This repairs test reporting, not the unmet convergence claim.

6. **Resolved: documentation of the trainer boundary.**
   `GANTrainer` remains a separately constructed convenience for scalar,
   unconditional GANs with the exact `ParticlePrior` type. It owns Adam,
   scheduling, train/eval modes, EMA, RNG restoration and update cadence.
   Recipes no longer construct it. Named AE/VAE/DDGAN and UCD configurations
   are explicitly rejected by the helper and documented with caller-owned
   component composition. Stale claims about unchanged numeric defaults were
   corrected in the changelog and stamp documentation.

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

## Current validation

- CPU suite: **706 passed, 5 skipped, 1 strict expected failure**, plus 27
  passing subtests, in 70.27 seconds. The expected failure is the unmet
  native-2D convergence target discussed above.
- Installed wheel: explicit trainer quickstart, named AE/VAE reconstruction
  examples and model-family selection pass outside the checkout.
- Named recipe tests verify shared current hyperparameters, explicit overrides,
  serialization and the absence of training lifecycle methods on `Recipe`.
- The plain GAN example rejects encoder configurations before training.
- Refreshed README animation: one 7,000-update run, 99/100 modes and 75.08% HQ
  on 20,000 EMA samples. See the [image report](../reports/readme-100gaussians/README.md)
  for metrics, limitations and reproduction. This does not establish full
  convergence or live-model performance.
- Final suite log: `/tmp/particlegan-develop-qc/pytest-final.log`; image log:
  `/tmp/particlegan-develop-qc/readme-image.log`.

## Integration baseline verification

These results precede the recipe repair and test fix.

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

Recommended next work: review application policy in the public namespace and
verify current defaults across the supported model families with matched
configurations rather than seed sweeps. Keep the archived numerical evidence
distinct from validation of the current API.
