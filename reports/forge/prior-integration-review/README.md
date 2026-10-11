# Combined prior feature integration review

**Recommendation: safe after the explicit integration fixes in this branch.**
The five PRs can coexist in one build and remain opt-in. Their conflicting
edits should not be merged mechanically: the initial union broke old positive
Coulomb checkpoints and checkpoint creation on non-sampled hosts. Both cases
are fixed here. No defaults are adopted and no scientific qualification is
transferred to the combined source.

The local branch is `review/prior-feature-integration`, based on
`origin/develop` at `0337bbbc2300fd351e77e9a222b92401137fd421`. It incorporates:

| PR | Feature | Reviewed published head |
|---|---|---|
| [387](https://github.com/255BITS/ParticleGAN/pull/387) | Proximal spring | `59d46d57bbc78acdeb486fba8c796256ab8015b5` |
| [388](https://github.com/255BITS/ParticleGAN/pull/388) | Coulomb | `5719cf00b41043602cffa2ca550273ce9b7c3895` |
| [389](https://github.com/255BITS/ParticleGAN/pull/389) | SIGReg | `40acfafa8eacfe7a0a8168eefd7797f28fbb4635` |
| [390](https://github.com/255BITS/ParticleGAN/pull/390) | Moment tether | `67badefcab620f32a8982dd9cd144f04ddac2c5d` |
| [391](https://github.com/255BITS/ParticleGAN/pull/391) | Initial MMD | `dd677a69bf10887830a3af65582ecdfac934b22f` |

## Conflicts and fixes

| Area | Combined behavior / fix |
|---|---|
| Recipe fields, exports and tunable registry | Retain every new field and all selectors: `vicreg`, `none`, `coulomb`, `sigreg`, `moment_tether`, `initial_mmd`. Default additions remain omitted from `Recipe.to_dict()`. |
| Factory and host calls | Preserve both public aliases, `prior=` and `reference=`. If both are supplied they must identify the same initialized prior. All supported host paths pass their own initialized prior. |
| Trainer dispatch | Call the selected regularizer once under the existing row law; apply `prior_reg` once. Preserve the historical unweighted VICReg diagnostic at zero. |
| Checkpoint state | Only an actual active SIGReg module adds the auxiliary private projection RNG packet. Coulomb, moment tether and MMD retain prior-owned reference state; proximal retains optimizer-owned initial positions and settings. |
| Direct-coordinate hosts | A positive global coefficient with no regularizer module emits no auxiliary packet. Loading the corresponding old checkpoint remains valid. |
| Applicability | `prior_reg_eps` remains active for moment tether. Coulomb and SIGReg knobs are inactive on frozen or non-sampled recipe-owned hosts; receipts retain their declared reference values with `not_applicable`. Legacy task-owned fields are unchanged. |
| Word adapter | Retain the feature branches' missing `recipe_owned_prior` import fix, which repairs a pre-existing legacy-word `NameError`. |
| Current task source metadata | Refresh only the 11 affected current/dependency task source bindings, including word acquisition. Record a combined source revision with the previous revision intact. Full task contracts excluding source metadata match develop exactly. |
| Compiled memory | Refresh summaries only. Preserve the existing qualification snapshots, canonical selection and leaderboard. Keep all five source-bound search publications. |

Two pre-fix exceptions were reproduced from the initial union at
`f60609e9fa0b9a4e23cb636ec51ce683c4b45bd4`:

```python
# An old positive Coulomb checkpoint has no SIGReg auxiliary packet.
recipe = Recipe(prior_regularizer="coulomb", prior_reg=.1)
module = recipe.make_prior_regularizer(prior=initialized_prior)
recipe.restore_prior_regularizer(module, None)
# ValueError: active prior regularizer requires checkpoint state

# Non-sampled public hosts do not construct a latent regularizer module.
Recipe(prior_reg=.1).prior_regularizer_checkpoint(None)
# AttributeError: 'NoneType' object has no attribute 'state_dict'
```

The fix restricts auxiliary checkpoint ownership to active SIGReg with a real
module. It neither fabricates old checkpoint fields nor relaxes SIGReg's active
state validation.

## Validation and limits

**939 distinct software tests pass**, with 991 passing executions including
focused strengthened rechecks. The compact [receipt](receipt.json) binds source
files, frozen fixtures and external JUnit/log identities. There are 47 combined
compatibility cases in [test_prior_feature_integration.py](../../../tests/test_prior_feature_integration.py):

- Exact frozen-develop initial state, two updates, checkpoint packets, losses
  and consumed RNG for BCAP default, positive VICReg, frozen and `none`, plus
  default and positive VICReg GAN. All eight existing preset dictionaries are
  identical to frozen develop.
- Load frozen-develop checkpoints and each original positive feature PR's
  checkpoint into the combined build; the next update and full resulting state
  match exactly.
- Every zero-weight selector preserves model/optimizer state, losses used for
  training, consumed streams and disabled auxiliary state. An explicitly changed
  selector can change an unused unweighted regularizer diagnostic.
- Each positive regularizer together with the proximal spring activates on CPU
  and CUDA, resumes exactly, and changes only the actual unique G-sampled prior
  rows. The initial anchor remains the public initialized table.
- Frozen positive configurations do no prior updates; direct hosts need no
  auxiliary packet. Coulomb applicability is checked on frozen sampled and
  non-sampled hosts.
- Existing feature/core/search tests (345) and host/API tests (547) cover public
  vector, image, word, Gaussian and behavioral paths. The final boundary recheck
  passes 24 existing cases plus the two new cases.

These are bounded public-API **software compatibility fixtures**, using seed 0,
fixed data and two small CPU reference updates, plus CPU/CUDA activation checks.
They are not archived full-training gates. There were no new scientific
experiments, queue admissions, seed sweeps, or rescoring of archived evidence.
The branch does not claim universal bitwise equivalence for every architecture
or GPU. Existing tests and the explicit frozen-source cases support the merge
determination.

The regularizer selector chooses one family at a time. A positive proximal
spring composes with any selected family and runs after the prior optimizer and
direction blend; it may change the final protected derivative. Simultaneous
multiple regularizer families are not exposed by this API. Native reference
buffers, word host's own certified prefix and Gaussian's fixed prefix remain
owned by their original hosts and receipts. Direction-blend counter semantics
are unchanged.

All five mechanism result/media/configuration-search publications are byte
identical to their respective reviewed heads. Their original executed sources,
recipes, initializers, streams, budgets and verdicts are retained. Combined
source validation creates no new pass, incumbent or qualification credit.

## Reproduction and proposed merge

Use [reproduce.py](reproduce.py) to extract the exact six frozen sources, export
the bounded fixtures with [probe.py](probe.py), and run the same selected suites.
It writes checkpoints, stdout and JUnit only to the requested external archive:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  reports/forge/prior-integration-review/reproduce.py \
  --archive /mnt/ml7tb/ParticleGAN-forge/prior-integration-review-reproduction
python -m experiments.forge validate
python -m experiments.forge compile --summaries-only
python -m experiments.forge compile --check
```

The original archive is
`/mnt/ml7tb/ParticleGAN-forge/prior-integration-review-20261011`; logs are easy
to inspect with `tail -f <archive>/logs/<suite>.log`. The initial final-boundary
fixture had a keyword-call typo; its failed software attempt is preserved as
`combined-boundary-test-signature-error.*`, followed by the passing correction.

Prefer one reviewed combined integration PR containing this branch, so the
checkpoint fixes ship with the conflicting feature changes. The tested local
integration order is **387 → 388 → 389 → 390 → 391**, followed by the compatibility
fixes and summaries-only refresh. If individual PRs are merged instead, port the
same resolutions and checkpoint fixes before using the combined options, then
rerun this compatibility suite on that exact result. This worktree is isolated;
the user checkout and feature execution worktrees were not edited. Nothing was
pushed or merged remotely by this review.

## Merge follow-up against the promoted default

After the initial review, `develop` advanced to
`3f9846cb7cb276dfbd0802b6eb4c100a8b75ac41`, promoting the BCAP trainer through
`DEFAULT_RECIPE`. The integration preserves that promotion and resolves the
single export conflict by retaining both `DEFAULT_RECIPE` and the new prior
exports. No additional prior option is enabled by default.

**1,202 distinct software checks pass** across 42 affected/core/host test files,
including all 50 combined cases. Historical fixtures now restore their saved
resolved `Recipe` independently of the mutable public selector; historical
model-family preset checks explicitly pin KA2. Three additional cases compare
the promoted model presets, two updates, checkpoint continuation and consumed
RNG exactly against the frozen new develop source. The original 939-test
[receipt](receipt.json) remains bound to review commit
`555a42c5a0bc37436a3885d75d20a503ff76995f`; it is not rewritten as new evidence.
The separate [merge receipt](merge-receipt.json) records the follow-up sources,
selected tests, archive identities and corrected fixture/environment attempt.

For the updated source, add
`--promoted-source 3f9846cb7cb276dfbd0802b6eb4c100a8b75ac41` to the reproduction
command above. The merge receipt also lists the default-promotion test files
included in the broader follow-up. Logs remain outside Git at
`/mnt/ml7tb/ParticleGAN-forge/prior-develop-merge-20261011/logs`.
