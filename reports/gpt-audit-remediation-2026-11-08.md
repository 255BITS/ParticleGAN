# Scientific-contract audit remediation — 2026-11-08

This follows the [source-pinned audit](gpt-audit-2026-11-08.md). The governing rule is now explicit in [AGENTS.md](../AGENTS.md): **a faithful numerical FAIL is better evidence than a PASS obtained by changing the formula.** The work repairs experiment implementation, evidence attribution and software checks. It does not claim that previously failing distributions now converge.

The [technique inventory](forge/technique-inventory.md) remains the single current leaderboard. Its scientific and operational outcomes must be read separately: a numerical FAIL, an unsupported recipe, an execution error and an unmeasured task are different facts.

## Changes

| Audit finding | Repair | PR |
| --- | --- | --- |
| A1: neural-imputer evaluator bypass | The neural question calls live `G_i`; posterior resampling is forbidden in that evaluator. The actual predictor and clean/live sampling law are recorded. | [#532](https://github.com/255BITS/ParticleGAN/pull/532) |
| A2: task-specific training repairs | Removed empirical grid cluster fitting/particle placement, generator resets, MisGAN subspace weight/prior reconstruction and the undeclared imputer diversity loss. Optimizer updates are no longer followed by these additional solvers. | [#529](https://github.com/255BITS/ParticleGAN/pull/529), [#532](https://github.com/255BITS/ParticleGAN/pull/532) |
| A3: recipe and sampling substitutions | Retired the alternate formulaic coverage executor and its loose result harvesting. Unsupported intact mechanisms refuse execution. Fixed active providers' discovered recipe/sampling mismatches instead of choosing replacements that pass. | [#530](https://github.com/255BITS/ParticleGAN/pull/530), [#529](https://github.com/255BITS/ParticleGAN/pull/529), [#532](https://github.com/255BITS/ParticleGAN/pull/532) |
| A4: incomplete continuation | Versioned complete checkpoints for supported owners, including RNGs, actual EMA histories, critic/penalty histories and regularizer references. Caller-owned loops lacking complete restoration explicitly refuse continuation. Fresh-object continuation is checked; incomplete historical states require their original source. Removed auxiliary solvers no longer create uncheckpointed caches. | [#529](https://github.com/255BITS/ParticleGAN/pull/529), [#532](https://github.com/255BITS/ParticleGAN/pull/532) |
| A5: import-dependent architecture | Removed the shared model monkeypatch. Original unnormalized, unscaled residual, and explicitly scaled/normalized task architectures have distinct factories and metadata. | [#529](https://github.com/255BITS/ParticleGAN/pull/529) |
| A6: status and evidence conflation | Removed the score overlay that replaced exact selected-cohort evidence with diagnostic coverage and imputed numerical FAIL. Source-bound corrections preserve affected observations while excluding them from the formula-faithful comparison. | [#531](https://github.com/255BITS/ParticleGAN/pull/531) |
| A7: unsupported scientific ranking | Retain provisional calibration, unequal tuning budgets and missing independent confirmation explicitly. Current-source measurement checks cannot borrow family/task-only diagnostic results. No accepted calibration or confirmation result was manufactured. | [#531](https://github.com/255BITS/ParticleGAN/pull/531), [AGENTS.md](../AGENTS.md) |
| A8: undeclared surrogate update | The bounded tanh head uses the exact automatic derivative of its forward map, without clipping the upstream gradient or injecting a restoring force. Declared BISPF shape/call-count rules remain part of that recipe; this repair does not silently redefine BISPF. | [#529](https://github.com/255BITS/ParticleGAN/pull/529) |
| Engineering followups | Added MIT distribution metadata, complete Python 3.11/3.12 test sharding and an actual minimum-dependency installed-wheel continuation check. Undefined-name findings cannot be accepted by updating the lint baseline. | [#533](https://github.com/255BITS/ParticleGAN/pull/533) |
| Prospective measurement provenance | Future supported runs emit actual execution witnesses and certified per-attempt coverage exports. Missing or incompatible recipe, initialization, runtime, sampling or serving observations remain gaps; numerical grades are unchanged. | [#534](https://github.com/255BITS/ParticleGAN/pull/534) |

## Evidence and compatibility

Old receipts, checkpoints and recorded qualifications retain their original source and execution identities. The [correction registry](forge/evidence-corrections-2026-11-08.json) identifies inspected assisted/adapted coverage observations by exact scientific key and original-record digest. It does not rewrite a stored numerical PASS as a FAIL. It explains why that observation cannot establish the formula or question now being compared. Corrected-source execution has a different identity and cannot inherit an old pass.

Public-API vector and MisGAN checkpoint versions change where the old implementation or saved state cannot provide exact continuation. An old checkpoint must be resumed with its pinned original implementation. Unsupported host mechanisms are reported before training rather than dropped from a recipe.

Caller-owned vector paths without a complete restore owner explicitly export diagnostic-only state and refuse a required checkpoint capability. They cannot present that artifact as a resumable checkpoint. MisGAN standalone particle-cloud storage and its normal-noise transform are declared separately from a positive-width MoG case; supplied priors must match the actual task law and initialization.

The retired runner cannot be revived by `--force`, a direct task call or an internal batch spec. New research measurements require normal frozen Forge study admission, the declared protocol and budgets. A publication refresh reads retained evidence and adds no training. Retained-publication integrity and current-source measurement coverage have separate status fields.

The new controls test the scientific boundary, rather than just exercising a successful code path: damaging the measured neural imputer changes its predictions; its evaluator cannot call ppost; disabling an optimizer prevents hidden model/prior writes; importing providers cannot change a shared architecture; automatic derivatives match finite differences and a zero upstream signal; and fresh-object continuation preserves complete owned state. Software prefixes and constructive/oracle controls confer no full-budget learned-quality credit.

For lint, three control-flow false positives in frozen reproduction sources are exempted only by exact path, message and source SHA-256. Their source bytes are preserved. The remaining reviewed redefinitions are imported pytest fixtures, not blanket permission for new redefinitions. The engineering PR reduces the reviewed baseline from 302 to 297 findings; integration removes one further resolved unused-import exemption (296 remain).

## Remaining scientific questions

These repairs do not supply a winning global recipe, accepted screen calibration, an untouched task-level confirmation result or an image-scale advantage from learned particles. Those remain research questions. The earlier matched prior ablation retains its original scoped seed-0 toy precision result and unresolved CIFAR ordering; the repair does not extend either conclusion.

Before new formulation work, query Forge recall, check the unresolved problems, preregister one substantive recipe delta with an explicit control and finite budget, and retain every required regression. Keep task architecture, initial prior law, data sequence, sampling, budget and evaluation cadence fixed across candidates. Do not run seed-only searches or tune against a confirmation task after seeing its result.

Raw software verification logs stay under the ignored local directory `runs/software/audit-repairs-2026-11-08/`. The final integrated verification and merge links are recorded with the publication repair; component PR descriptions retain their own exact validation scopes.
