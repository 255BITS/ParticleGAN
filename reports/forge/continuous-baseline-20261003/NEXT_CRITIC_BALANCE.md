# Next critic balance: proposal only

**DECLARED_NOT_EXECUTED / PROPOSED.** Declare one complete shared config for Atlas and one for E22: `lr=0.0053125`, `prior_lr_mult=1.5`, **`d_lr_mult=2.25`**, explicitly overriding every supported host. No models, samples, updates, gate rescoring, resource reservation or default changes occurred. Root finalizes source/admission/budget after the faithful Atlas19 baseline concludes.

The question is whether higher nominal critic response changes shape persistence under the intact family policy. The saved broad run has original 1200 PASS/study-INCOMPLETE and new 1350 study-FAIL (3/5 holds; KS fails at 1250/1300 and recovers). No new discrete move, reopen or surprise fire is recorded. The generator/prior/controller cause remains unresolved; this contrast tests sensitivity, not a demonstrated defect. See [HOLD_RESULTS.md](HOLD_RESULTS.md).

## Rate ownership and inherited host differences

C6 broad/vector hosts declare `d_lr_mult=1.5`, nominal D LR 0.00796875. Their saved 1200 effective D LR is 0.00597655784264745, approximately 1.125× the Recipe generator LR. That 1.125 is realized policy output, not the frozen multiplier. The proposed 2.25 is **+50% versus declared 1.5**. Image/native host defaults are 1.0, so the explicit shared override is 2.25× their declared bases.

| Host | Updates | Eval samples | Baseline declared D | Baseline nominal D LR | Proposed nominal D LR | Cap + export |
|---|---:|---:|---:|---:|---:|---:|
| image-develop-img_intensity2-source-transpose12 | 600 | 1024 | 1.0 | 0.00531250 | 0.011953125 | 180s +60s |
| api-vector-two-broad | 1200 | 4096 | 1.5 | 0.00796875 | 0.011953125 | 180s +60s |
| api-grid100 | 7000 | 20000 | 1.0 | 0.00531250 | 0.011953125 | 2100s +60s |
| api-rotated100 | 7000 | 20000 | 1.0 | 0.00531250 | 0.011953125 | 2100s +60s |
| api-staggered100 | 7000 | 20000 | 1.0 | 0.00531250 | 0.011953125 | 2100s +60s |
| api-vector-unequal-mass | 1200 | 4096 | 1.5 | 0.00796875 | 0.011953125 | 180s +60s |
| api-vector-anisotropic | 1200 | 4096 | 1.5 | 0.00796875 | 0.011953125 | 180s +60s |
| image-develop-img_bars4-source-transpose12 | 600 | 1024 | 1.0 | 0.00531250 | 0.011953125 | 180s +60s |

The explicit 2.25 override wins all inherited host values. Other adaptations stay fixed: images use 32 rows/z8/batch32; ordinary vectors use 256 rows/z4/batch128 and their frozen betas/prior regularizer; native tasks retain 20,000 rows/z2/batch2048. The full family controls remain intact: DV12/stationarity, row evidence/holds, birth/death/isolation, learned output noise, serving averaging 4, and Atlas auto-backend/settled guard. Sixteen pure Recipe checks change only `d_lr_mult`.

Actual C6 terminal effective rates exist only for the two smoke hosts: intensity at 600 has D LR 0.005165296435888595; broad at 1200 has 0.00597655784264745, matching across families. The six deeper cases were not executed in that config, so their effective terminal rates are **NOT_MEASURED**.

`Recipe.make_critic_optimizer` sets the nominal D rate to `lr*d_lr_mult`. Policy stationarity then applies a critic scale floored relative to the table scale, followed by payoff damping when enabled; optimizer guards, moments, EMA and penalty behavior remain active. Future realized D rates therefore need not increase by a fixed ratio. Auto-selected native Atlas feature cells retain the existing G/noise factor 0.25 and D/table factor 1: nominal mapped D/G becomes 9 versus the previous 4. Small-population reference paths and E22 preserve their mappings. These are inherited backend differences, not hidden changes. ([Recipe](https://github.com/255BITS/ParticleGAN/blob/4749b2780add539df4bd8d2dd1d3cc9f002f77ad/particlegan/recipes.py#L515), [dynamic rates](https://github.com/255BITS/ParticleGAN/blob/4749b2780add539df4bd8d2dd1d3cc9f002f77ad/particlegan/policy.py#L619), [feature mapping](https://github.com/255BITS/ParticleGAN/blob/4749b2780add539df4bd8d2dd1d3cc9f002f77ad/particlegan/feature_policy.py#L125))

## Scientific decision and fixed denominator

Keep eight required cases per family, 16 cells total, and one complete config per family. Scientific execution starts at the original provider-owned initialization with seed 24002 and unchanged named streams. It does not modify the trained 1200 checkpoint and call that a new full-budget config. Original targets, hosts, 600/1200/7000 horizons, real-data laws, evaluation sampling and every numerical bound remain fixed. Images/vectors remove additive output noise for their primary gate while retaining DV12/selected serving. Native primary evaluation includes its original output noise; noise-off diagnostics stay separate. Neither historical noisy Atlas19 nor ordinary clean-MoG results supply new qualification credit.

The first two prerequisites are full intensity 600 and broad 1200, in that order. Each must pass its original full-budget/sustained gate and the added first-window contract before six deeper cases become eligible. Confirm the first five consecutive primary PASS checks, require five later checks and retain every subsequent primary check. Later failure means FAIL; late arrival with insufficient later checks means INCOMPLETE. No reacquisition, horizon extension, weakened gate or unchanged retry follows. Each family stops at its own first non-pass; unrun cases remain UNKNOWN in its eight-case denominator.

Metric cadence stays 24 post-update observations, independent of nine GIF frames. The latest first confirmation leaving five later checks is 475 for images, 950 for vectors and 5542 for natives; the earliest five-check confirmation is 125/250/1459. Thus broad first confirmation at 1100 is still study-INCOMPLETE at 1200. Success requires all eight original and added gates under the same config. No best-case mixing, speed ranking, calibration credit or default adoption follows.

The numeric prediction is improved retention **without losing either prerequisite**: all original bounds pass and every primary check after the first confirmed window stays within those same bounds, with five later checks available. Any original non-pass, post-confirmation KS >0.06 on broad, failed later bound on another host, or insufficient confirmation/hold time falsifies full qualification for this config. A positive outcome would establish this finite sensitivity result, not prove why the old config failed.

## Smallest faithful execution path and binding blockers

Public Recipe/Forge ownership and the API override guard already admit positive `d_lr_mult` for all eight unconditional hosts. The frozen family-study coordinator does not: it admits only its four two-LR/two-prior grids and old two-field capacity overrides. This singleton three-field pair **cannot run through that coordinator as written**. Preserve those old grids and add one isolated audit helper for this exact pair. Reuse the unchanged public `api_run` child CLI and per-case strict receipt verification; the helper only owns the explicit two-config/16-cell denominator, prerequisite progression, first-window decision and finite paid-cost admission. No new optimizer/training loop or production/default edit is needed. ([API guard](../../../benchmarks/toy_audit/api_contract.py), [field ownership](../../../experiments/forge/boundaries.py), [frozen study guard](../../../benchmarks/toy_audit/api_family_search.py))

All 16 old capacity cards retain their historical SUPPORTED scope. Twelve vector source bindings match reviewed develop 4749; four image cards differ only in a WordFixture AST change, but strict whole-module hashes still require rebind. Every candidate cell needs a new full Recipe/initial-rate/zero-update state and actual public-sampler certificate. The old constructive learned G/D/prior/output-noise parameters can be a starting point; **old full controller/optimizer state, recipe/rate bindings and sample attestations cannot be relabeled**.

Before spending on science, prepare a fresh public trainer/policy/optimizer ownership under the candidate Recipe, with retained constructive parameters installed consistently before controller/average initialization. Preserve full horizons and source-owned first-shape lifecycle where required; record the new zero-update state and actual selected sampler under the unchanged original gates. Bind every package/provider/scorer file, Recipe, source/runtime and new artifact hash. This is necessary CPU snapshot-capacity evidence, not CUDA training or convergence credit. Its constructive initialization remains separate from the scientific run's unchanged random initialization. Missing raw witness artifacts or failed conformity remain honest blockers.

Then invoke each eligible existing case through the public CLI using the three explicit overrides, its default update/evaluation count, unchanged wall cap and nine frames. There is no seed flag or reduced step/sample override. Strict per-case original/scientific status, new source/Recipe/artifact hashes and added hold decisions must agree; a new tiny aggregator covers only these two candidates rather than borrowing the old eight-trial grid selector.

At inspection, the live image provider also differs from 4749 only in unrelated WordFixture prior-beta lines. No live image provider was imported. Freeze the actual physical source and reconcile that hash before launch. Full source, six deduplicated resolved Recipe records, all original metadata/bounds, exact saved rate identities and admission requirements are in [next-critic-balance.json](next-critic-balance.json). No candidate capacity replay was performed here.

## Finite proposed budget

Unchanged C6 caps plus 60s export give **7680s per family / 15360s for the pair** as the full worst-case allowance sum: 960s for both two-smoke sets, then 14400s for conditionally eligible deeper cells. This is a conservative proposed ceiling, not expected runtime or a reservation. It is disjoint from Atlas19, prior C3–C6, H1 and H2 budgets; it does not reuse unused quotas or extend old scientific horizons. Root finalizes after baseline completion and current GPU admission. Paid interruptions/timeouts stay charged; no automatic retry or extra parameter values are declared.

Reviewed develop: **4749b2780add539df4bd8d2dd1d3cc9f002f77ad**. None of the 42 old executed recipes has d2.25. This is a new balance contrast, not an unchanged failed config or seed study. Capacity/software admission is pending; scientific cells are NOT_RUN and future rates/GIFs are UNKNOWN. The separate weak serve-average note is excluded.
