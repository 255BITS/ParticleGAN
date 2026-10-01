# Top-tier toy proposal merge readiness

The cutoff is the audit's existing **5/5 and 4/5 scientific-quality tiers**,
with a passing declared diagnostic. Three proposals have useful locally
verified diagnostics: **224, 226 and 227 after its signed assertion repair**.
Actual merge eligibility remains unresolved until the current remote heads,
develop base and required checks can be refreshed. No GitHub merge or push
was performed during this preparation.

The later [local Git integration receipt](local-integration.json) records an
actual prospective tree at `f65fca69fbd24c62c2fe044309caae249b241e6c`, based on
the newer known develop `6ec7e578`, with original PR224/226 and committed signed
PR227 repair `0b196047`. All fifteen changed Python sources and the complete
native package exactly match the retained tested overlay. Its thirty changed
files are proposal tests/examples/reports and two run-directory ignore entries;
no production package, library or configuration changes are imported. The
existing 54/1/1 suite is reused under those exact byte comparisons. This local
branch is **not a remote develop merge**; actual remote merge count remains zero.

The concrete follow-up branch is `codex/toy-pr227-signed-use`, based on original
PR227 head `b0e4f420`. Its five files match the portable patch source hashes.
Publishing and checking this repair is the next external step once GitHub
access works; while PR227 is open it can be stacked on its proposal branch.

| Proposal | Quality | Scientific result | Local integration | Decision |
|---|---:|---|---|---|
| 224 | 5/5 | Unsafe native release reproduces; cancelling only the release and safe geometry both remain stable. | Seven tests pass; the native stability test remains strict XFAIL. Prospective tree clean. | Candidate for a diagnostic/test merge, with the native failure explicitly retained. It is not a controller fix. |
| 226 | 4/5 | Exact odd-critic force and estimator oracles pass; all three trained profiles improve. The even critic reduces final RMSE 19.96%. | All 25 focused tests pass on the locally available develop package. Prospective tree clean. | Candidate for the bounded critic-lag diagnostic. No absolute accuracy or full image quality pass is claimed. |
| 227 | 4/5 | H/b neutralization closes the measured common-critic gap; all four saved endpoint ablation deltas demonstrate useful particles. | Signed follow-up: 22 tests pass; saved tensor assertions pass and four harmful-direction controls reject. Prospective original tree clean. | Apply the separate signed patch, verify that new head and its checks, then consider the diagnostic merge. |
| 45 | 4/5 | Archived GAN-v3 gauge-complete counterexample reproduces FAIL/PASS. | Prospective tree clean, but its unadapted source requests the removed public `Recipe.loss_type` API. | Hold current-develop eligibility. The explicit archive adaptation has no KA2/Atlas pass. |
| 196 | 4/5, four laws | Measured generation/imputation results have no frozen binary gate; conditional posterior deficiencies remain. | Whole-branch prospective merge conflicts in `.gitignore`. | Hold. A 4/5 problem definition is not a passing learned conditional sampler. |
| 22 | 4/5 | Training blocked by removed `edit_cap` import. | The whole old branch has 509 changed files and 40 conflicting files against both checked bases. | Hold. Port the isolated toy test rather than importing old config/library history. |
| 153 | 4/5 | Training blocked by removed `Recipe.make_gradient_penalty` API. | Whole-branch prospective merge conflicts in `README.md`. | Hold pending a scoped current-API replay and qualified multi-step evidence. |

Already merged develop problem definitions and source-only reviews do not become
new merge candidates merely because their quality is at least 4/5. The audit
PR225 itself is an evidence report, not an independently graded algorithm PR.

The inherited failure cases also keep their original meanings. In particular,
PR224's **diagnostic reproduces a failure**: its strict expected failure is
not a pass for native controller stability. Merging this test would preserve
the bug visibly until a separate controller repair makes it XPASS. Likewise,
the stronger PR227 test demonstrates useful particles only in its declared
teacher-aligned family, with H and b intervened together.

## Tested source and base ancestry

The locally available develop ref is `de9b9ae5fa9873c3a41407029221556c246a0442`.
The proposals' known base `6ec7e5788e14ea15ddc3e16ac71110458108b6a6` is its
**direct child**, recorded later on the same day. Both have identical
`particlegan/*.py` sources. Therefore the new package-level checks apply to
both source identities, but the Git bases and inherited report/config edits
remain distinct.

Old Git2.34 three-way `merge-tree` checks against both bases find no conflict
markers for original PR224/226/227. Their proposal-only file sets do not
overlap. A bounded isolated overlay uses de9's package and only each proposal's
new test/example/report files, then applies the signed PR227 patch. The combined
suite records **54 passed, one opt-in training test skipped and one strict
XFAIL**, in 20.07 seconds. This is a package integration check, not a full merge
of inherited base changes or fresh remote CI.

Merging the old PR226/227 heads into de9 would also import the intervening
develop commit's 29 files, including `configs/forge/catalog.json`. Those are
not toy changes from either proposal. A current remote develop refresh must
resolve this ancestry before merging; it must not pull the separate config
repair work into an unexplained toy change.

## PR227 signed gate repair

[The portable follow-up patch](pr227-signed-gate.patch) is bound to original
head `b0e4f420856d7607baa98cbef488e569d20a3f31`. It changes the runner, recovered
scorer and tensor-based long regression to require
`zero_code_minus_live > 1e-6` under every mandatory critic. Lower game loss is
better, so an absolute delta incorrectly accepted harmful particle use.

The new software tests keep valid native short-loop ownership/recovery states
and inject a harmful score under each mandatory judge in turn. All four
reject with the signed assertion, while the positive control passes. Under
the original unsigned assertion, the same tests produce **four expected
regression failures and one passing positive control**. These synthetic scores
are software controls, not additional trained models or seed experiments.

The recovery proof also needs a repair: the old verifier admitted only the
fresh checkpoint carrier. It now admits exactly that carrier plus the signed
evaluation gate. Its receipt explicitly distinguishes the archived training
gate `unsigned_effect_v1` from the new classification `beneficial_signed_v2`.
An unrelated training-loop change still fails preflight. Original qualified
source hashes, archived source, failure receipts, curves and checkpoints stay
unchanged; the original training is not rebound to this new source.

[Signed saved-endpoint verification](pr227-signed-endpoints.json) independently
runs the tensor assertions on actual retained states. Its four ablation deltas
are **0.755647 / 0.584848 / 1.185280 / 0.930370**, all positive. It then injects
one harmful score under each judge and confirms rejection, with original
artifact hashes and owned state preserved. New loop construction and scoring
are isolated with `fork_rng` to preserve the caller's RNG. No unchanged campaign
was retrained, and the opt-in full subprocess campaign remains unexecuted for
this follow-up source.

Apply the patch to an isolated checkout of that original proposal head:

```sh
git apply --check /path/to/reports/toy_audit/merge_readiness/pr227-signed-gate.patch
git apply /path/to/reports/toy_audit/merge_readiness/pr227-signed-gate.patch
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
    tests/test_e22_routed_convergence.py \
    tests/test_e22_routed_convergence_neutral.py \
    tests/test_e22_routed_convergence_signed_gate.py \
    tests/test_e22_routed_convergence_long.py
```

To reclassify the existing saved endpoints without rewriting their receipts,
run from the audit checkout, using a fresh output file:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m benchmarks.toy_audit.check_signed_particle_gate \
    --source /path/to/isolated-patched-pr227 \
    --parent /path/to/routed-convergence-v1 \
    --candidate /path/to/routed-convergence-neutral-v1 \
    --out /path/to/local-artifacts/signed-endpoint-reclassification.json
```

## Remote check limitation and next action

The earlier retained review records successful original PR226/227 Python
3.10/3.11/3.12 CI and builds. Those runs do not verify the new patch head, and
they skip the long training test. Fresh authenticated PR heads, develop head,
required status checks and merge permissions are **unknown** in this network
restricted turn. Old green CI and a clean local tree cannot authorize an actual
merge of an unverified remote revision.

Once authenticated access is available, refresh every head and required check,
publish the PR227 repair as a separate follow-up preserving original results,
and merge eligible diagnostics sequentially while rechecking each updated
base. Do not force, bypass required checks or auto-merge the hold cases.

Exact compact source bindings, prospective conflict lists, test counts and
external log hashes are in [merge-readiness.json](merge-readiness.json).
Full stdout, tensor checkpoints and three-way tree output remain outside Git
under the recorded artifact locations.
