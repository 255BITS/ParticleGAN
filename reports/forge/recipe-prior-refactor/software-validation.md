# Recipe-owned priors: software validation

The original full CPU suite failed at `5a38c8505945e8b4ce806f4d1bd8c4b95e4184e6`: **5,886 passed, 96 failed, 35 errored, 299 skipped, and 1 expected failure**. The separately bounded renderer check passed all three tests. The full suite was not repeated.

Later focused checks cover **all 131 exact original failed nodes**. Final memory/tier freshness and family-page/historical controls pass across two saved invocations. This is failure coverage across recorded source revisions, not a claim that a new full suite passed. The [compact validation receipt](software-validation.json) binds the exact-node mapping and preserved evidence hashes.

| Focused verification | Saved outcome |
| --- | --- |
| Archived compatibility and migrated task controls | 168 passed, 10 failed, 7 errored, 1 skipped; affected recheck: 29 passed |
| Prior ownership and public API repairs | 219 passed, 10 skipped |
| Current boundary/source controls | 95 distinct checks passed across initial run and bounded repairs |
| Workflow and publication checks | 345 passed, 1 freshness failure; final publisher controls: 34 passed |
| Final freshness and historical-family controls | 2 passed in first bundle; repaired remaining 3 passed |

The first final bundle retains its three failures (14.6754 seconds); the interrupted repair retains its timeout (13.0016 seconds), with no PASS inferred. A separately declared 60-second completion stage finished the three guards in 7.2497 seconds. The final 120-second publisher stage records 86.8140 seconds, including a passing normal summary-only refresh in 13.3045 seconds. Every stage and earlier failed attempt remains in the receipt; these reporting checks add no training.

The audit retains the immutable original comparator and its numerical assertions. Historical software replays explicitly use task sources `5737ade47dca89b04d338ada20667781d1f2b5df` or `03466efa4de8271b9a2c964406dc6f4f0f260792`; they do not relabel those questions as current qualification. The repair preserves original archive pins and validates actual initial, checkpoint, final and resumed tensors. Current ownership is checked separately against all 33 migrated task declarations.

Current comparisons retain protocol seed 0, the public deterministic initializer, initial prior distribution and capacity, task architecture/data law, sampling law, update budget and evaluation cadence. Recipe-owned prior updates and regularization are explicit deltas; frozen-policy, alias, zero-weight and checkpoint/resume controls are software checks. Historical identity fixtures retain their original initialization cohorts. No software receipt replaces a trained gate or rewrites archived science.

The [integration verification](publication/integration-verification.json) confirms all 1,206 executed files remain unchanged at integrated develop commit `d65b4862c39bb61858de11d61348b5952b687f90`, with both actual execution freezes in its ancestry. The unrelated incoming diagnostic helper lies outside that executed boundary; this report makes no qualification or rerun claim for it.

Bulk logs, JUnit files and tensor packets remain outside Git. Archive IDs:

- `/mnt/ml7tb/ParticleGAN-forge/recipe-prior-refactor-20261010/audit/{full-cpu,renderer,legacy-focused,legacy-failed-recheck}.{json,log,xml}`; original full-suite JUnit SHA256 `0eddf4cac34be3d94fe18242fd6bc920a8bbd51c3a15c8bf115832a2d9cefbd2`.
- `/mnt/ml7tb/ParticleGAN-forge/recipe-prior-refactor-20261010/development/prior-api-repair.{json,log,xml}`; repair commit `ad9daaf259b0273c127f4407db429c718319b1c8`.
- `/mnt/ml7tb/ParticleGAN-forge/recipe-owned-priors-20261010/{root,compatibility}/`; focused receipt names and exact hashes are retained in the compact validation receipt.
- Archived replay repair: `75adade75947a9e4ded1262f542b204afce881d9`; original comparator SHA256 `0ef9c1a1cc30abe3ac02e999d989ef6fbb7374d3ca4b6e85bc87aa9ff7ad2211`.
