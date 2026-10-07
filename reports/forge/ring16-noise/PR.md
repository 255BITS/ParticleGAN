Ring16 succeeds under a fixed tiny-noise rule applied to weak gradient singular
subspaces before ordinary polar normalization. The fresh-live every-step arm
first passes at 817, passes its one independent confirmation, and ends with 45
consecutive full passes (covariance .504405, HQ .965576). Covariance marginally
fails once at 850 (.851437 > .85), then all checks from 867 onward pass;
strict post-acquisition hold is not claimed. The 401-only arm ends
with six full passes (covariance .481363), but its first-pass confirmation
fails covariance .958379, so confirmed smoke is FAIL for that arm.

Both GPU arms retain the public deterministic initializer, seed 0, learned MoG,
batch 128, architecture, constant rates, clean sampling and full distribution
bounds. Neither reloads the archive; the boundary arm's complete 400 prefix is
exactly the archived baseline. One formula and isolated checkpointed CUDA
noise stream apply across all G/D matrix weights, with no amplitude tuning.

Validation: saved-gradient implementation gate passes, including captured
ordinary factors, exact same-noise repeats, checkpointed stream after-states
and unchanged ambient RNG. The probe increases paired polar discrepancy
(.252 → .323), so numerical insensitivity is not claimed. Both 1,600-update arms
complete: 3,200 updates, 53.25 training seconds, 194 draws, zero retries. Bindings,
prefix identity, target batches, schedule counts, budgets, raw hashes and
actual-training GIF provenance verify. Bulk states/logs are archived outside
Git with exact SHA/bytes/member receipts. Original preparation and PR331 source
identities remain intact. No production default or qualification change.

Source inventory coverage passes (11,776/11,776 paths); two new current-source
diagnostic entries preserve all original pinned sources.

See reports/forge/ring16-noise/README.md, results.json,
execution-verification.json, archive.json and both actual-training GIFs.
