# Generalization evidence and archive plan — preparation snapshot

Prepared 2026-09-30T22:46:28.225659+00:00. Destination: `reports/toy100/lrfree-search/feature-cells-cb64-ra/generalization-20260930/` in PR223. This is an archive plan; the **19-gate qualification and final RA14 full suite remain PENDING**. No repository files, frozen evidence or numerical jobs were changed. The queue snapshot has 11/19 completed records, status `RUNNING`; individual observed verdicts are in `QUEUE-SNAPSHOT.json` and `INVENTORY.json`. Remaining records are explicitly pending.

## Closed learned evidence

| Original fresh run | Toy noisy precision / modes / TV | MNIST active39 FD / precision / recall | Reopen events |
|---|---|---|---|
| RA12-auto | 0.668579 / 22 / 0.336016 — Toy FAIL | 1.880969 / 0.767578 / 0.719238 | Toy820, MNIST202 |
| RA13-settled | 0.965332 / 25 / 0.052114 — Toy PASS | 0.544488 / 0.869141 / 0.847168 | Zero in both fixtures |

Both fresh runs completed the unchanged original2000 updates. RA13's nine postupdate Toy score/LR records exactly match frozen RA11; all ten MNIST score/LR records exactly match corrected E22, with ten confident classes. MNIST is a comparative regression fixture with **no newly invented numerical PASS threshold**. Backend selection is capability based: complete bounded raw-output frame plus finite-population feasibility selects feature cells/quarter generator+noise bases; otherwise reference KNN/original bases. Prior and critic bases are preserved.

RA14 performed **no fresh2000-update training**. Its configuration is byte identical to RA13. The only code change is checkpoint helper `_state_to_device`; all fresh training/sampling arithmetic is unchanged. The formal source bridge keeps the original RA13 source, inputs, checkpoints, labels and receipts. It supports carrying that evidence forward without relabeling it as a fresh RA14 training run.

## Repairs and retained failures

1. **Static endogenous R1 fires.** RA12 Toy820 follows the known KA2 objective epoch; MNIST202 occurs in an uncontracted game. Closed scalar reconstruction reproduces270 original rows and suppresses both first static fires with the settled-network witness plus objective-epoch rebase. It preserves original moving fires514/1021 exactly. Existing detector thresholds and all gradient groups are unchanged. The guard uses actual contracted generator/encoder/router/critic ownership; it is not a classifier for external target changes. Actual original moving qualification remains pending.
2. **CPU optimizer context.** Current Torch Adam's accelerator health check could open CUDA for CPU-only parameters. The device-scoped repair preserves accelerator delegation and exact CPU Adam/AdamW update/moment behavior. Thirteen focused checks plus35 existing controls passed without CUDA initialization; marker tests establish forwarding, not GPU numerical performance.
3. **Strict RA13 replay failure.** Both original native versus CPU-map continuations remain FAIL at1008–1010: cross-device conversion broke `last_block is blocks[-1]`, then moved-row rebase left the diagnostic tensor stale. Within the ten-update window, model/optimizer/active-history/RNG/loss/sample bytes match; this does not waive the strict semantic-state failure.
4. **RA14 restoration correction.** A transfer-local Tensor-identity memo preserves that alias. Both corrected original Toy/MNIST replays PASS: two branches×ten updates each,40 updates total; restored state, per-update semantic state, losses and noisy-primary samples agree. Native RA14 continuation also matches the sealed original RA13 native branch. Only original observational `birth_death.last.eval_seconds` is excluded. Distinct storage views/shared container identity are outside the repair.

Retain the RA14 r1 zero-update bridge preflight failure: raw shared configuration was compared before original N1024/z128/batch128 and tuple normalization. It performed zero updates/forwards/sampling and no CUDA initialization; corrected r2 changed adapter preparation only. Retain RA13's untrained first lane with missing external-validator guards and the corrected frozen r2 lane, plus private review preparation errors. Historical closure files keep their historical pending status; the final report should append current outcomes rather than edit them.

## Tests and pending qualification

| Scope | Closed evidence | Final status |
|---|---:|---|
| Settled guard/default controls | 19+35+4=58 PASS | Closed CPU contracts |
| Restoration alias/default controls | 4+54=58 PASS | Closed CPU contracts |
| RA13 integrated suite | 1404 PASS,12 skipped,18 subtests | Historical RA13 PASS |
| RA14 original13 ports +3 moving +3 static native | 11/19 observed in snapshot | **PENDING** |
| RA14 integrated full suite | No final attestation in this plan | **PENDING** |

These test counts overlap; do not sum them into an independent total. Preserve skip reasons. The current frozen lane retains original host/scorer/seeds/budgets, live noisy primary sampling, indexed row IDs, both scheduled moving rotations, and full native five-terminal/100k-holdout conjunction. Source freeze pins103 files including the unchanged external validator and lane. Final quality and validity are distinct: original FAIL stays FAIL even with valid evidence.

## Archive and recommendation

`INVENTORY.json` lists 510 already available small files (17.27MiB), their source paths, target relative paths and SHA256. It includes source/configs, patches/helpers, source/run/failure closures, metric/LR curves, observation traces and independent reviews. Candidate closure digests use compact source-hash JSON; runner digests use path/NUL/source bytes. Both conventions are explicit.

Raw tensor checkpoints, datasets and saved clouds stay local; archived input manifests retain their original path/hash references. No rescoring or checkpoint conversion is needed for archival. After root closes all19 gates and the actual RA14 full suite, append the final receipts/logs and source hashes, state every failure, then byte-verify the archive and intended staged Git scope. Keep original RA12/RA13 failures and bridge preparation errors.

The learned recovery and replay repair support RA14 as the current prospective general candidate. A completed general recommendation requires the pending gates/full suite. Preserve E22 as the broad baseline in comparisons; these two learned fixtures and short replay windows do not establish universal optimizer-shock discrimination, high-dimensional moment coverage, serving equivalence, or scalability.
