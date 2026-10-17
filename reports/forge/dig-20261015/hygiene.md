# Lead 5: leaderboard hygiene (CPU only)

Branch `task/dig-hygiene-20261015`, based on develop plus the 10-15 review
(`reports/forge/WHAT_WORKED_REVIEW_20261015.md`). **No GPU job was run: 0 GPU
seconds.** No default, task definition, gate or budget changed. No new scores are
published.

Terms used below:

- **Family**: an owner of a trainer formulation (for example BCAP). A family row
  can show the measurement of a separate **configuration family**. BCAP's row shows
  the `bcap-dualnorm` configuration.
- **Stale**: a recorded measurement whose task contracts have changed since it was
  measured. It is retained, but it is not current qualification.
- **Wall-clock limit**: the task's `resources.timeout_seconds`. The runner kills a
  job that reaches it.
- **TIMEOUT / INCOMPLETE**: a run that ended before its gate could be graded.
  INCOMPLETE is the label on retained receipts.
- **Sustained gate**: a gate that needs `minimum_stable_checks` (5) consecutive
  passing checks. Tier 3 100-mode tasks and ring16 use one.
- **Earliest-confirmed gate**: word smoke and Gaussian smoke pass when one scheduled
  check passes and an independent redraw at the same state confirms it, provided
  the run completes all its updates.
- **KS**: the Kolmogorov-Smirnov distance between the generated and target 1-D
  CDFs. The Gaussian smoke limit is `cdf_ks <= 0.05`.

## (a) Stale-label display bug: confirmed and fixed

**Hypothesis.** The main table omits BCAP's stale label because the labeller keys
on the family id. The stale row belongs to the configuration family that the row
displays.

**Preregistered rule.** The fix counts if four things hold:

1. The BCAP main row gains exactly the stale label.
2. No score cell changes, and no other row changes.
3. Regenerating with the publication's own documented command changes only the
   label line, the input digest and the reducer hash.
4. The full related test set shows no new failures compared with the base commit.

**Confirmation.** In `technique-inventory.json`, `rows[]` has
`trainer_family: "bcap-dualnorm"` with `selection.freshness: "stale"`. The family
`bcap-pure` has `freshness: None`. The main table row is `family_progress`
family `bcap-pure`. Its current cohort `cohort-cuda-1bf9d7d34422` has
`configuration_family: "bcap-dualnorm"`, so the 22/22, 49/116 and 71/152 cells are
the stale dualnorm measurement. `_stale_measurement_labels` checked only
`family["id"]` (`bcap-pure`) and skipped the row.

There was a second latent problem. The anchor `"](families/bcap-pure.md)** |"`
can never match this row, because the row continues
`** · [BCAP dualnorm ...](families/bcap-dualnorm.md) |`. A naive id fix would
have raised `ValueError`.

**Fix** (`reports/forge/regenerate_technique_inventory.py::_stale_measurement_labels`):

- A family is labelled when its own id, or the `configuration_family` of a cohort
  it displays, is stale. "Displayed" means the cohort at `current_cohort_anchor`,
  which is the same rule `render_leaderboard` uses.
- The label goes at the end of the first table cell, so a configuration link is
  allowed. For the rows that were already labelled, the output is byte-identical.

**Test.** I added
`tests/test_core_inventory_retained_measurement.py::test_stale_label_follows_displayed_configuration_family`.
It checks three things: a stale configuration family labels its owning row, the
score cells are unchanged, and a non-stale configuration adds no label.

**Regeneration.** The previous publication carried `publication_refresh`, so the
matching documented command is:

```sh
OMP_NUM_THREADS=1 python reports/forge/regenerate_technique_inventory.py --refresh-publication
```

The first attempt failed. The worktree was missing recorded source-origin
commits, and `git ls-tree` returned "not a tree object". I fetched the 6 missing
SHAs that origin serves by id (`a0493ca3`, `d2ea851a`, `9563dea5`, `f9f7ed9d`,
`fb7acc77`, `ff94453b`). Two others (`36718f9e`, `9da0e927`) are not on the
remote, but regeneration did not need them. Result:

- `technique-inventory.md` changed in 2 lines: the BCAP label and the input digest.
- `EXPERIMENTS_BY_TIER.md` changed in 1 line, the digest.
- In `technique-inventory.json`, only `provenance.input_digest` and
  `publication_reducer_sha256` changed.
- No family pages changed, and no score changed.

Running without `--refresh-publication` would have dropped the `declared_view`
and `publication_refresh` blocks, so I did not use it.

**Tests.** I ran the 21 test files that import the regenerator or the leaderboard
renderer:

| Run | Failed | Passed |
| --- | ---: | ---: |
| Patched | 124 | 500 |
| Base commit | 124 | 499 |

The failure sets are identical. The one extra pass is the new test. The 124
failures exist before this change: 104 are `FileNotFoundError` from evidence
files that are not in a fresh worktree, and 5 are `NameError: deepcopy` in
`experiments/forge/knowledge.py:632`. The directly relevant file,
`tests/test_core_inventory_retained_measurement.py`, passes 11/11.

**Verdict: PASS.** All four parts of the rule hold.

## (b) Wall-clock limits versus gate failures (analysis only)

**Rule, set before reading the logs.** A verdict is a **wall-clock hit** when
either of these holds:

- Its status is TIMEOUT.
- Its status is INCOMPLETE, its raw terminal status is `timeout`, and its seconds
  are at least the limit.

Each hit is then classified from its saved `run.log`:

- **would-PASS**: the gate's own pass condition was already met before the kill.
  For earliest-confirmed gates, that means a `confirmed_pass` observation exists.
  Only update completion was missing.
- **determined FAIL**: no check had passed, and fewer than 5 checks remained.
- **undetermined**: anything else.

**Counts** in `reports/forge/attempt-coverage.json`, 853 records:

- 19 TIMEOUT and 3 INCOMPLETE.
- **20 are wall-clock hits**: all 19 TIMEOUTs, plus the INCOMPLETE
  `bcap-default-baseline-direction-v1` word run (`raw_terminal_status: timeout`
  at 900.2 s, at least 19,168 of 20,001 lifecycle steps).
- The other 2 INCOMPLETEs (`generator-idle-ema-bcap-direction-v1` word and
  Gaussian) completed every step and are not wall-clock hits.
- **No FAIL is a wall-clock hit.** Of the 186 FAILs with saved timing, none reached
  90% of its limit; the longest was ring16 at 228 of 300 s. The 109 retained FAILs
  have no timing in this file.
- **8 BLOCKED verdicts are downstream of these hits**, all labelled
  `checkpoint parent ...: TIMEOUT/INCOMPLETE`. Seven are `five_word_joint_hold`;
  one is `gaussian1d_stability`.

| Task (limit) | Recipe | Progress at kill | Last saved check | Class |
| --- | --- | --- | --- | --- |
| five_word_joint_smoke (900 s) | signed-confidence-lr-gaussian-v1 | 11,668/20,001 | confirmed pass at step 834 | **would-PASS**; Tier 1 5/6 to 6/6 |
| five_word_joint_smoke | positive-evidence-lr-gaussian-v1 | 12,501/20,001 | confirmed at 834 | **would-PASS**; Tier 1 5/6 to 6/6 |
| five_word_joint_smoke | bcap-default-baseline-direction-v1 (INCOMPLETE) | at least 19,168/20,001 | acquisition checkpoint at 834 | **would-PASS**; Tier 1 5/6 to 6/6 |
| five_word_joint_smoke | kinetic_transport_local_v2 | 8,334/20,001 | confirmed at 834 | would-PASS; Tier 1 still fails (two_pole, unused_token, ring16) |
| five_word_joint_smoke | kinetic-local-dualnorm-g-history-v1 | 10,834/20,001 | confirmed at 834 | would-PASS; Tier 1 still fails |
| five_word_joint_smoke | confidence-lr-finite-cap-gaussian-v1 | 2,501/20,001 | 3 modes, inexact reconstruction | undetermined; Tier 1 already fails on Gaussian |
| rotated100 / staggered100 (3,600 s) | kinetic-local-dualnorm-g-history-bank8-v-v1 | 6,750/7,000 | precision at most 0.565 (gate 0.97); 0 passing checks | determined FAIL ×2 |
| rotated100 / staggered100 | ...-bank8-u-v1 | 6,750 and 6,500 of 7,000 | precision at most 0.569; 0 passing | determined FAIL ×2 |
| ring16_acquisition (300 s) | ...-reference8-u-v1 | 1,567/1,600 | 0 of 94 checks pass (cov. error at least 1.95, gate 0.85) | determined FAIL |
| grid/rotated/staggered100 | ...-reference8-v1 | 3,250–3,500 of 7,000 | precision at most 0.466; 0 passing | undetermined (likely FAIL) ×3 |
| grid/rotated/staggered100 | ...-reference8-u-v1 | 1,750–3,000 of 7,000 | precision at most 0.448; 0 passing | undetermined (likely FAIL) ×3 |
| grid/rotated/staggered100 | cell-quota-dualnorm-cross-v1 | **1** of 7,000 | none | undetermined ×3; one step per hour points to a recipe/runtime defect, not contention |

**Would the pass/fail flip?**

- Six word-smoke hits (5 TIMEOUT, 1 INCOMPLETE). Five of them had already
  confirmed acquisition and would pass if allowed to finish. The prediction assumes
  that nothing in the remaining updates triggers a guard failure.
- For three recipes, Tier 1 flips from fail to 6/6: signed Confidence LR, Positive
  evidence LR, and the matched BCAP direction baseline.
- Each such flip also unblocks `five_word_joint_hold`.
- None of the 13 Tier 3 and ring16 hits would flip to PASS on the saved evidence.
- The review's claim checks out exactly. Confidence LR
  (`generator-confidence-lr-bcap-direction-v1`) is +2 over its matched baseline.
  The +2 is `five_word_joint_smoke` (the baseline timed out at 900 s; Confidence LR
  passed at 846.6 s) plus `five_word_joint_hold` (blocked downstream). The baseline
  had confirmed acquisition at step 834, so on this evidence the two are tied.

**Why the limit is tight.** Word-smoke runs that complete, in saved evidence, take
**724.6 s and 846.6 s** of the 900 s limit, at about 24–28 updates/s. The
timed-out attempt-fill runs managed 2.8–13.9 updates/s. That fill ran
`maximum_per_gpu: 2` on GPUs shared with a desktop and llama-server. Recipe cost
explains part of the gap: the kinetic recipes are slower. But signed Confidence LR
is a near-copy of a recipe that finished in 847 s, and it ran at about half that
speed.

**Recommendation (not applied):**

1. Report wall-clock hits as a separate **NOT MEASURED (wall-clock)** outcome, not
   inside "fail" or "incomplete" totals. Exclude them, and anything they block
   downstream, from paired recipe-versus-baseline deltas. Show
   "acquired before limit" when an earliest-confirmed gate had already confirmed.
2. Treat the wall-clock limit as a safety cap, not part of the measurement. One
   option is a contention-aware cap: measure the per-update time of the first
   N updates against a solo reference, and extend the limit by that factor up to a
   hard ceiling (for example 3× the declared limit). Another is to base word smoke
   on the longest completed run: 847 s × 2, so about 1,800 s.
3. Before counting a wall-clock hit, re-run it once under the one-job-per-GPU
   lease.

## (c) Thin Gaussian smoke margin (saved evidence only)

**The claim checks out.** `reports/forge/bcap-six/readout.json` records the
selected smoothing arm (1e-5): first confirmed pair at update 959, **primary KS
0.044836, confirmation KS 0.033797**. Only **1/24** scheduled pairs pass, and the
endpoint KS is 0.084096, so the endpoint fails. The previous unsmoothed
configuration failed because the confirmation draw scored 0.054185.

| Arm (bcap-six) | First confirmed step | Primary KS | Confirmation KS | Passing pairs | Endpoint KS |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoothing 1e-5 (selected) | 959 | 0.0448 | 0.0338 | 1/24 | 0.0841 |
| smoothing 1e-4 | 459 | 0.0296 | 0.0359 | 3/24 | 0.1371 |
| smoothing 1e-3 | 875 | 0.0377 | 0.0430 | 1/24 | 0.0634 |

**Distribution across recipes.** The KS at the confirmed state is not stored in
`test-performance.json`. Only `passing_observations` (out of 24) and
`first_confirmed_step` are. I use the passing count as the margin proxy:

- Across 180 distinct PASS results, the count of passing scheduled observations is
  `{1: 29, 2: 22, 3: 18, 4: 8, 5: 8, 6: 7, 7: 11, 8: 5, 9: 7, 10: 6, 11: 6, 12: 10, 13: 2, 14: 1, 15: 4, 16: 3, 17: 1, 18: 4, 19: 8, 20: 7, 21: 4, 22: 5, 23: 4}`.
- **16% pass on one observation, and 28% on two or fewer.**
- The median first confirmed step is 292.

Selected recipes:

| Recipe | Passing observations |
| --- | --- |
| GAN v3 release 0.7 | 1/24 |
| BISPF | 2/24 |
| BCAP selected (`recipe-prior-refactor-bcap-pure-v1`) | 3/24 |
| R1/R2 | 6/24 |
| K3P | 7–9/24 |
| KA2 | 7–10/24 |

Endpoint KS (step 1,000) on PASS results ranges from 0.012 to 0.72. Most PASSes
fail the limit at their endpoint, which is expected for an acquisition smoke.

The attempt-fill run logs log only the primary draw. They show first-pass KS
0.0484 (`cell-quota-dualnorm-cross-v1`) and 0.0488
(`kinetic-local-nsgda-cross-v1`), against 0.0173 for `kinetic_transport_local_v2`.

**Conclusion.** The thin margin is general, not specific to BCAP. About a quarter of
Gaussian smoke passes rest on one or two scheduled checks, and several selected
recipes are among them. A Gaussian smoke PASS should be read as "acquired once",
not as a stable fit.

## Commands

```sh
# (a)
python -m pytest -q tests/test_core_inventory_retained_measurement.py
python reports/forge/regenerate_technique_inventory.py --refresh-publication
# (b), (c)  (read-only; (b) reads /ml2/hypergan/attempt-fill-20261015/attempts/*/run.log)
python reports/forge/dig-20261015/hygiene/timeouts.py   > /ml2/hypergan/dig-20261015/hygiene/timeouts.jsonl
python reports/forge/dig-20261015/hygiene/gauss_margin.py > /ml2/hypergan/dig-20261015/hygiene/gauss_margin.txt
```

## Results

| Test | Verdict | Key metrics |
| --- | --- | --- |
| (a) stale label | PASS | BCAP row labelled; 0 score cells changed; JSON diff limited to 2 provenance hashes; test failures 124 = 124 base, +1 new pass |
| (b) wall-clock classification | done | 20 of 22 TIMEOUT/INCOMPLETE are wall-clock hits; 0 FAILs; 5 word hits would-PASS (3 flip Tier 1 to 6/6); 5 determined FAIL; 10 undetermined |
| (c) Gaussian margin | claim confirmed | primary KS 0.044836 / confirmation 0.033797, 1/24 pairs; 28% of 180 PASSes use 2 or fewer of 24 observations |

GPU seconds: **0**.

## Caveats

- The would-PASS predictions in (b) are inferences from earliest-confirmed gate
  semantics. They do not count as gate results, since the runs did not finish their
  updates, and a guard failure late in the run is still possible.
- The (b) timing comparison mixes recipe cost with GPU contention. Without a solo
  re-run, the two cannot be separated.
- The (c) passing-observation counts come from `test-performance.json`, which pools
  Forge and scoped-diagnostic cohorts. Some PASSes are under changed task contracts
  (`definition_match: CHANGED`). Duplicate cells were removed by attempt id.
- Regeneration in (a) needed source commits fetched by SHA. Two recorded SHAs
  (`36718f9e`, `9da0e927`) are not on origin. Regeneration did not need them.
- 124 related tests fail before this change; see (a). That is unrelated to this
  change, but it is a hygiene item on its own.
