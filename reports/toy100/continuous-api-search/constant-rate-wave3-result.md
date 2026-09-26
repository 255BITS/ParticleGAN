# Constant-rate API repair: no qualified winner

Three proposals completed from PR195 `fa511ce010120b502f494d717d01b14b8551eed8`; this attempt is at its **3/3 review point**. No merge, publication or release claim. Exact code/artifact root: `/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T222523Z-1514963/constant_rate_stability/20260926T222523Z-1514972/repo/experiments/constant_fresh`. Gate ledger: `/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T222523Z-1514963/constant_rate_stability/20260926T222523Z-1514972/tests.jsonl`. Tested implementations are preserved separately in each run's `source.zip`; current package equals C9's tested package.

| Candidate | Mechanism | Cold acquisition through2400 | Changed target through4600 | Decision |
|---|---|---|---|---|
| API-C7 | Explicit fresh one-step critic reference; inherited secant response |Arrival590;182/182 since arrival|Arrival+270;194/194 since arrival|Stationary7500 FAIL|
| API-C8 | Local same-oracle secant estimate with binary probe backtracking |No arrival; bestHQ .423828|No arrival; bestHQ .258301|Screen FAIL; longer evidence NOT_RUN|
| API-C9 | Same local probe condition; direct extragradient correction |No arrival; bestHQ .187744|No arrival; best/finalHQ .597412|Screen FAIL; image gate FAIL|

C7 retains120/120 historical prehold observations. MinimumHQ since cold/shifted arrivals is .900879/.938232; neither segment has a departure. Its declared3600 comparison has94/94 shifted checks after arrival. Every frozen no-update comparator across C7/C8/C9 passes0/220 shifted checks. C8/C9 have0/240 cold and0/220 shifted passing checks; no arrival means no post-arrival retention or final passing suffix. These are finite-window observations, not proof of impossible eventual acquisition or a new recovery deadline.

**C7 stationary failure:** first arrival590,641/692 passing thereafter,51 failures at2740–2960,6440–6690 and6710–6720 inclusive, observed every10 updates. MinimumHQ .0009765625 and one mode. Final retrospective suffix6730–7500 contains78 checks; it does not erase the separated late collapses. All individual departures/minima/suffixes are in `C7-stationary/assessment.json` and `leaderboard.json`.

| Own frozen image gate | Passing observations | Final passing suffix | Final live quality |
|---|---:|---:|---|
| C7 img_intensity2 |6/24|5/5,500–600|2 modes,HQ1.0|
| C9 img_intensity2 |0/24|0|1 mode,HQ.375|

C7 has1/22 broader passes; C9 has one measured broader failure and21 tasks unrun. Image host remains the frozen residual_upsample16 fixture,32 particles,z8,batch32,600 updates, actual noise stream and original24/five-suffix scoring. These short image windows end before critic-reference initialization and do not prove continuous image stability. Ring/image learner package hashes match within each candidate.

The causal lead was preserved and retested. C5's accidental hard-copy short screen suggested removing C6's long reference lag. C7 implements the copy explicitly, without resetting models, Adam moments, particle history or noise. Its failure shows freshness alone is insufficient. At2710→2750, C7 G displacement grows .020906→1.073617 while the secant norm factor opens .125997→.388510. The fitted-plane correction again contracts a proposal that is itself growing. See `C7-first-excursion.json`; retained C6 excursions and historical KA2 evidence remain under `predecessor-*.json` and `retained-direct-read.json`.

C8 localizes the finite-difference probe until relative field change<=.5, then uses the secant model. C9 instead accepts the locally checked native correction. Both begin every call at probe fraction1 and halve only within that numerical transaction; no persistent decay schedule or LR change. All checks pass, but acquisition fails in the declared windows. C8/C9 cost9.138/8.561 field evaluations per update versus C7's2. Full screen runtimes are 245.0/901.4/943.0 seconds. C9's stronger shifted progress is a retained research lead, not a qualification pass.

Public changes are in `particlegan/ka2.py`, `recipes.py`, `training.py`, `game_update.py`, `local_game.py`, with inherited native-optimizer plumbing in `k3p.py`/`update_limit.py`. `get_recipe(continuous=True,critic_memory="fresh",game_update=...)` and actual `GANTrainer.step(serial_backward=True)` implement the policies. `copied-source.json`, C7/C8/C9 predecessor diffs and `implementation.patch` provide exact provenance and reviewable changes.

Every one of **22,500 quality-run updates** verifies constant G/D/prior rates .00425/.00425/.0085 and positive movement in all three roles. Prior rows remain sparse; all per-update rates/noise/controller/probe/displacement histories and late movement summaries are in `learning-rates.jsonl` and `update-audit.json`. The local methods' smaller positive movements do not establish adequate learning speed.

Continuous mode ignores evaluator budgets for rates, noise and stopping. Input noise initializes .5→0 over360 calls; output0→.029 over720. The799 initial A-penalty calls initialize the critic rule once. These absolute initialization rules never restart on a target change or require a caller phase switch. Thereafter the same memory/update rules apply at arbitrary ages. This source property does not rescue measured quality failures, nor prove literal infinite stability.

Verification:

- **PASS:**108 final regression tests, including native Adam math/counters, repaired ordinary optimizer post-step hooks, serial execution, sparse sampling, local transactions and small-model horizon/checkpoint contracts. The initial fixture-only failure (invalid family-factory call in a new test;35 other tests passed) is preserved; the corrected36-test run and later43-test run pass.
- **PASS:**C7, C8 and C9 each earn their own true subprocess1600→1800 exact continuation, including native optimizer/controller/reference/EMA/RNG and caller data. CUDA parameters, gradients and moment tensors verified; ordinary CPU scalar Adam counters retained.
- **PASS:**C7 identical2400 prefix under4600/7500 evaluator budgets:24 full-state receipts plus2400 applied histories equal. C8/C9 full-size horizon-prefix audits are NOT_RUN after quality failure; unit tests are not substituted.
- **PASS:**nine saved ring checkpoints have exact fresh references and one native Adam counter advance per accepted step. Inherited alpha=.1 telemetry does not describe the fresh copy's decay; see `reference-conformance-audit.json`.
- **PASS:**145 snapshotted source entries plus one explicitly post-run replay dependency supplement; initialization model/EMA/CPU/CUDA/stream hashes match the retained public fixture for all candidates. Shared evaluator declaration remains byte-identical. See `integrity-audit.json`, source/artifact indexes and `manifest.json`.

C7's image archive omitted `summarize.py`, imported transitively through worker.py. Its original archive/declaration remain untouched; `C7-img_intensity2/source-supplement.json` and ZIP supply exact bytes from C7's earlier ring snapshot. This is a transparent replay-artifact correction, not a learner/scoring change or new pass. C9's image archive includes the dependency before execution.

Ledger: **16 executed gates =11 PASS,5 FAIL,0 ERROR;16 SKIPPED**. Six quality runs =2 PASS,4 FAIL. Regression/audit counts are not qualification scores. No seed sweep, coefficient grid, borrowed pass, nested agent, detached worker or alternative Adam kernel.

**NOT_RUN:**delayed/repeated9000 (changes after6000/7800), uninterrupted30000 (additional change after27000), C8/C9 stationary7500, remaining broader tasks and matched ordinary public K3P. The supervisor's latest conditional K3P owner is RP5; no duplicate comparator was launched. All required protocols remain declared in unchanged `evaluation-protocols.json`; historical declarations are untouched.

Recommendation: do not promote any candidate. C7 remains the strongest measured acquisition lead, but fresh memory does not cure late collective instability. The two local-probe repairs add substantial cost without meeting acquisition/broad-quality requirements. Review joint-field conditioning and how role-wise proposal scales affect corrections before another mechanism; do not repeat these unchanged variants, seeds or a scalar-tolerance grid. Preserve C9's gradual shifted improvement for supervisor review, with its explicit image failure and missing longer evidence.

Replay source preparation (does not launch training):
```bash
python experiments/constant_fresh/replay.py experiments/constant_fresh/C7-stationary experiments/constant_fresh/replay-C7
python experiments/constant_fresh/replay.py experiments/constant_fresh/C8-single experiments/constant_fresh/replay-C8
python experiments/constant_fresh/replay.py experiments/constant_fresh/C9-single experiments/constant_fresh/replay-C9
```
The helper restores the pinned base plus the exact source ZIP and prints the explicit Python/GPU/thread command. `run.sh` pins the runtime environment. Logs remain easy to inspect: `tail -f /ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T222523Z-1514963/constant_rate_stability/20260926T222523Z-1514972/repo/experiments/constant_fresh/C9.log`; assessments, raw traces and all failed variants remain in the artifact root. No GPU worker remains running.
