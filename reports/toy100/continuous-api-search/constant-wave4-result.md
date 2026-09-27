# Constant-rate conditioning: no qualified winner

Three proposals completed from PR195 `fa511ce010120b502f494d717d01b14b8551eed8`. This attempt is at its **3/3 review point**. No merge or publication. Public experimental options remain opt-in; no release default was promoted.

Artifact root: `/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T232429Z-1607588/constant_rate_stability/20260926T232429Z-1607596/repo/experiments/constant_conditioning`. Gate ledger: `/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T232429Z-1607588/constant_rate_stability/20260926T232429Z-1607596/tests.jsonl`. Earlier incremental notes are preserved in `progress-history.md`.

| Candidate | Mechanism | Cold arrival / retention to2400 | Shift delay / retention to4600 | Own image gate |
|---|---|---|---|---|
| API-C10 | Joint secant fit with equal native-proposal energy per role |540 /187 of187|250 /196 of196|FAIL:4/24, final suffix1|
| API-C11 | Separate role response planes from the joint probe |660 /175 of175|280 /193 of193|FAIL:0/24, suffix0|
| API-C12 | Original joint plane, then per-role nonlinear implicit-residual check |NOT_OBSERVED|NOT_OBSERVED|FAIL:0/24, suffix0|

C10/C11 have **no departures** in either ring segment. Their final passing suffixes begin at540/2650 and660/2680 respectively and cover all listed post-arrival observations. Minimum HQ since cold/shifted arrival is .91015625/.914306640625 for C10 and .907470703125/.910888671875 for C11; minimum modes is8. Both retain120/120 observations in the historical prehold window. These favorable screens do not establish late stationary stability.

C12 passes0/240 cold and0/220 shifted observations, with prehold0/120. Cold best/final HQ is .044921875/.043701171875; shifted best/final is .1015625. No arrival means post-arrival retention/minima are undefined and no final passing suffix exists. This is finite-window **NOT_OBSERVED**, not proof that acquisition is impossible or a new deadline rule. At the prescribed3600 comparison, C10/C11 retain96/96 and93/93 shifted checks after arrival; C12 has eight modes/HQ .065673828125 without arrival. Every frozen no-update comparator passes0/220 post-change checks. Exact transitions, minima and endpoints remain in each `assessment.json`, raw metrics and `leaderboard.json`.

| Frozen img_intensity2 | Final HQ / quality modes | Actual failure |
|---|---|---|
| C10 |.96875 /2|Passes475,500,525,600; HQ .8125 at550 and575 breaks the required final five-check suffix.|
| C11 |.65625 /1|No passing observation in24 checks.|
| C12 |.46875 /1|No passing observation in24 checks.|

Each candidate has **one measured broader failure and21 tasks NOT_RUN**, with no inherited broader passes. The frozen image host, all24 observations and original quality/five-suffix thresholds are preserved. Only fixed host resources differ from the ring (32 particles,z8,batch32,600 updates); each image package matches its candidate's ring package. These image windows precede critic-reference activation, so their failures implicate the update rules before the long-memory phase. No samples were visually selected.

C10/C11/C12 ring runtimes are252.90/225.70/1719.76 seconds. Their mean field evaluations per accepted ring update are2/2/17.4743. C12 accepts numerical fractions from1/512 to1/16, median1/128, restarting at1 every call. Every accepted per-role implicit residual satisfies .5. In the fixed cold2400 window the critic alone fails the last rejected trial on1507 updates; average cost is16.7858 fields. Numerical conformance does not establish quality. `C12-cold-residual-diagnosis.json` preserves this conditioning failure.

The retained evidence remains useful. Original constant KA2 collapses at1750 with HQ .0029296875, surprise .84193396 and402 skipped reference updates; by1800 surprise is12.4599 and anchor weight0. Its full native1750 checkpoint is unavailable: retained baseline tensors cover initial/2400/4600 only. We inspected those accessible native moments and own C10 states1740/1750 (`native-state-read.json`), without rerunning the unchanged baseline. C7's recorded G movement grows .020906 at2710 to1.073617 at2750 while the response factor opens .125997→.388510. C7's later stationary failures remain rejection evidence, despite its successful ring/image screens. C8/C9's slow local corrections and C9's separate image failure remain preserved in the predecessor and research archives.

C10 tested role scaling to address this conditioning tradeoff. Its image misses motivated C11's independent responses; C11's final image G movement .251757 versus D .031457 and zero passing observations show that separation also fails. C12 therefore retained the joint fit but checked the proposed implicit equation against another same-oracle field evaluation. Severe numerical damping returned without adequate acquisition. These are distinct mechanisms, not seeds or a tolerance/coefficient grid.

Every one of **15,600 quality-run API updates** records constant generator/critic/prior rates **.00425/.00425/.0085**, controller state, actual noise and positive movement in all three roles. Prior rows remain sparse:1911–1979 per ring update and16–26 per image update. Ring last500 mean G/D/prior L2 movements are C10(.004876,.021413,.014176), C11(.026072,.063822,.027762), C12(.002048,.001882,.006183). Small positive motion is not substituted for quality. `update-audit.json` and per-update histories retain the full accounting.

The API uses actual `get_recipe(continuous=True,critic_memory="fresh",game_update=...)` and `GANTrainer.step(serial_backward=True)` with native non-fused PyTorch Adam. Implementations are `particlegan/game_update.py`, `checked_game.py`, `training.py`, `recipes.py` and inherited `ka2.py`/`k3p.py`/`update_limit.py`. Current package equals C12's tested package. Each run's immutable `source.zip` preserves its own implementation; `copied-source.json`, per-candidate predecessor diffs, `C12-C9-helper-provenance.json` and `implementation.patch` make lineage and changes reviewable.

Continuous mode ignores evaluator budgets for rates, noise and API stopping. Input noise initializes .5→0 over360 calls, output0→.029 over720, then stays fixed. The799 initial A-penalty calls initialize the critic rule once; the200-observation spike-guard startup and Adam/EMA smoothing also have no target-change or evaluator endpoint input. These cold-start rules never restart on a target change, need no caller phase switch and apply the same subsequent rule at arbitrary ages. Fresh references copy each accepted critic after activation; inherited alpha=.1 telemetry is not a fresh-reference decay. Finite failures and conformance tests cannot prove infinite stability.

Verification:

- **PASS:**111 final public/API regression tests; earlier own C10/C11 runs pass74/8. Coverage includes native Adam equality and ordinary single-hook boundary, explicit rejection of staged optimizer hooks, sparse/same-oracle sampling, accepted clocks, numerical residuals, small-model horizon independence and beyond-budget stepping.
- **PASS:**each candidate's own genuine subprocess1600→1800 continuation, including models, native optimizer/controller/reference, EMA, managed RNG and caller data stream. CUDA parameters, gradients and moments verified; ordinary CPU scalar Adam counters preserved.
- **PASS:**12 saved states have the expected native/controller clocks and devices; all9 activated references equal accepted critics. Three600-update image states correctly have no activated reference.
- **PASS:**149 source entries across six quality runs match their immutable archives. All ring initial model/EMA and RNG/data-stream hashes match the retained public fixture. Image/ring learner packages match within each candidate. Shared `evaluation-protocols.json` is byte-identical. CPU initialization followed by CUDA training, FP32, deterministic algorithms and TF32-off controls remain intact.
- Replay preparation initially raised `TypeError` because host Python lacked tarfile's `filter` argument. That setup ERROR is retained; a validated fallback now prepares the pinned source successfully. No training ran during either preparation. Its initial elapsed time was not instrumented and is explicitly null in the ledger, not fabricated.

Ledger: **16 executed gates =11 PASS,4 FAIL,1 resolved setup ERROR;18 SKIPPED**. Six quality runs are2 PASS/4 FAIL. Audit/regression counts are not qualification scores.

**NOT_RUN:**all candidates' stationary7500, delayed/repeated9000 (changes after6000/7800), uninterrupted30000 (additional change after27000), full-size differing-budget prefix checks and remaining broader21 tasks. Own image failures gate off expensive qualification. The declared full-size prefix checks are not replaced by CPU unit tests. Matched public K3P is NOT_RUN here; the supervisor assigns the ring comparator to RP5. No duplicate comparison or borrowed scores. The shared longer protocols remain declared unchanged.

Public integration remains incomplete: full correction requires `GANTrainer.step`; optimizer factories alone do not install it. Conditional/MoG/encoder, auxiliary-loss and multi-critic hosts need explicit transaction support before full22 qualification. C11/C12 now reject optimizer step/state/load hooks before any preview, preventing repeated external side effects; accepted-update effects can run after `step` returns. Arbitrary Python attributes and caller-owned state remain outside rollback ownership. No default promotion is warranted.

Recommendation: retain C10 as the strongest current ring acquisition lead, with its image failure and unmeasured stationary stability explicit. Review a richer coupled response model that preserves cross-role motion without the critic-dominated damping exposed by C12. Do not repeat these unchanged variants, weaken image scoring, sweep the .5 tolerance or launch new seeds. The supervisor review point is reached.

Replay source preparation (does not train):
```bash
python experiments/constant_conditioning/replay.py experiments/constant_conditioning/C10-single experiments/constant_conditioning/replay-C10-review
python experiments/constant_conditioning/replay.py experiments/constant_conditioning/C11-single experiments/constant_conditioning/replay-C11-review
python experiments/constant_conditioning/replay.py experiments/constant_conditioning/C12-checkpoint experiments/constant_conditioning/replay-C12-checkpoint-review
```
The helper restores the pinned base plus the exact source archive and prints the explicit benchmark Python/GPU/thread command. Failed variants should not be rerun unchanged without a new request. Logs remain in `C10.log`, `C11.log`, `C12.log` and corresponding image/replay/regression logs; `tail -F experiments/constant_conditioning/current.log` follows the latest log. `manifest.json` hashes the retained source, initialization, checkpoints, results and logs. No benchmark worker remains running.
