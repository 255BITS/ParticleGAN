# Continuous API search — September 26

Three external Codex sessions are testing distinct approaches with `gpt-6-astra`
and max reasoning. **No qualified winner.** The public API base is
`fa511ce010120b502f494d717d01b14b8551eed8`; neither PR is merged.

These are new measurements through `get_recipe()` and `GANTrainer.step()`.
All runs use the declared public fixture and seed 0. No seed sweeps. The learner
does not receive the evaluator's ending or target-change times.

## First measured results

Each retention fraction starts at actual arrival. Slow acquisition is reported
as time, not turned into a failure by imposing the old prehold deadline.

| Candidate / configuration | Initial arrival | Retention until change | Changed-target arrival delay | Retention after changed-target arrival | Longer evidence / scope |
|---|---:|---:|---:|---:|---|
| API-DV1: separate data drift and gradient coherence | 790 | 162/162 | 430 | 178/178 | Stationary 7500: **655/672**, collapse at 4850–5010; rejected despite good short recovery. |
| API-DV2: real-data gate on critic memory | 790 | **159/162** | 400 | 181/181 | Original-target departures; longer tests NOT_RUN. |
| API-DV3: accumulated drift evidence and a slow stationary reference | 790 | **153/162** | 520 | 169/169 | Original-target departures; longer tests NOT_RUN. |
| API-C1: continuously moving critic reference, constant rates | 580 | **172/183** | 570 | 164/164 | Original-target departures reject this version. |
| API-C2: C1 plus a bound on each coordinate's Adam displacement | 1960 | 45/45 | 1890 | 32/32 | Stationary 7500: **296/555** after arrival; rejected for repeated loss of the unchanged distribution. |
| API-C3: bounded optimistic displacement correction | NOT_OBSERVED | — | NOT_OBSERVED | — | Neither target acquired in the declared window; final HQ .1245. |
| API-C4: coupled predictor/corrector with fresh gradients | NOT_OBSERVED | — | NOT_OBSERVED | — | Neither target acquired through 4600; successor API-C5 is testing a local implicit game response. |
| API-C5: local implicit game update, actual hard-copy critic reference | 560 | 185/185 | 270 | 194/194 | Declared .99 reference averaging was not executed; these scores belong to the accidental hard-copy version. A corrected successor must earn its own evidence. |
| API-C6: corrected averaged reference and serial implicit game update | 590 | 182/182 | 300 | 191/191 | Own image and exact checkpoint PASS. **Rejected:** stationary7500 retains638/692 after arrival, with54 departures and minimumHQ0. |
| API-RP1: reversible precision, ordinary public initialization | 640 | **146/166 through 2290** | NOT_RUN | NOT_RUN | Valid partial measurement; the run was stopped under an incorrect assumption about CPU scalar counters. |
| API-RP1-CUDA-EAGER: same rate rule with test-script optimizer initialization | 640 | 177/177 | 500 | 171/171 | Stationary 7500: 687/687 after arrival. **Diagnostic only:** worker edits optimizer state after API construction. |
| API-RP2: precision controller with explicit library-owned initialization | 640 | 177/177 | 500 | 171/171 | Own stationary 7500: **687/687** after arrival. Own 30000: **537/537**, **145/145**, **1879/1879**, **269/269** after each arrival; recovery delays **360, 420, 320**. **Rejected:** frozen img_intensity2 stability fails. |
| API-RP3: precision adds generator-update cancellation, serial execution | 610 | 180/180 | 460 | 175/175 | **Rejected:** frozen img_intensity2 passes0/24; controller stays open. Longer tests NOT_RUN. |

Single-change evaluations end at 4600, with a data change after 2400. The
stationary runs end at 7500. Passing means all eight modes and HQ ≥ .90, sampled
every ten updates using an isolated evaluation stream. There is no 81/81
deadline gate. Every later departure and retrospective final suffix is retained
in [first-results.json](first-results.json), with the compressed raw observations,
applied rates, source ZIPs, declarations and initialization hashes under `evidence/`.
Full checkpoints remain at the original paths recorded in that manifest.

## What these results teach us

API-DV1 correctly reopened after a real target change. Its longer unchanged-target
run nevertheless reached HQ 0 and zero modes before recovering. A short recovery
test alone would have missed this failure. Its first 2400 updates match across
different evaluation budgets. Exact checkpoint continuation is under investigation;
it is not claimed as passed here. The [independent single-run source audit](api-dv1-single-independent-audit.md)
found matching initialization and no target/horizon leakage.

API-C1 removes KA2's reference freezing/reseeding cycle. It greatly improves
recovery stability, but still loses the original distribution. API-C2 limits
individual parameter jumps and acquires later; its longer test shows that many
bounded coordinates can still move together and destabilize learning. It fails
259 observations after arrival, with minimum HQ .0344 and five modes.

API-RP1 exposed a separate reproducibility issue. Moving Adam's scalar step
counters from their ordinary CPU placement to CUDA changes KA2's tensor-based
surprise calculation and the training trace. CPU scalar counters are valid
metadata; their presence alone does not invalidate a CUDA training run. The
original directory name `rp1-invalid-counters` is misleading and retained only
for provenance. The passing CUDA-initialized version cannot qualify the ordinary
API implementation. API-RP2 implements the initialization change in the library
and has now reproduced the single-shift result through public construction.
It also passes its own stationary run through 7500 and its uninterrupted
30000-update run. Target changes after 6000, 7800 and 27000 are followed by
arrival after 360, 420 and 320 updates, with no subsequent departures before the
next change or the end of observation. The declared 9000 prefix independently
records 537/537, 145/145 and 79/79 after arrival. Exact cross-process continuation,
matched K3P and full22 remain incomplete. The first frozen broader task,
`img_intensity2`, rejects this version: only observations 450 and 600 pass out
of 24, with a final passing suffix of one instead of the required five. Final
HQ .90625 does not erase the five preceding failures. The unchanged candidate
and corrected frozen evaluator are verified by the
[independent archived-source audit](api-rp2-image-completed-audit.md). Its
[complete image evidence](broader-results.json) is retained; remaining21 tasks
and the comparator are NOT_RUN after this measured failure. The [independent source audit](api-rp2-single-independent-audit.md)
verifies public factory ownership and exact single-run parity with the diagnostic.

## Ongoing work

The three lanes remain active. Data-drift was refilled at22:18 UTC after its
[completed review](data-drift-wave1-review.md); reversible precision was refilled
at22:19 after [its review](reversible-precision-wave1-review.md). The latter will
consider combining RP2 retention control with C6 implicit updates, earning new
evidence. The constant-rate lane was refilled at22:25 after its
[second review](constant-rate-wave2-review.md), which preserves
C5's useful hard-copy behavior and C6's measured limitations.

At 21:42 UTC the constant-rate lane was first refilled
after its first three measured failures. The [review](constant-rate-wave1-review.md)
and [completed report](constant-rate-wave1-result.md) explain the next direction:
a coupled predictor/corrector update. [Attempt records](attempts.json) preserve
both generations. The data-drift lane now owns the shared checkpoint investigation;
all three lanes observed small cross-process discrepancies despite identical
immediate restored state. The investigation has localized the first difference
to higher-order critic gradient accumulation: autograd node priority changes
between fresh and warm processes. A scoped serial-backward execution mode passes a fresh-process 100-update
replay: all final state hashes and ten observations agree. The
[independent audit](serial-backward-audit.md) confirms scope and context
restoration, and records remaining provenance/documentation details. Its
arithmetic differs from these archived runs; it cannot inherit their quality
scores or supply the candidates' missing continuation checks.

The [broader-task route audit](api-rp2-frozen22-route-map.md) found that historical
runners silently select legacy K3P. Fourteen tasks can be adapted to current
GANTrainer; eight require a component-controller integration. Unsupported routes
remain NOT_RUN. The first frozen image task was tested before that larger
integration effort, with its original model, evaluation stream and quality gates.
API-RP2 failed it; future candidates can reuse the audited evaluator with source
receipts, but must earn their own scores.

API-C5 passes its first single-change window, but a source/checkpoint audit found
that its reference update runs while critic gradients are disabled. The reference
is copied exactly instead of receiving the declared .99 EMA. Its archived scores
therefore describe the actual hard-copy behavior. They cannot qualify a corrected
EMA implementation; that successor must be measured separately. API-C6 now passes its own first
single-change screen with the corrected .99 reference and explicit serial
backward execution. The [delta audit](api-c6-independent-audit.md) confirms the
source repair and saved reference behavior. Its own frozen image and exact
subprocess continuation checks pass under independent audit. The stationary7500
run then fails54 checks after arrival, retaining638/692 with minimumHQ0 and zero
modes; its final passing suffix begins7020. Remaining costly qualification is
gated off. These successes do not erase the measured retention failure. The
final API regression also finds duplicated optimizer post-step hooks in the
copied KA2 wrapper. Its source is preserved; the next candidate must repair this
contract. This does not explain the no-hook training collapse by itself.

The [six-vector preparation scaffold](vector-harness-preparation/README.md)
archives frozen declarations and dependencies only: no training loop or quality
result. Host backend/RNG semantics remain unresolved. It is useful preparation
for a future survivor, not additional C6 qualification.

Each lane preserves failed versions, explains a proposed repair, and tests it
before broad qualification. A survivor still needs
long retention, delayed/repeated changes, uninterrupted 30000-update continuation,
complete checkpoint and horizon checks, a matched K3P comparison, and its own
broader quality evidence. Historical or sibling passes are never inherited.

[Launch receipt](../continuous-eligibility/launch/first-launch.json) ·
[Shared evaluation declarations](../continuous-eligibility/launch/evaluation-protocols.json) ·
[Historical scores and eligibility](../continuous-eligibility/README.md)

Local live dashboard:

```bash
python /ml2/hypergan/monitor-gan.py --batch /ml2/hypergan/gan-attempts/continuous-api-20260926
```
