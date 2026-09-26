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
| API-RP1: reversible precision, ordinary public initialization | 640 | **146/166 through 2290** | NOT_RUN | NOT_RUN | Valid partial measurement; the run was stopped under an incorrect assumption about CPU scalar counters. |
| API-RP1-CUDA-EAGER: same rate rule with test-script optimizer initialization | 640 | 177/177 | 500 | 171/171 | Stationary 7500: 687/687 after arrival. **Diagnostic only:** worker edits optimizer state after API construction. |
| API-RP2: precision controller with explicit library-owned initialization | 640 | 177/177 | 500 | 171/171 | Own stationary 7500: **687/687** after arrival. Long continuation running; checkpoint discrepancy unresolved. |

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
It also passes its own stationary run through 7500. Its 30000-update run is
now executing; delayed/repeated changes, checkpoint continuation, matched K3P
and broader qualification remain pending. The [independent source audit](api-rp2-single-independent-audit.md)
verifies public factory ownership and exact single-run parity with the diagnostic.

## Ongoing work

The three lanes remain active. At 21:42 UTC the constant-rate lane was refilled
after its first three measured failures. The [review](constant-rate-wave1-review.md)
and [completed report](constant-rate-wave1-result.md) explain the next direction:
a coupled predictor/corrector update. [Attempt records](attempts.json) preserve
both generations. The data-drift lane now owns the shared checkpoint investigation;
all three lanes observed small cross-process discrepancies despite identical
immediate restored state. That gate remains unresolved.

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
