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
The [supervisor interpretation](arrival-stability-supervisor-interpretation.md)
also separates first threshold contact from early settling transients. All
failures and original automated labels remain recorded; a retrospective suffix
alone is insufficient, and longer verification remains required.

| Candidate / configuration | Initial arrival | Retention until change | Changed-target arrival delay | Retention after changed-target arrival | Longer evidence / scope |
|---|---:|---:|---:|---:|---|
| API-DV1: separate data drift and gradient coherence | 790 | 162/162 | 430 | 178/178 | Stationary 7500: **655/672**, collapse at 4850–5010; rejected despite good short recovery. |
| API-DV2: real-data gate on critic memory | 790 | **159/162** | 400 | 181/181 | Original-target departures; longer tests NOT_RUN. |
| API-DV3: accumulated drift evidence and a slow stationary reference | 790 | **153/162** | 520 | 169/169 | Original-target departures; longer tests NOT_RUN. |
| API-DV4: data evidence plus a stability brake, serial execution | NOT_OBSERVED | — | 460 | 175/175 | Initial target not acquired by2400; finite-window result, not impossibility. |
| API-DV5: payoff imbalance drives reversible mobility | 550 | 186/186 | 350 | 184/186 | Early recovery misses2760/2780, then182 passing checks from2790. Longer stability UNVERIFIED; DV6 prioritized. |
| API-DV6: current data evidence authorizes memory release | 550 | 186/186 | 390 | 182/182 | Own stationary7500: **696/696**. Completed30000 with early recovery transients and long stable suffixes. **Rejected:** bars4/blobs4 each0/24; other two images pass. |
| API-DV7: payoff feedback adjusts critic rate alongside data evidence | 640 | 177/177 | 330 | 183/188 | Early settling misses2740–2780; then182 passes. Stationary7500 **687/687**; four images pass. Completed30000: delays310/270/250, early settling then long stability. **Rejected:** unequal_mass0/24; rare-component spread collapses (.0213 eigenratio vs required.15). |
| API-DV8: real/fake moment discrepancy requests mobility | 640 | 175/177 | 310 | 171/190 | **Rejected:** repeated departures; shifted minimumHQ.07568 and2modes; finalsuffix73. |
| API-DV9: moment discrepancy drives prior mobility | 580 | 181/183 | 320 | 189/189 | Early initial misses590/600 then180 straight. **Rejected:** unequal_mass0/24, covariance error1.1702 and eigenratio.02724. |
| API-C1: continuously moving critic reference, constant rates | 580 | **172/183** | 570 | 164/164 | Original-target departures reject this version. |
| API-C2: C1 plus a bound on each coordinate's Adam displacement | 1960 | 45/45 | 1890 | 32/32 | Stationary 7500: **296/555** after arrival; rejected for repeated loss of the unchanged distribution. |
| API-C3: bounded optimistic displacement correction | NOT_OBSERVED | — | NOT_OBSERVED | — | Neither target acquired in the declared window; final HQ .1245. |
| API-C4: coupled predictor/corrector with fresh gradients | NOT_OBSERVED | — | NOT_OBSERVED | — | Neither target acquired through 4600; successor API-C5 is testing a local implicit game response. |
| API-C5: local implicit game update, actual hard-copy critic reference | 560 | 185/185 | 270 | 194/194 | Declared .99 reference averaging was not executed; these scores belong to the accidental hard-copy version. A corrected successor must earn its own evidence. |
| API-C6: corrected averaged reference and serial implicit game update | 590 | 182/182 | 300 | 191/191 | Own image and exact checkpoint PASS. **Rejected:** stationary7500 retains638/692 after arrival, with54 departures and minimumHQ0. |
| API-C7: explicit fresh critic reference and serial implicit update | 590 | 182/182 | 270 | 194/194 | Image/checkpoint pass. **Rejected:** stationary641/692, late collapses2740–2960 and6440–6720. |
| API-C8: local secant probe with backtracking | NOT_OBSERVED | — | NOT_OBSERVED | — | Complete4600; neither target reached; finalHQ.2583. Exact numerical solver retained as a research lead. |
| API-C9: local extragradient acceptance | NOT_OBSERVED | — | NOT_OBSERVED | — | Complete4600; finalHQ.5974 while improving. Separate image **FAIL0/24**, finalHQ.375. Own exact checkpoint replay passes. |
| API-C10: role-balanced joint secant update | 540 | 187/187 | 250 | 196/196 | Own replay passes. **Rejected:** intensity image4/24, final suffix1; finalHQ.96875 does not establish stability. |
| API-RP1: reversible precision, ordinary public initialization | 640 | **146/166 through 2290** | NOT_RUN | NOT_RUN | Valid partial measurement; the run was stopped under an incorrect assumption about CPU scalar counters. |
| API-RP1-CUDA-EAGER: same rate rule with test-script optimizer initialization | 640 | 177/177 | 500 | 171/171 | Stationary 7500: 687/687 after arrival. **Diagnostic only:** worker edits optimizer state after API construction. |
| API-RP2: precision controller with explicit library-owned initialization | 640 | 177/177 | 500 | 171/171 | Own stationary 7500: **687/687** after arrival. Own 30000: **537/537**, **145/145**, **1879/1879**, **269/269** after each arrival; recovery delays **360, 420, 320**. **Rejected:** frozen img_intensity2 stability fails. |
| API-RP3: precision adds generator-update cancellation, serial execution | 610 | 180/180 | 460 | 175/175 | **Rejected:** frozen img_intensity2 passes0/24; controller stays open. Longer tests NOT_RUN. |
| API-RP4: precision plus implicit game update | 1250 | 108/116 | 350 | 170/186 | **Rejected:** intensity image0/24. |
| API-RP5: smoothed precision signal plus implicit game update | 570 | 184/184 | 270 | 194/194 | Stationary7500: **694/694**; all four images and six vectors pass; checkpoint and horizon checks pass. Completed30000: delays350/450/250, with544/544,146/146,1876/1876,276/276 after arrival. Broader/API/K3P incomplete. **Not selected:** mode_hold returned0/24, with independent host/source audit pending. |

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

The strongest result so far is **API-RP5**, with unchanged-target retention694/694 and
four independently verified image passes and six vector passes. It closes precision on its own, reopens
after the single target change, and settles again. This remains partial evidence:
its complete30000 run has no post-arrival departures across three changes, recovering
after350/450/250 updates. It reopens after the first change; the next two recover
while rates remain reduced. Remaining broader tasks, long-change checkpoint
replays, component API coverage and matched public K3P are incomplete. The precision lane owns that comparator conditionally
on surviving the remaining gates. **API-DV7** also passes all four images and
retains an unchanged target687/687. Its first shifted threshold contact is2730,
followed by five early misses and a sustained suffix from2790. Its complete30000 run reaches the changed targets after310,270,250 updates, with
four early misses in the first recovery, then long stable periods. It is now
**rejected for broader quality**: unequal_mass passes0/24. The rare component has
enough occupancy but too little spread in one direction: eigenratio.021303
against the required.15. Two_broad and unequal_width pass18/24 and20/24. All
parameters, package and frozen scoring were independently verified. Its own
2400→2500 fresh-process replay passes; none of this removes the measured failure.

The small-particle `mode_hold` host has now returned **0/24** for RP5. Its
independent adapter audit is pending; the result blocks selection while we check
whether it represents learner quality or a host mismatch. The ten broader passes
and full long-run strengths remain recorded either way.

The first direct [public K3P comparison](k3p-comparison-results.json) is complete
on `vector_unequal_mass`, with the same frozen task, canonical parameters and
scorer. K3P passes21/24, with its final16 checks passing from450; RP5 passes18/24,
all consecutive from350. K3P first touches the threshold earlier (150 versus350)
and finishes more accurately (normalized distance .0563 versus .1139). This is
one task, not overall dominance or recovery proof. The exact released0.8 package
uses its declared1200-update benchmark schedule; RP5's learner has no ending.

API-DV6 retained the stationary target696/696 and completed30000 updates. First
arrival delays were410,420,370; two early misses at8280/8290 and one at27380 are
kept alongside stable suffixes of1871 and262 checks. These are settling transients,
not late losses of an established target. Its separate image failures reject it:
bars4 and blobs4 each pass0/24, ending with only3 and2 qualified modes. All successes
and failures remain available to the next data-drift attempt.

API-C7 also passed short recovery, image and exact replay tests before losing an
unchanged target twice:641/692 stationary checks, minimumHQ.00098. The constant-rate
lane is testing the next distinct local numerical mechanism. The search continues
with three external Astra/max sessions and reviewed replacements.

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
result. The [backend/fixture audit](api-dv6-frozen-vector-backend-contract.md) now resolves
the canonical CUDA route and verifies all six initial parameter fixtures. Explicit
isolated observation noise differs from historical K3P global noise and must be
declared. This is preparation, not additional C6 qualification.

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
