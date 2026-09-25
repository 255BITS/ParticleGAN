# Search inventory at user-requested pause

**Search stopped on 2026-09-25. K3P remains selected; no new formulation meets
the complete qualification bar.** The user requested no new attempts and asked
that existing work be allowed to finish. The eight most recently tracked agents
had already exited around 19:06 UTC when stop handling began at 19:08 UTC.
Seven have exit 1; the Codex audit has exit 0 but delivered no implementation or
result report. Their exit cause is not established. Progress messages saved as
`final.md` are not completed reports.

The rolling launcher now rejects new attempts, including dry runs. No registered
search agents or benchmark descendants remain live. A separate K3P baseline
audit under `/tmp/k3p-audit-20260925` was still running at inventory time; it was
left untouched and its results are not included here. See the timestamped
[stop receipt](search-stop.json) and [search state](../continuous-search-state.json).

## What is best now?

Recovery counts below are **passing checks out of the required 81**, not a
success probability. A high count cannot compensate for a failed hold, transfer
gate, or incomplete control audit. Each modified formulation earns its own tests.

| Formulation | Own hold + extension | Live recovery | Other evidence and binding limitations | Decision |
|---|---|---|---|---|
| **K3P, selected baseline** | 1200/1200 + 300/300 | **28/81 FAIL** | **22/22 frozen GPU toys PASS**; original horizon-based rate/noise schedules remain | Keep selected; continuous learning unsolved |
| **RP1, signal-controlled closing/reopening** | 1200/1200 + 300/300 | **81/81**; stationary 5/5, pre-hold 120/120 | Training-state horizon audit PASS, but image sustained confirmation FAIL and native grid accuracy FAIL; full-state frozen-control audit incomplete | Rejected despite strongest live recovery |
| **PM1 / PM3, relative prior/network mobility** | Both 1200/1200 + 300/300 | **79/81 FAIL** each; stationary and pre-hold pass | Brief late mode loss; scheduled noise remains; transfer and deeper qualification not run | Closest partial recovery leads with intact holds |
| **PB2 / DI2 / P3** | Each 1200/1200 + 300/300 | **77/81 FAIL** each; stationary and pre-hold pass | Four misses; scheduled components remain. P3 has only a four-task diagnostic transfer screen | Partial leads only |
| **TD3, training-distribution discrepancy** | FAIL; no qualifying settling checks | **53/81 FAIL**; stationary 0/5, pre-hold 0/120 | Short image screen PASS; persistent high mobility and no anchor activation | Rejected |
| **EP1, excursion-phase control** | 1200/1200 + 300/300 | **16/81 FAIL**; stationary 5/5, pre-hold 120/120 | Latest recovered result; horizon-based noise remains | Rejected |
| **PD1, projected discrepancy** | NOT_CONVERGED | **0/81 FAIL**; pre-hold 33/120 | Latest recovered raw results; original worker never wrote its ledger | Rejected |

PB1 also reaches 79/81, but fails pre-hold 112/120, so it does not join the stable
PM1/PM3 leads. A3's broader verification finished **4/22 PASS, 18 FAIL**, including
coverage and accuracy failures on all three native layouts.

Sources: [selected K3P declaration](../current-research-base.json),
[RP1 rejection and audit limits](rp1-rejection.md),
[PM results](completed-mobility/attempts/k3p_particle_mobility/result.md),
[PB results](completed-penalty-balance/attempts/k3p_penalty_balance/result.md),
[DI results](completed-data-innovation/attempts/k3p_data_innovation/result.md),
[round 2 / P3 and A3](../continuous-round-2/README.md),
[TD results](completed-training-discrepancy/attempts/k3p_training_discrepancy/result.md),
and [final-batch evidence manifest](stop-inventory/manifest.json).

## Native tests and long-term qualification

Selected K3P's historical native evidence is **grid100 4/4, rotated100 1/1,
staggered100 1/1**. That covers **6 of the required 12 cells**, with six additional
rotated/staggered seeds unmeasured. It is not 4/4 on every layout. The three seed
1234 layout tests are already included in its 22 toys.

RP1's grid100 passes coverage but fails center accuracy: RMS **0.2860 sigma**
against a **0.20** limit. Its 19 transfer tasks have **8 PASS, 1 FAIL, 10 NOT_RUN**.
Rotated100 was interrupted before a verdict. Further qualification stopped after
the measured failures. Its minimum observed grid RMS was about 0.2063 at update
5500, followed by deterioration after a stationary rate reopen. This association
does not prove that suppressing that reopen would pass.

No proposed continuous formulation has completed the conjunction of all 22 toys,
all 12 native cells, full-state matched frozen control, horizon independence, and
delayed/repeated changes through the declared **30,000 uninterrupted updates**.
The [long-term protocol](long-term-stability-protocol.json) remains a declaration,
not a result. K3P is not release-qualified by this search.

## Accounting and unfinished work

Earlier reports record 24 round-1 formulations and 21 round-2 formulations with
no qualified winner. Their missing/interrupted protocols remain missing; these
are not counts of fully qualified experiments.

| Round-3 group | Formulations | Canonical hold/shift protocols completed | Unfinished |
|---|---:|---:|---|
| Initial wave | 17 | 33/34 | EG1 shift not run |
| First replacement wave | 20 | 40/40 | None; all 20 shifts fail |
| TD1–TD3 | 3 | 6/6 | None; all six canonical protocols fail |
| Recovered PD1 + EP1 | 2 | 4/4 | Worker summaries incomplete; raw results retained |
| EP2 partial attempt | 1 | 0/2 | Hold interrupted; shift not run |

Thus **42 round-3 formulations have reported result groups (83/84 canonical
protocols complete), plus EP2's unfinished attempt**. Including EP2 gives 43
benchmark-started formulations, 83/86 protocols complete, one interrupted hold,
and two unrun shifts. PB0 unchanged-parent instrumentation and evaluator work
are excluded. Draft proposals below are not counted as tested formulations.

The [final eight-attempt archive](stop-inventory/manifest.json) preserves terminal
statuses, original ledgers, recovered results, benchmark logs, and draft source
with hashes. No original ledger has been retroactively filled in.

| Last attempt | Inventory disposition |
|---|---|
| Projected discrepancy | PD1 hold and shift completed and failed, recovered from unledgered JSON. PD2 is an untested, incomplete source draft. |
| Excursion phase | EP1 hold+extension PASS, shift FAIL 16/81. EP2 last logged update 2050, holding 650/1200; no final result or extension. |
| Generator restoring reference | Source draft only; no measured candidate result. |
| Game consensus | Research/progress messages only; no implemented or measured candidate. |
| Anchor-force balance | Source draft only; no measured candidate result. |
| Acquire then maintain | No implemented candidate; the optional PM1 image screen was not run. |
| Long-run evaluator repair | Partial repair and short checks, still NOT_READY for qualification. |
| Codex canonical-control audit | Exit 0, but no source changes, ledger, or result report; NOT_DELIVERED. |

The repaired long-run evaluator records 19 passing checks and one earlier error.
Its final **two-update** instrumented/uninstrumented identity check passes. This
compares two arms of the rewritten evaluator; it does not yet demonstrate
identity with the independent canonical driver. It installs the previously
missing fixture/bootstrap/hooks, but complete review and canonical equivalence
remain unfinished. No 9000/30000 candidate run was performed. The earlier error
artifact was overwritten during repair: its ledger row is retained, and the
archive explicitly labels the later PASS snapshot rather than claiming it is the
old error artifact. See [integration review](qualification-harness-review.md).

## Interpretation

The search demonstrates that fast adaptation on the ring is possible, but we
have not preserved K3P's full toy and native accuracy at the same time. RP1 shows
the strongest adaptation signal and the clearest transfer failure; PM1/PM3 show
near-complete recovery with smaller but still binding stability failures. These
are useful research leads, not replacement bases.

Keep K3P selected. If the research is resumed later, the key unresolved question
is how to retain precision while adapting after a change; strict control and
long-run evaluator validation also remain prerequisites for final qualification.
No further search, verification, or gap-filling is being launched during this
pause.
