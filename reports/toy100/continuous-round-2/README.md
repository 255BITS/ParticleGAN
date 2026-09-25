# Continuous-search round 2: results so far

K3P remains selected; **no new winner qualifies**. A3 verification is complete:
**4/22 PASS, 18 FAIL, 0 ERROR, 0 NOT_RUN**. Its better partial ring recovery lost
most original toy gates. All three natives fail coverage AND accuracy at 7000
updates. The four passes are stripes, anisotropic, overlap and spiral.

| Candidate | Own hold | Extension | Shift pre-hold | Recovery deadline | Own toys |
|---|---|---|---|---|---|
| K3P, retained parent evidence | 1200/1200 | 300/300 | PASS | FAIL 28/81 | 22/22 |
| A3, rejected | 1200/1200 | 300/300 | FAIL 89/120 | FAIL 71/81 | **4/22** |
| P3 floor reopen | 1200/1200 | 300/300 | 120/120 | FAIL 77/81 | NOT_RUN |
| AP3 partial reopen | 1200/1200 | 300/300 | 120/120 | FAIL 72/81 | NOT_RUN |
| P2 generator collapse | 1200/1200 | 300/300 | 120/120 | FAIL 0/81 | NOT_RUN |
| NS3 fixed schedule + shock | 1200/1200 | FAIL 274/300 | FAIL 92/120 | FAIL 0/81 | NOT_RUN |

A3's fixed-budget ring acquisition also fails: HQ .818115 at update1200. Its
later successful convergence-gated hold cannot substitute for that gate. See
[full verification and matrix](attempts/verify_a3_gates/result.md).

AP3's moderate anchor-on reopen reacquires eight modes early, then loses precision
during its fixed 800-update mobility dwell: nine deadline checks fail, worst HQ
.34839. Sustained recovery is delayed780 updates against the400-update deadline.
Its noise remains horizon-dependent. This motivates a new hypothesis, not
promotion or a proof that removing the dwell will solve it. AP1/AP2 only ran
shift: their missing hold protocols remain NOT_RUN. See the
[adversarial-progress report](attempts/k3p_adversarial_progress/result.md).

P3 passes stationary5/5 as well as hold and pre-hold, but misses recovery checks
at2970,3150,3160,3170; deadline minimum HQ .32910 and sustained delay780. It retains
horizon-based noise. Its rate increases briefly, then closes against the much
larger lifetime acquisition-gradient peak. This differs from AP3's forced dwell.
Its unchanged bundle receives a diagnostic sensitive-gate screen before further
search uses that lead. P2's reopening detector instead tracks the new gradient
level too quickly and never raises the rate. See the
[precision report](attempts/k3p_precision_recovery/result.md) and
[frozen source hashes](sources/source-hashes.json). Neither is a verified base.

The [anchor-tracking report](attempts/k3p_anchor_tracking/result.md) adds no winner:
AT3 fails hold after356 good checks and recovery38/81, with stationary0/5 and
pre-hold0/120. AT1/AT2 hold runs were stopped early without a final raw result;
their ledger FAIL labels are **incomplete observations**, not completed canonical
verdicts. P1's stopped hold is likewise incomplete. Missing protocols remain
NOT_RUN and do not inherit another variant's scores.

The user reduced concurrent search to **one Codex plus seven Grok**. The extra
reversible-schedule Codex attempt was stopped after all three candidates failed
both canonical protocols; its remaining prefix audit was interrupted. Completed
results are retained. Other round-2 attempts continue, with freed slots used by
the [rolling launcher](../continuous-round-3/launch.py). Historical launch receipts
retain the actual original3+5 allocation.

[Evidence index](evidence.json) preserves completed attempt ledgers and hashed
compressed raw JSON. Ledger PASS also includes integrity audits; it is not a count
of qualified formulations. Live attempts are not yet archived. Selected K3P source
and its original22/22 evidence are unchanged.
