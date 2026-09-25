# Rolling continuous-learning search

Current policy: **at most one Codex and seven Grok attempts**, one benchmark worker
each, at most four workers per GPU. The launcher checks registered live processes
before reserving new capacity. Earlier round receipts preserve their original
three-Codex/five-Grok allocation. Both Codex and Grok over-cap launches have been
checked and rejected.

K3P stays selected. [Round2](../continuous-round-2/README.md) finished21 new
proposals with no winner. P3 was the strongest round-2 partial lead: hold1200/1200,
extension300/300, stationary5/5, pre-hold120/120 and failed recovery77/81.
Its remaining horizon-based noise and four deadline misses are binding failures.

The initial wave of round 3 has finished **17 formulations, no qualified winner**.
Of 34 expected canonical hold/shift protocols, 33 completed; EG1's shift was not
run after its slot moved to independent RP1 auditing. Negative momentum reached
its agent time cap after saving all six failed canonical runs. Its interrupted
report-writing is separate from those completed measurements.

| Initial lane | Proposals | Hold + extension passes | Complete live shift passes | Outcome |
|---|---:|---:|---:|---|
| Responsive precision | 1 | 1 | 1 | RP1 rejected by image stability and native accuracy |
| Reference response | 3 | 3 | 0 | Best deadline count 52/81; no transfer qualification |
| Local curvature | 3 | 0 | 0 | Acquisition/hold failures |
| Negative momentum | 3 | 0 | 0 | Acquisition/hold failures |
| Signal and noise | 3 | 0 | 0 | Acquisition/hold failures |
| Joint trust | 3 | 0 | 0 | Acquisition/hold failures |
| Predictor/corrector | 1 | 0 | Not run | Failed hold; slot redirected to auditing |

[Exact accounting](first-wave-summary.json) separates raw UNCONFIRMED live shift
from matched-control qualification. No row borrows parent passes. Later search
launches below are ongoing and excluded from this completed-wave count.

| New search | Engine | Question |
|---|---|---|
| Acquisition across tasks | Codex | Which training signal preserves both short-image convergence and native precision? |
| Reference-gap release | Grok | Can contraction of the reference gap end adaptation without premature closing? |
| Critic confidence | Grok | Does minibatch uncertainty distinguish learning from stochastic oscillation? |
| Prior mobility | Grok | Can relative prior/network motion retain precision without preventing adaptation? |
| Acquisition noise | Grok | Can achieved optimizer motion govern noise without a final training duration? |
| Data innovation | Grok | Can minibatch changes reopen learning without stationary false alarms? |
| Penalty balance | Grok | Does separating adversarial and regularizer forces improve control? |
| Qualification harness | Grok | Prepare exact long-run evaluation; no candidate qualification runs |

Earlier local-curvature, negative-momentum, responsive-precision, reference,
signal/noise, joint-trust and predictor/corrector attempts retain their own
records. Live lanes are listed by the monitor; this table describes new work.

Every new proposal earns both canonical protocols. A prospective live winner
then gets its matched frozen control and all22 toys. Only full evidence can
authorize promotion; paper hypotheses and partial scores are not qualifications.
Delayed/repeated changes and horizon invariance remain required before claiming
continuous learning. See [search brief](SEARCH.md) and [research notes](../continuous-search-tools/research-notes.md).

Launches are staggered as slots free up; [receipts](receipts/) record exact base
commits, commands, engines, GPUs and PIDs. The initial [launch receipt](launch-receipt.json)
contains only the first three replacements. Watch all live batches with:

```sh
python3 /ml2/hypergan/monitor-gan.py --once
```

**RP1 passed the live ring requirements, then failed transfer verification.**
Own hold was 1200/1200, extension 300/300, stationary 5/5, pre-hold 120/120 and
recovery 81/81. However, `img_intensity2` failed sustained confirmation (3/24
passing observations), and native grid100 passed coverage but failed center
accuracy. Eight transfer gates pass; ten remain unrun. RP1 is rejected and further
native, seed and stress qualification is stopped. K3P stays selected.

See the [failure diagnosis and evidence](rp1-rejection.md). The substantial
horizon-prefix audit passes for training state; its separate raw whole-capture
comparison fails on evaluation counters. The tested observer adapter preserves
updates while fixing checks that previously compared rates from different steps.
No audit pass overrides a quality failure.

[Reference-response results](completed-reference/attempts/k3p_reference_response/result.md)
add three passing own holds/extensions, but every shift fails (29/81, 52/81,
0/81). These are diagnostic leads only. Fresh searches focus on acquisition and
precision across tasks, uncertainty in the closing signal, and reference-gap
release. Every new formulation must earn its own scores.

The user clarified the final native matrix: **all three layouts each4/4 seeds**,
grid100/rotated100/staggered100 at1234–1237. This is12 full7000-update native runs,
each requiring coverage AND accuracy; it adds9 runs beyond the22-toy matrix.
This fixed winner qualification is explicitly authorized, with no seed search.
The separately declared [30000-update long-term continuation](long-term-stability-protocol.json)
preserves the9000 stress windows and adds another change at27000. After the same
formulation clears every requirement, the supervisor will stop remaining searches
and promote it in PR155. Until then the search and verification continue.

The [fixed native qualification wrapper](../continuous-search-tools/NATIVE_QUALIFICATION.md)
is prepared for surviving candidates. Its twelve configuration checks pass with
zero training updates; it pins candidate source and preserves full7000 budgets.
No additional RP1 seed qualification was run.

The first replacement Codex attempt also finished: [AC1–AC3 results](completed-acquisition/attempts/k3p_transfer_acquisition/result.md).
All six canonical protocols fail. AC1/AC2 pass the short image screen, but AC1
never acquires eight ring modes and AC2 loses a mode after 187 good hold checks.
AC3 holds for 1027 checks before failure and also fails the image screen. These
are three additional rejected formulations, excluded from the initial17 count.
The next Codex direction uses ordinary real/generated training discrepancy,
with a sampling-noise reference, as a hypothesis for separating model error from
critic fluctuations. It remains K3P-derived research, not a new selected base.

The first long-run harness preparation passed its synthetic checks but failed
[root integration review](qualification-harness-review.md): it omitted canonical
candidate hooks and initialization. It is marked NOT_READY and assigned for
repair. No candidate long-run score is claimed from those harness checks.
