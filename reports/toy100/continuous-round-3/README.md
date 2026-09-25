# Rolling continuous-learning search

Current policy: **at most one Codex and seven Grok attempts**, one benchmark worker
each, at most four workers per GPU. The launcher checks registered live processes
before reserving new capacity. Earlier round receipts preserve their original
three-Codex/five-Grok allocation. Both Codex and Grok over-cap launches have been
checked and rejected.

K3P stays selected. [Round2](../continuous-round-2/README.md) finished21 new
proposals with no winner. P3 is the best new partial lead: hold1200/1200,
extension300/300, stationary5/5, pre-hold120/120 and failed recovery77/81.
Its remaining horizon-based noise and four deadline misses are binding failures.

| Role | Engine | Work |
|---|---|---|
| Unchanged P3 verification | Grok | Sensitive4 first; stop on failure, otherwise own remaining18 |
| Local curvature | Grok | Reversible secant-based step control |
| Negative momentum | Grok | Recursive memory of actual game displacement |
| Responsive precision | Grok | Signal-based closing after an adaptive rate increase |
| Reference response | Grok | Critic reference tracking that preserves acquisition |
| Signal and noise | Grok | Remove horizon-dependent noise without losing acquisition |
| Joint trust | Grok | Bound coupled game displacement from training signals |
| RP1 independent audit (original predictor/corrector slot) | Codex | Verify horizon independence, source and adapter behavior |

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

**RP1 is a provisional lead, not promoted.** Its own hold passes1200/1200
(minimum HQ .96850586), followed by300/300 extension checks (minimum HQ .96801758).
The live shift passes stationary5/5, pre-hold120/120, and deadline81/81
(minimum deadline HQ .91381836). The frozen driver status is UNCONFIRMED pending
matched frozen control. [Exact source and first raw artifacts](sources/rp1-provisional-manifest.json)
are preserved before further verification.

RP1 retains K3P's acquisition penalty, then holds the gradient cap/EMA anchor on
while rates can increase again. Critic gradient RMS relative to a decaying peak
controls that increase and later decay; it removes AP3's forced800-update high-rate
dwell and peak reset. Input/output noise warmups last fixed120/240 updates and
do not require knowing the final training duration. That formula description
still requires a substantial paired horizon audit and actual installation in
all adapters. See [exact declaration](sources/rp1_signal_close/DECLARATION.md).

The Grok lane owns full matched-control evidence and toy gates. The single Codex
slot was redirected to independent source, adapter and horizon-prefix auditing;
its original experiment remains separate. Missing provenance cannot be treated
as matching because both records omit a field. Observer/adapter execution errors
are distinct from candidate gate failures and must be repaired without changing
the learner or weakening a check.

Only after canonical gates qualify does the separately frozen
[delayed/repeated-change protocol](stress-protocol.json) run. No indefinite-learning
claim follows from one successful target shift. All22 toys, horizon equivalence,
matched control and stress results are still pending for RP1.

The user clarified the final native matrix: **all three layouts each4/4 seeds**,
grid100/rotated100/staggered100 at1234–1237. This is12 full7000-update native runs,
each requiring coverage AND accuracy; it adds9 runs beyond the22-toy matrix.
This fixed winner qualification is explicitly authorized, with no seed search.
The separately declared [30000-update long-term continuation](long-term-stability-protocol.json)
preserves the9000 stress windows and adds another change at27000. After the same
formulation clears every requirement, the supervisor will stop remaining searches
and promote it in PR155. Until then the search and verification continue.
