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
