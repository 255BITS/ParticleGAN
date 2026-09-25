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
| Shared-sample predictor/corrector | Codex | Restore temporary state and commit one update per role |

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

New results are pending. No candidate has been promoted.
