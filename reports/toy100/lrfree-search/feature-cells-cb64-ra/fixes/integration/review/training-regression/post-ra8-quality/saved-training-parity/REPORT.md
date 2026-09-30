# RA8 versus RA7: serialized training neutrality

**All ten saved endpoints have exact original training-state parity.** The
CPU comparator found zero unexpected differences at updates
0,100,250,500,750,1000,1250,1500,1750,2000. It compared exact tensor types,
shapes and bytes, scalar types/values, dictionary keys and sequence structure,
with no tolerance or rounding.

The helper and its complete allowlist were frozen at 07:51:13.917 UTC before
loading any RA8 numerical checkpoint. The first comparison began afterward.
All checkpoint reads required the post-save training log event and unchanged
before/after file hashes. All123 pinned source/READY/config/lane identities
remain exact. No CUDA, model forward, model construction, evaluator draw,
optimizer step, training update, RNG restoration/consumption or new seed was
used. The watcher completed and exited.

| Saved update | Exact legacy state | Unexpected differences | Saved joint count | Eligibility lease live |
| ---: | --- | ---: | ---: | --- |
| 0 | yes | 0 | 0 | false |
| 100 | yes | 0 | 95 | false |
| 250 | yes | 0 | 259 | false |
| 500 | yes | 0 | 521 | false |
| 750 | yes | 0 | 852 | false |
| 1000 | yes | 0 | 638 | false |
| 1250 | yes | 0 | 943 | false |
| 1500 | yes | 0 | 881 | false |
| 1750 | yes | 0 | 963 | false |
| 2000 | yes | 0 | 977 | true |

The fixed requirement is973/1024. These are actual saved GPU reaction stamps,
which differ from prior descriptive CPU-refit counts. No historical GPU chart
was reconstructed, no policy threshold selected and no emissions generated.

## Compared state and precise exclusions

Every original serialized trainer component is exact: live FAST G/D/prior,
EMA G/prior, parameter/buffer and requires-grad state, both optimizers and
their moments/history, controller, all stationarity testers and row evidence,
initial/current rates, completed steps, output-noise state, global CPU/CUDA
RNG bytes and every private stream. Legacy backend FIFO, graph, row state,
action/evidence/count/copy/birth counters and reaction metadata are exact.
Saved data positions also match. RA8 `state_dict` records FAST weights even
when its serving parameters are temporarily swapped to EMA.

The only allowed training-state changes are backend6->7, its two declared
paired-average settings, its typed scalar stamp and corresponding last stamp,
and the new last forward-row diagnostic. The only excluded original diagnostic
leaves are reaction `eval_seconds`, last work `distance_cells` and
`projection_products`, and the cumulative distance/projection performance
counters. All other work/action fields are compared. New metadata is checked
against its declared policy, requirement, step, snapshot and copied stamp.

Outer checkpoint records contain serving metrics, wall/evaluation timing,
diagnostic logs and new-variant config provenance. They are outside serialized
training state; their saved step and data position are checked. Original
CUDA evaluator outputs remain root-owned quality evidence.

## Scope

This establishes exact parity at every prescribed saved endpoint. It does not
observe every intermediate update or certify future trajectories, geometry,
stationarity or learned quality. The final live lease is the deliberately
bounded stale anti-blur view, not an equivalence certificate. Full toy and
canonical grid gates remain separate root-owned requirements.

Evidence: `SOURCE-FROZEN.json`, `accepted-attempt1/summary.json`, the ten
exclusive comparison receipts and per-step seals, watcher log and final audit
receipt. No failed helper attempt occurred; the pre-read helper control and
its original log are retained unchanged.
