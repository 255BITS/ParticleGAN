K3P stays the selected base. Three reference-gap proposals each passed their own ring hold and 300-update extension, and none recovered in time. Nothing is promoted.

| Candidate | Hold | Extension | Pre-shift | Deadline | Delay |
|---|---|---|---|---|---|
| K3P parent (not rerun) | 1200/1200 | 300/300 | published pass | FAIL 28/81 | 1130 |
| px3 calm innovation | 1200/1200 | 300/300 | 5/5 and 120/120 | **FAIL 71/81** | 850, stable at 3250 |
| px1 half-peak cutoff | 1200/1200 | 300/300 | 5/5 and 120/120 | FAIL 16/81 | none |
| px2 proportional gap | 1200/1200 | 300/300 | 5/5 and 120/120 | FAIL 0/81 | none |

px3 is the strongest count and still fails. All 81 deadline checks have 8 modes. The 10 misses are HQ under 0.90, at steps 2800–2870 and 3230–3240. The full 0.2 cap was on at step 2449, then the calmer floored the rate by step 2499 while the smoothed gap was still rising and only 7 modes were back. It did not reopen. Final live state is 8 modes, HQ 0.977, with both optimizers still updating on the floors.

px1 floored the rate once the gap fell to half its impulse peak, which was still about 25× the quiet baseline. px2 kept a moderate prox/peak rate and never stabilized. Horizon-1200 noise is still the driver schedule on every candidate. Frozen control, transfer gates, native coverage, and the later stress tests are NOT_RUN.

The full write-up, replay commands, and hashes are in `result.md` next to `tests.jsonl`. The next rule should keep px3’s full cap and floor, and arm that floor only after the smoother has peaked and its contraction has calmed.
