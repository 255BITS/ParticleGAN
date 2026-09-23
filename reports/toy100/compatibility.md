# One-recipe toy compatibility verification

The combined gate in [`benchmarks/toy_suite.py`](../../benchmarks/toy_suite.py)
requires one global optimizer, loss, learning-rate, and noise policy across all
22 toys. The three 100-mode problems must pass their coverage **and** accuracy
gates; each of the 19 canonical transfer hosts must pass its frozen live gate.
Each host keeps its declared architecture, data, batch, prior size, budget,
evaluation schedule, and thresholds. The separate installed-wheel public v3
replay is a control, not a substitute for testing the candidate recipe.

The control passes **19/19** on this branch after the optional custom-host
noise adapters were added: summary (raw `artifacts/toy100-accuracy/compatibility/current-branch-public19/summary.json`).
Raw `artifacts/` evidence is retained in the local workspace and is not linked from GitHub. Its default-off adapter behavior is also checked by exact zero-noise parity
tests. Candidate runs use seed 0 and one CPU thread, with no seed search or
changes to the historical gates.

| Shared candidate | Frozen live observations | Strict replay | Finding |
| --- | ---: | --- | --- |
| Public v3 core; output noise 0.029; input noise 0.5→0 by 50% of budget | 13/19 (raw `artifacts/toy100-accuracy/compatibility/v3core-noise029-adapted-all19/summary.json`) | Valid FAIL | Six frozen misses, including trajectory identity. |
| LR anneal starts at 40%; immediate output noise; input noise ends by 10% | 18/19 (raw `artifacts/toy100-accuracy/compatibility/v3-end01-anneal04-adapted-all19/summary.json`) | **Invalid common-recipe evidence** | Old custom-host bridge still applied a 60% LR anneal start, despite the declared 40%; observed count is diagnostic only. |
| 40% LR anneal; output noise warms over first 20% of host budget | 16/19 (raw `artifacts/toy100-accuracy/compatibility/v3-end01-anneal04-outwarm02-adapted-all19/summary.json`) | **Invalid common-recipe evidence** | Same custom-host LR schedule mismatch; observed count is diagnostic only. |
| Same 40% LR anneal and 20% output warmup, with the legacy LR bridge corrected | 14/19 (raw `artifacts/toy100-accuracy/compatibility/exact-schedule-warm02-all19/summary.json`) | Valid FAIL | Trajectory passes; unused-token, mode-hold, unequal-width, anisotropic, and four-bar image fail. |
| β₂ 0.999 and 60% LR anneal; 20% output warmup; input noise ends by 10% | 15/19 (raw `artifacts/toy100-accuracy/compatibility/exact-beta999-a6-all19/summary.json`) | Valid FAIL | Trajectory, unequal-width, and anisotropic pass; mode-hold, overlap, stripes, and blobs fail. |

The corrected 20%-warmup result demonstrates why an isolated trajectory repair
is not enough for the common-recipe claim. Bounded 5% and 10% warmup screens
on four affected hosts were run before the LR-bridge correction and are
diagnostic only. The complete [search ledger](shared-recipe-search.md) marks
every historical subset, source/config digest, and LR schedule mismatch.
With β₂ 0.99 and a 60% LR anneal, the ten native hosts passed 8/10; unequal
mass and four-bar images failed. Extending input-noise decay from 10% to 20%
repaired those two in a six-host subset, while mode-hold remained at 7/8 modes.
These subsets do not establish a full 19-case result.
The combined gate remains **incomplete** until one
identical recipe passes both the three 100-mode problems and all 19 transfer
cases. Every listed full candidate run includes actual G-output and
D-input-noise receipts for all nine custom hosts and the ten standard trainer
hosts. Parameter-based legacy readouts retain their frozen metric formulas;
generated-sample readouts include the declared output noise.

For a fresh complete run, use:

```bash
python -u -m benchmarks.toy_suite run \
  --config PATH_TO_SHARED_CONFIG.json \
  --output PATH_TO_NEW_OUTPUT
python -m benchmarks.toy_suite regrade --output PATH_TO_NEW_OUTPUT
```

The independent regrader reads and hashes every compressed transfer episode,
recomputes frozen live verdicts, checks all 24 observations and complete update
traces, checks the noise schedule and receipts, and rejects mixed recipe fields
or noise settings. The `compatibility.json` and `compatibility.md` files under
the output directory contain the per-case result and the 22-case status.
