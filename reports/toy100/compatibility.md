# One-recipe toy compatibility verification

The combined gate in [`benchmarks/toy_suite.py`](../../benchmarks/toy_suite.py)
requires one global optimizer, loss, learning-rate, and noise policy across all
22 toys. The three 100-mode problems must pass their coverage **and** accuracy
gates; each of the 19 canonical transfer hosts must pass its frozen live gate.
Each host keeps its declared architecture, data, batch, prior size, budget,
evaluation schedule, and thresholds. The separate installed-wheel public v3
replay is a control, not a substitute for testing the candidate recipe.

The final current-branch installed-wheel control passes **19/19** after the
optional custom-host noise adapters and exact candidate-schedule bridge were
added. Its raw summary is
`artifacts/toy100-accuracy/compatibility/final-branch-public19/summary.json`;
strict episode regrading independently reports **PASS 19/19**.
Raw `artifacts/` evidence is retained in the local workspace and is not linked
from GitHub. Exact zero-noise parity tests also check the adapters' default
behavior. Candidate runs use seed 0 and one CPU thread, with no seed search or
changes to the historical gates.

| Shared candidate | Frozen live observations | Strict replay | Finding |
| --- | ---: | --- | --- |
| Public v3 core; output noise 0.029; input noise 0.5→0 by 50% of budget | 13/19 (raw `artifacts/toy100-accuracy/compatibility/v3core-noise029-adapted-all19/summary.json`) | Valid FAIL | Six frozen misses, including trajectory identity. |
| LR anneal starts at 40%; immediate output noise; input noise ends by 10% | 18/19 (raw `artifacts/toy100-accuracy/compatibility/v3-end01-anneal04-adapted-all19/summary.json`) | **Invalid common-recipe evidence** | Old custom-host bridge still applied a 60% LR anneal start, despite the declared 40%; observed count is diagnostic only. |
| 40% LR anneal; output noise warms over first 20% of host budget | 16/19 (raw `artifacts/toy100-accuracy/compatibility/v3-end01-anneal04-outwarm02-adapted-all19/summary.json`) | **Invalid common-recipe evidence** | Same custom-host LR schedule mismatch; observed count is diagnostic only. |
| Same 40% LR anneal and 20% output warmup, with the legacy LR bridge corrected | 14/19 (raw `artifacts/toy100-accuracy/compatibility/exact-schedule-warm02-all19/summary.json`) | Valid FAIL | Trajectory passes; unused-token, mode-hold, unequal-width, anisotropic, and four-bar image fail. |
| β₂ 0.999 and 60% LR anneal; 20% output warmup; input noise ends by 10% | 15/19 (raw `artifacts/toy100-accuracy/compatibility/exact-beta999-a6-all19/summary.json`) | Valid FAIL | Trajectory, unequal-width, and anisotropic pass; mode-hold, overlap, stripes, and blobs fail. |
| Same β₂ 0.999 core; prior LR multiplier 3; input noise ends by 20% | 16/19 (raw `artifacts/toy100-accuracy/compatibility/exact-beta999-priorlr3-end02-all19/summary.json`) | Valid FAIL | Mode-hold and all four images pass; trajectory sustained identity, unequal-mass rare covariance, and overlap terminal stability fail. |
| Same shared core, with one learnable G-owned output-noise scale initialized at 0.029 | 13/19 (raw `artifacts/toy100-accuracy/learnable-shared/candidate19/summary.json`) | Valid FAIL | Trajectory and unequal-mass pass, but residual-student, mode-hold, unequal-width, anisotropic, overlap, and four-bar image fail. The corresponding native grid run also collapses to 4/100 modes (raw `artifacts/toy100-accuracy/learnable-shared/grid/grid100/summary.json`). |

The corrected 20%-warmup result demonstrates why an isolated trajectory repair
is not enough for the common-recipe claim. Bounded 5% and 10% warmup screens
on four affected hosts were run before the LR-bridge correction and are
diagnostic only. The complete [search ledger](shared-recipe-search.md) marks
every historical subset, source/config digest, and LR schedule mismatch.
With β₂ 0.99 and a 60% LR anneal, the ten native hosts passed 8/10; unequal
mass and four-bar images failed. Extending input-noise decay from 10% to 20%
repaired those two in a six-host subset, while mode-hold remained at 7/8 modes.
These subsets do not establish a full 19-case result.
The learnable-scale replay is a complete 19-host result with all G ownership
and actual-noise receipts independently regraded; its varying final scale
(0.0054–0.0411 across hosts) does not resolve the one-recipe incompatibility.
A bounded variant initialized the same learnable scalar at 0.2, started output
noise immediately, and set input noise to zero on every host. It passes 5/8
selected bottlenecks, including unequal-mass, unequal-width, and anisotropic,
but fails mode-hold, overlap, and four-bar images (raw
`artifacts/toy100-accuracy/learnable-broad/bottleneck8/summary.json`). Its
episodes pass strict evidence checks, but this subset is **incomplete** for
the canonical 19 and cannot establish a common 22-case recipe. The mode-hold
terminal generator support misses all eight centers by at least 0.318, above
its 0.21 HQ radius; the final learned noise scale alone does not explain that
failure.
A separate fixed-noise timing interpolation uses the public-v3 β₂=0.99,
prior multiplier 2, and input-noise decay ending at 15% of each host's budget;
the other shared fields stay at the declared candidate values. It passes
6/8 selected bottlenecks, including trajectory, mode-hold, unequal-width,
anisotropic, overlap, and four-bar images, but fails unequal-mass (only four
passing terminal checks) and stripes (only two). Raw summary:
`artifacts/toy100-accuracy/compatibility/exact-beta99-end015-bottleneck8/summary.json`.
This is a strict-valid **subset diagnostic**, not a complete 19-host replay.
Two further fixed-noise startup controls use the same β₂=0.99/public-v3 core
and eight bottlenecks. Output-only noise with 20% warmup passes 5/8 but misses
mode-hold, unequal-width, and stripes (raw
`artifacts/toy100-accuracy/compatibility/output_only_warm02-bottleneck8/summary.json`).
Coupled input noise ending at 10% with a shorter 5% output warmup passes 4/8
but misses trajectory, unequal-mass, unequal-width, and overlap (raw
`artifacts/toy100-accuracy/compatibility/coupled_short_warmup-bottleneck8/summary.json`).
Both are strict-valid **subsets** and were not expanded to all 19 hosts.
A [frozen 12-row noise screen](noise-grid-v1.md) then tested a bounded
output-scale × input-decay table plus three controls on the same eight hosts.
The closest row uses β₂=0.99/prior multiplier 2, output σ=0.0275 warmed over
20%, and input σ=0.5 ending at 12.5% of each host budget. It passes 7/8;
overlap has four passing terminal checks instead of the required five. All
rows are strict-valid but **subset-only** evidence, so none was promoted to
a full 19-host replay.
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
the saved source archive, copied candidate config, and noise implementation;
recomputes frozen live verdicts, checks all 24 observations and complete update
traces, checks the noise schedule and receipts, and rejects mixed recipe fields
or noise settings. The `compatibility.json` and `compatibility.md` files under
the output directory contain the per-case result and the 22-case status.
