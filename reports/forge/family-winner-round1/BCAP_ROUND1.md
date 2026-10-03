# BCap round 1 result

All eight configurations stopped on a required failure. No full-view winner qualified, and no default changed. The [frozen search report](../configuration-search/bcap-family-defaults-round1-v1.json) selects coefficient 0.5 / critic multiplier 2 / prior multiplier 4 as **best observed** by the declared pass-count/hash rule. It passed 8 of the 24 required questions: all three smoke questions and the first five quality questions. The other ring survivor passed the same eight questions; the hash tie-break selects one, rather than claiming better convergence speed or quality on untested tasks.

The exhaustive [eight-configuration readout](bcap-round1-readout.json) binds all 30 independently graded attempts, their original request/result/terminal hashes, exact source/runtime/protocol and completed numeric verdicts. Execution cost was 284.366656 seconds against the declared 352,800-second family reservation. No vector, image, native or endurance question was reached. Unreached tasks remain `UNKNOWN` / `NOT_REACHED`, rather than failed measurements.

| Coefficient | Critic multiplier | Prior multiplier | Required passes | First required failure |
|---:|---:|---:|---:|---|
| 0.5 | 2 | 4 | 8/24 | Ring: 7/8 modes, HQ 1, no passing observation in 24 |
| 1 | 2 | 1 | 8/24 | Ring: 7/8 modes, HQ 0.999755859375, no passing observation in 24 |
| 0.5 | 2 | 1 | 3/24 | Trajectory: MSE 0.1782831103 > 0.02 |
| 1 | 2 | 4 | 3/24 | Trajectory: MSE 0.2120648623 > 0.02 |
| 0.5 | 0.5 | 1 | 0/24 | Two pole: mean absolute position 0.1418880820 < 0.3 |
| 0.5 | 0.5 | 4 | 0/24 | Two pole: mean absolute position 0.1418880820 < 0.3 |
| 1 | 0.5 | 1 | 0/24 | Two pole: mean absolute position 0.1418880820 < 0.3 |
| 1 | 0.5 | 4 | 0/24 | Two pole: mean absolute position 0.1418880820 < 0.3 |

Each first failed question completed its declared optimizer budget and all 24 observations, with zero passing terminal suffix. The two-pole failures satisfy the critic gradient bound (0.3138940632 ≤ 1), but the sample particles do not move far enough. The trajectory failures retain excessive conditional identity MSE. The ring failures retain seven occupied modes with high sample fidelity, leaving one mode uncovered. These are evidence-backed numerical failure signatures. The underlying optimizer mechanisms remain unresolved; the grid does not justify attributing every failure to one knob.

The [zero-update capacity states](bcap-representation.json) show that the corresponding targets can be represented within their numeric tolerances using compatible public hosts and sampling laws. They do not supply trained passes or prove reachability from the ordinary initialization. Seven modes versus the historical five modes is an observed occupancy difference across different source/runtime/configuration cohorts, not an isolated causal improvement claim. First-acquisition speed is unavailable: the frozen adapters did not supply comparable per-observation timing, and terminal suffix step labels are not wall-clock speed.

These outcomes remain a provisional configuration screen. Full required gates, accepted historical calibration and independent robustness would all be needed before default adoption. Any further substantive round should declare changed public settings and retain the same laws, gates, budgets and every failed configuration's evidence.
