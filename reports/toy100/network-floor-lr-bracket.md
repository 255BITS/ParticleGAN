# Residual-student global LR bracket

The network-floor `.01`, κ=1.0 common candidate passed the predeclared nine-host bottleneck screen but failed `residual_student` in the full 19-host replay. This bounded screen changed only the global LR and recipe name from that candidate. It used frozen source commit `a3be165e2ab47290d35ed98426be77d148f04320`, seed 0, the host's 400-step budget, and the original 24 evaluation checkpoints. The four values and run order were [declared before training](../../artifacts/toy100-accuracy/compatibility/network-floor-lr-bracket-v1/plan.json).

| Global LR | Final identity MSE ↓ | Final success ↑ | Final wrong-pad ↓ | Passing checks / 24 | Sustained suffix | Verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 0.00420 | 0.05114 | 0.5000 | 0.5000 | 0 | 0 | FAIL |
| 0.00430 | 0.05116 | 0.5000 | 0.5000 | 0 | 0 | FAIL |
| 0.00415 | **0.04624** | **0.6667** | **0.3333** | 0 | 0 | FAIL |
| 0.00435 | 0.06509 | 0.5000 | 0.5000 | 0 | 0 | FAIL |

The frozen gate requires identity MSE ≤ 0.02, success rate 1.0, wrong-pad rate 0, and five sustained passing checks. None of the four values repaired the host, so none was promoted to the expanded ten-host screen, full 19-host replay, or native 100-mode tests. The best result, LR 0.00415, remains more than twice the MSE limit and misses the two discrete outcomes. This local LR bracket supplies no evidence for a common 22-task recipe.

Complete configs, logs, source archives, compressed episodes, protocol receipts, and summaries are retained in the [artifact directory](../../artifacts/toy100-accuracy/compatibility/network-floor-lr-bracket-v1). All 38 copied files match their RAM originals by SHA-256. Independent `_episode_rows` regrading of each copied run validated the archived config/source and action receipts, then reproduced FAIL with 24 observations and zero passing suffix. The suite summary labels these one-host subsets `INCOMPLETE` for the full mechanism; the evaluated host's strict verdict is FAIL in every row.
