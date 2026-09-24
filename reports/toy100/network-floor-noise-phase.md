# Residual-student noise timing screen

The network-floor `.01`, κ=1.0 candidate passed nine frozen bottleneck hosts but failed `residual_student` in its full 19-host replay. This eight-row screen changed exactly one noise timing field and the recipe name per row from that candidate. The plan, configs, source manifest, and source archive at the local ignored path `artifacts/toy100-accuracy/compatibility/network-floor-noise-phase-v1` were written before training. Source commit was `a3be165e2ab47290d35ed98426be77d148f04320`; every row used seed 0, the frozen 400-step budget, and 24 checkpoints.

| Changed field | Value | Identity MSE ↓ | Success ↑ | Wrong-pad ↓ | Passing checks / 24 | Sustained suffix | Verdict |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Output warmup | 0.10 | 0.04827 | 0.5833 | 0.4167 | 0 | 0 | FAIL |
| Output warmup | **0.15** | **0.000925** | **1.0000** | **0.0000** | **20** | **20** | **PASS** |
| Output warmup | **0.25** | **0.000906** | **1.0000** | **0.0000** | **18** | **18** | **PASS** |
| Output warmup | 0.30 | 0.05141 | 0.5000 | 0.5000 | 0 | 0 | FAIL |
| Input anneal end | 0.050 | 0.05110 | 0.5000 | 0.5000 | 0 | 0 | FAIL |
| Input anneal end | 0.075 | 0.04914 | 0.5833 | 0.4167 | 0 | 0 | FAIL |
| Input anneal end | 0.125 | 0.04613 | 0.6667 | 0.3333 | 0 | 0 | FAIL |
| Input anneal end | 0.150 | 0.05115 | 0.5000 | 0.5000 | 0 | 0 | FAIL |

The frozen host requires identity MSE ≤ 0.02, success rate 1.0, wrong-pad rate 0, and five sustained passing checks. The 0.15 and 0.25 output-warmup rows were handed to the independent nine-host screen in predeclared order. Neither is a common 22-task result until the remaining hosts, a fresh full 19-host replay, and all three native 100-mode problems pass with one exact recipe.

All eight configs, logs, source archives, compressed episodes, and protocol receipts are retained in the linked artifact directory. All 76 copied files match the RAM originals by SHA-256. Independent `_episode_rows` regrading of each copied run validated its source, config, frozen specification, noise and optimizer actions, and strict verdict. The one-host subset suite summaries say `INCOMPLETE` for full-mechanism scope; the table reports each evaluated host's strict verdict.
