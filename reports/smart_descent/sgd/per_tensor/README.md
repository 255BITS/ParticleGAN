# Per-tensor relative-step grid

This is per-tensor adaptive gradient descent, not common-LR SGD or Adam. Each step is `−alpha_tensor × raw_gradient`, with no update momentum. The rule is `alpha = clip(target_role × max(parameter_RMS,.01) / max(gradient_RMS,1e-12), base_LR × 1e-4, base_LR × 100)`. Base rates are G=.25/D=.005. Targets are swept independently over .001/.003/.01/.03. The same ring4/grid9 development tasks, seed 0, and 1,200 updates are used throughout.

Live weights determine all scores; EMA is separate in the raw records. Sustained success requires full mode coverage, HQ≥90%, and at least five passing observations through the final step. The objective strongly rewards each sustained task, so one task can improve the mean while the other regresses. Overall success still requires both tasks.

| Config | G target | D target | Objective ↓ | Ring4 modes / HQ | Grid9 modes / HQ | Sustained tasks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| relative_10 | 0.01 | 0.01 | 22.7123 | 4/4 / 74.95% | 6/9 / 67.77% | 0/2 |
| relative_06 | 0.003 | 0.01 | 23.6309 | 2/4 / 75.90% | 2/9 / 100.00% | 0/2 |
| relative_11 | 0.01 | 0.03 | 24.2080 | 2/4 / 48.97% | 3/9 / 83.91% | 0/2 |
| relative_07 | 0.003 | 0.03 | 24.4063 | 1/4 / 100.00% | 1/9 / 90.99% | 0/2 |
| relative_04 | 0.003 | 0.001 | 24.5085 | 1/4 / 83.06% | 2/9 / 84.01% | 0/2 |
| relative_09 | 0.01 | 0.003 | 24.7614 | 2/4 / 67.04% | 4/9 / 40.41% | 0/2 |
| relative_03 | 0.001 | 0.03 | 25.2619 | 1/4 / 58.37% | 1/9 / 100.00% | 0/2 |
| relative_02 | 0.001 | 0.01 | 25.2889 | 1/4 / 58.37% | 1/9 / 100.00% | 0/2 |
| relative_05 | 0.003 | 0.003 | 25.6233 | 1/4 / 91.58% | 2/9 / 60.13% | 0/2 |
| relative_01 | 0.001 | 0.003 | 26.2306 | 1/4 / 26.12% | 2/9 / 92.16% | 0/2 |
| relative_00 | 0.001 | 0.001 | 26.2336 | 1/4 / 92.33% | 1/9 / 32.64% | 0/2 |
| relative_14 | 0.03 | 0.01 | 26.4365 | 1/4 / 8.42% | 4/9 / 58.47% | 0/2 |
| relative_15 | 0.03 | 0.03 | 26.8568 | 1/4 / 9.01% | 3/9 / 49.02% | 0/2 |
| relative_08 | 0.01 | 0.001 | 29.7352 | 0/4 / 0.00% | 1/9 / 17.14% | 0/2 |
| relative_12 | 0.03 | 0.001 | 29.7370 | 1/4 / 8.42% | 1/9 / 25.76% | 0/2 |
| relative_13 | 0.03 | 0.003 | 61.3987 | 1/4 / 16.16% | 3/9 / 49.51% | 0/2 |

Download [all scores and protocol](sweep.json.gz), [exact source](sources.tar.gz), and [original-byte archive hashes](archive_manifest.json). Every feature/action/metric trace and separate EMA result is retained in `episodes/*.json.gz`.
