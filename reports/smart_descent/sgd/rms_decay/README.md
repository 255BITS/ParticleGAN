# Per-tensor RMS memory and decay

Targets remain G=.01/D=.01, selected by the preceding grid. This follow-up varies only target-fraction cosine decay and a causal scalar running RMS denominator. Running RMS uses an EMA of gradient-RMS squared initialized from the first gradient. It stores one scalar per tensor, includes the current gradient, and accumulates no update momentum or coordinate-wise Adam moments. Existing development tasks are reused; these are not fresh transfer results.

Live weights determine all scores; EMA is separate in the raw records. Sustained success requires full mode coverage, HQ≥90%, and at least five passing observations through the final step. The objective strongly rewards each sustained task, so one task can improve the mean while the other regresses. Overall success still requires both tasks.

| Config | RMS beta | Cosine start | Objective ↓ | Ring4 modes / HQ | Grid9 modes / HQ | Sustained tasks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current_cosine06 | current | 0.6 | 12.1699 | 4/4 / 100.00% | 3/9 / 75.78% | 1/2 |
| rms099_constant | 0.99 | none | 21.1546 | 4/4 / 100.00% | 7/9 / 83.42% | 0/2 |
| current_cosine03 | current | 0.3 | 22.6756 | 3/4 / 50.68% | 4/9 / 74.27% | 0/2 |
| current_cosine0 | current | 0.0 | 22.7083 | 4/4 / 58.11% | 3/9 / 90.99% | 0/2 |
| current_constant | current | none | 22.7123 | 4/4 / 74.95% | 6/9 / 67.77% | 0/2 |
| rms099_cosine06 | 0.99 | 0.6 | 22.7560 | 4/4 / 83.03% | 4/9 / 49.73% | 0/2 |
| rms09_cosine06 | 0.9 | 0.6 | 23.0620 | 4/4 / 92.33% | 2/9 / 50.22% | 0/2 |
| rms09_constant | 0.9 | none | 24.6656 | 2/4 / 59.11% | 2/9 / 57.32% | 0/2 |

The constant/current-RMS control exactly reproduces the preceding grid: `{'ring4': {'max_metric_difference': 0, 'same_curve_length': True}, 'grid9': {'max_metric_difference': 0, 'same_curve_length': True}}`.

Download [all scores and protocol](followup.json.gz), [exact source](sources.tar.gz), and [original-byte archive hashes](archive_manifest.json). Every feature/action/metric trace and separate EMA result is retained in `episodes/*.json.gz`.
