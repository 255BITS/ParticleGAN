# Stress/dynamics solvability search

Actual GAN training under unchanged numerical gates. Live weights determine final and sustained passes; EMA is separate. Every episode has 24 fixed observations and needs a passing final suffix of at least 5. Cadence has already been seen and is development data.

The short card screen used the fast-critic case. Prior LR 30 and five resource/tuning finalists were then evaluated across all six ranking stresses and the seen cadence. Untested cells are shown explicitly.

| Card | Resource class | stress_fast_critic | stress_slow_critic | stress_small_batch | stress_large_critic | stress_long_horizon | stress_r1_r2 | reserved_alternating_critic_updates | Sustained / evaluated | Final / evaluated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| budget3 | more updates | PASS (7/24) | FAIL (0/24) | PASS (11/24) | PASS (8/24) | PASS (6/24) | FAIL (0/24) | FAIL (0/24) | 4/7 | 4/7 |
| stockish_extended | more updates + changed capacity | PASS (19/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | PASS (16/24) | FAIL (0/24) | 2/7 | 2/7 |
| budget2 | more updates | PASS (10/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | Final only (4/24) | FAIL (0/24) | FAIL (0/24) | 1/7 | 2/7 |
| prior_lr_30 | same architecture/data/update budget | PASS (6/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | 1/7 | 1/7 |
| slow_g_every2_budget5 | changed update cadence + more outer steps (D6000/G3000) | Not run | PASS (11/24) | Not run | Not run | Not run | Not run | Not run | 1/1 | 1/1 |
| slow_g_every2_budget3 | changed update cadence + more outer steps (D3600/G1800) | Not run | Final only (2/24) | Not run | Not run | Not run | Not run | Not run | 0/1 | 1/1 |
| baseline | same architecture/data/update budget | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| beta2_0p9 | same architecture/data/update budget | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| beta2_0p99 | same architecture/data/update budget | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | 0/7 | 0/7 |
| cap10 | same architecture/data/update budget | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | FAIL (0/24) | 0/7 | 0/7 |
| cap1_k1 | same architecture/data/update budget | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| cap2_k1 | same architecture/data/update budget | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| cap5_k1p25 | same architecture/data/update budget | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| density4096_slow | more particles + larger batch + more updates | Not run | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| fourier3 | changed capacity | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| fourier3_particles1024 | changed capacity | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| fourier4 | changed capacity | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| lr1p5 | same architecture/data/update budget | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| particles1024 | changed capacity | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| particles4096 | changed capacity | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| prior_lr_1 | same architecture/data/update budget | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| prior_lr_3 | same architecture/data/update budget | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| prior_reg1_lr1 | same architecture/data/update budget | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| prior_reg1_lr3 | same architecture/data/update budget | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| prior_reg_0p2 | same architecture/data/update budget | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| prior_reg_1 | same architecture/data/update budget | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| prior_reg_3 | same architecture/data/update budget | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| prior_reg_zero | same architecture/data/update budget | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| slow_prior_lr3 | same architecture/data/update budget | Not run | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |
| slow_prior_lr3_budget3 | more updates | Not run | FAIL (0/24) | Not run | Not run | Not run | Not run | Not run | 0/1 | 0/1 |

`PASS` means final numerical pass AND sustained final suffix. `Final only` cannot count as converged. A one-task screen result is not an all-task win.

Full episode cards, original/effective specs, raw live/EMA curves, actions, runtime and independently recomputed verdicts: [episodes.json](episodes.json.gz). Original serial results: [results.json](results.json.gz); per-attempt numerical table: [README.md](attempts.md).

R1+R2 remains its declared formulation when cap-only knobs change. Task D-LR ratios, minibatch sizes and capacity/horizon perturbations are preserved. All metric thresholds are identical to the original task cards.

Recorded episode wall time: 888.82 seconds on a shared CPU host. Source files and exact runners are archived alongside the records.
