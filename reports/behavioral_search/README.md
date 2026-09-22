# Behavioral search notes

This folder is a side search. It does **not** replace the leaderboard Codex is maintaining on PR #38.

Codex owns:

- [Live leaderboard](../behavioral_baseline/README.md)
- [Default-selection note](../behavioral_baseline/default_selection.md)
- [62-attempt ledger](../behavioral_baseline/search/README.md)

Do not write those paths from this search. Thresholds, budgets, seed 0, and production defaults stay frozen.

## Which results are whose

| Source | Where | What it establishes |
| --- | --- | --- |
| Codex, on the CPU that reproduces published `r1_r2_0_1` | `reports/behavioral_baseline/` | Four `b_cap` configs pass all 29 bounds with **8/8** live modes. They beat `r1_r2_0_1` (7/8). Leading config: `bcap_k1p25_c3p0_lr0p85`. |
| Codex, same runs | `r1_r2_0_1_no_l2` in that leaderboard | The wave A “no L2” hypothesis. Trajectory MSE 0.252 **FAIL**. Live ring 8/8 at 100% HQ. EMA 7/8 at 83.3%. |
| This agent, earlier waves | `wave_a/`, `wave_a_l2/`, `wave_b/`, `control_r1_r2_0_1/` | Same protocol name, but this CPU does not reproduce the published ring. Not used to outrank Codex. |
| This agent, after Codex landed | `leader_repro/` | Re-ran only Codex’s leading config here. Two-pole agrees to ~1e-7. Trajectory and the ring do not. |

## Codex configs that beat `r1_r2_0_1`

These numbers are Codex’s, copied from the #38 leaderboard. Rank is theirs: passed toys, passed bounds, then live ring modes, HQ, and effective modes.

| Rank | Config | Arm | Live ring | Traj MSE | Grad med | EMA ring | Late 8/8 and HQ≥90% |
| ---: | --- | --- | --- | ---: | ---: | --- | ---: |
| 1 | `bcap_k1p25_c3p0_lr0p85` | b_cap, κ=1.25, coeff=3, L2=0, LR×0.85 | 8/8, 100% HQ | 0.002481 | 0.514 | 8/8, 100% | 4/5, worst HQ 83.9% |
| 2 | `b_cap_k1_25_c2_lr0_85` | b_cap, κ=1.25, coeff=2, L2=0, LR×0.85 | 8/8, 100% HQ | 0.002112 | 0.503 | 8/8, 100% | 3/5, worst HQ 41.9% |
| 3 | `b_cap_k1_25_c2_0_no_l2` | b_cap, κ=1.25, coeff=2, L2=0, LR×1 | 8/8, 91.75% HQ | 0.001478 | 0.584 | 8/8, 100% | 3/5, worst HQ 75.2% |
| 4 | `bcap_k1p25_c2p0_lr0p8` | b_cap, κ=1.25, coeff=2, L2=0, LR×0.80 | 8/8, 91.67% HQ | 0.001591 | 0.471 | 8/8, 100% | 2/5, worst HQ 34.4% |
| 5 | `r1_r2_0_1` | a_r1r2, coeff=0.1, L2=0.02 | 7/8, 100% HQ | 0.003768 | 0.734 | 5/8, 65.2% | 0/5, worst HQ 16.6% |

All four `b_cap` rows beat `r1_r2_0_1` on mode count, gradient median, trajectory MSE, and the EMA ring. They use the stock penalty arm rather than R1+R2. Codex’s leader is the coeff-3, LR×0.85 row because its late checkpoints stay at 8/8 and its worst late HQ is 83.9%, not because the other passes fail the 29 bounds.

`r1_r2_0_1_no_l2` does not beat that list. Codex already ran it on all 9 toys: trajectory fails, so it is not a PASS even though the live ring is 8/8 at 100% HQ.

## This CPU does not reproduce those rings

Published `r1_r2_0_1` finishes here at **1/8 modes, 8.3% HQ**. The EMA curve still matches Codex through step 800, then diverges. Codex’s leader, rerun in [leader_repro](leader_repro/README.md) with their current harness, finishes at **5/8 modes, 66.7% HQ**, and trajectory MSE is 0.0216 (over the 0.02 bound). Their recorded trajectory MSE is 0.00248. Two-pole travel matches their value exactly and the gradient median differs by about 6e-8.

So a PASS recorded only in `wave_a_l2/` is not a win over `r1_r2_0_1` or over `bcap_k1p25_c3p0_lr0p85`.

On this CPU only, R1+R2 coeff 0.1 with `particle_l2` 0.004, 0.005, or 0.007 met all 29 bounds, and 0.005 repeated exactly. Those runs are kept as raw evidence. They are not appended to `reports/behavioral_baseline/passing_configs.json`.

## Wave A / B hypotheses Codex already covered

Do not rerun these. Codex’s full-suite or screen result stands.

| Hypothesis | Codex result |
| --- | --- |
| `r1_r2_0_1_no_l2` | Full suite. Trajectory FAIL (MSE 0.252). Live ring 8/8, 100% HQ. |
| Drop particle L2 on b_cap, then raise coeff / change LR | Full suite for coeff 2 and 5, LR ×0.5 and ×0.25, then the κ=1.25 grid. The four passes above came out of that search. |
| VICReg 0, 0.01, 0.025, 0.075, 0.1, 0.2, 0.5 on κ=1.25 coeff 2, no L2 | 3-toy screen. None cleared the 8-mode bar they required before the other six hosts. |
| Hinge RP and least-squares RP on the strong b_cap settings | 3-toy screen. Mode hold failed. |
| Coeff 2.75, 3.25, 3.5, 4 at κ=1.25, LR×0.85 | 3-toy screen. 3.25 keeps 7/8 at 100% HQ but fails trajectory. The others miss mode hold or trajectory. |

## Gaps Codex did not run

Left for a machine that still reproduces the published `r1_r2_0_1` ring (7/8 at 100% HQ) and Codex’s leader (8/8 at 100% HQ). Not run here, because this CPU already moves both of those results.

- Learning rate 0.83, 0.84, 0.86, 0.87 at κ=1.25, coeff 3, L2=0. Codex tested LR×0.85 only at coeff 3. Their note flags late update size as the open question: the leader’s step-1,050 HQ is 83.9%.
- `particle_l2` 0.005 and 0.01 on that same coeff-3 config. Every full PASS Codex kept uses L2=0.
- Kappa 1.2 and 1.3 at coeff 3, LR×0.85. They tested 1.25 and, at coeff 2, kappa 1.5.

R1+R2 coefficients 0.05–0.2 and the L2 neighborhood around 0.1 were not in Codex’s 62. They are lower priority now: the stock `b_cap` arm already has four full passes with 8/8 modes, and R1+R2 is the more exotic penalty. This CPU’s copies of those R1 rows are in `wave_a/` and `wave_a_l2/` and should not be promoted.

## Reproduce

```bash
# Codex leader, separate output. Do not point --output at reports/behavioral_baseline.
python -m benchmarks.locked_shared.baseline \
  --configs reports/behavioral_baseline/leading_config.json \
  --reference /path/to/conceptmod \
  --output reports/behavioral_search/leader_repro \
  > /tmp/behavioral-search-leader-repro.log 2>&1
tail -f /tmp/behavioral-search-leader-repro.log
```

Conceptmod reference remains `5571213f5e8e129cfda45c785c3f30aad9c1d8c9`. This is one fixed seed. It is not a GPU or LunarLander result.
