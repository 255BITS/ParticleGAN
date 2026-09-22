# Behavioral search — live weights

Search for a config that beats published `r1_r2_0_1` on frozen protocol `behavior-v1`. Thresholds, budgets, seed 0, host code, and production training defaults are unchanged. Candidates use only the JSON fields in [BASELINE.md](../../benchmarks/locked_shared/BASELINE.md).

Runtime for every folder below: torch `2.13.0+cu126`, CPU, one intra-op thread (set by the harness), conceptmod `5571213f5e8e129cfda45c785c3f30aad9c1d8c9`. The ten shared checks passed in each output directory. They do not rank candidates.

## Same-runtime control

Published `r1_r2_0_1` does not reproduce on this CPU. Its EMA ring curve matches the published curve through step 800 (7 modes, about 75% HQ) and then diverges. Final live weights here are **1/8 modes at 8.33% HQ**, so the control is **FAIL**. Trajectory MSE 0.002824 and two-pole gradient median 0.733 still pass and sit close to the published 0.003768 / 0.734.

Rank a new PASS against that control first. When both pass, compare the tight live margins (trajectory `identity_mse`, mode-hold modes and HQ, two-pole `grad_med`), then the EMA ring, then simplicity.

## Configs that beat `r1_r2_0_1`

`r1_r2_0_1_l2_004` is the best measured config. `r1_r2_0_1_l2_005` also beats the published row. Both keep R1+R2 at coefficient 0.1 and only lower `particle_l2` from 0.02. A repeat of `l2_005` in the neighborhood wave matched the wave A live and EMA numbers exactly.

| Rank | Config | Overall | Traj MSE | Grad med | Live modes | Live HQ | EMA modes | EMA HQ |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `r1_r2_0_1_l2_004` | **PASS** | 0.001128 | 0.711 | 8/8 | 100% | 8/8 | 100% |
| 2 | `r1_r2_0_1_l2_005` | **PASS** | 0.001466 | 0.709 | 8/8 | 100% | 7/8 | 91.9% |
| — | published `r1_r2_0_1` | PASS on the original CPU | 0.003768 | 0.734 | 7/8 | 100% | 5/8 | 65.2% |
| — | this CPU `r1_r2_0_1` | **FAIL** | 0.002824 | 0.733 | 1/8 | 8.3% | 6/8 | 50.2% |

`l2_004` is ahead of published `r1_r2_0_1` on every tight live margin and on the EMA ring (8/8 at 100% HQ, so the EMA ring bounds would pass too). `l2_005` matches the live ring ceiling, improves trajectory and gradient median, and improves EMA from 5/8 at 65% to 7/8 at 92%. Its gradient median is 0.0015 lower than `l2_004`; trajectory MSE and the EMA ring favor `l2_004`.

`r1_r2_0_1_l2_007` is a third full PASS on this CPU (trajectory MSE 0.005990, gradient median 0.731, live ring 8/8 at 92.1% HQ, EMA 6/8 at 75.2%). It beats the failing same-runtime control. It does not beat the published margin profile: trajectory MSE and live HQ are worse than published `r1_r2_0_1`, and the HQ margin is only +0.021.

None of these is simpler than `r1_r2_0_1`. They use the same `a_r1r2` arm. Simplicity does not override the margin gap.

## What failed

EMA cannot rescue a live FAIL. Full per-metric tables are in each wave README.

### Wave A — R1+R2 neighborhood

| Config | Live | Failed live targets | EMA ring |
| --- | --- | --- | --- |
| `r1_r2_0_1_l2_005` | **9/9, 29/29 PASS** | None | 7/8, 91.9% |
| `r1_r2_0_15` | 8/9, 28/29 | trajectory MSE 0.297 | 8/8, 83.7% |
| `r1_r2_0_05` | 8/9, 27/29 | mode_hold 5/8, 73.9% HQ | 5/8, 66.9% |
| `r1_r2_0_1_cover_1` | 8/9, 27/29 | mode_hold 1/8, 8.3% HQ | 6/8, 50.2% |
| `r1_r2_0_08` | 7/9, 27/29 | trajectory MSE 0.0222; mode_hold 5/8 (HQ 100%) | 5/8, 83.0% |
| `r1_r2_0_1_l2_001` | 7/9, 27/29 | trajectory MSE 0.262; mode_hold 6/8 (HQ 100%) | 6/8, 74.6% |
| `r1_r2_0_1_no_l2` | 7/9, 26/29 | trajectory MSE 0.258; mode_hold 1/8, 7.9% HQ | 6/8, 83.9% |
| `r1_r2_0_12` | 7/9, 26/29 | trajectory MSE 0.261; mode_hold 3/8, 59.1% HQ | 1/8, 7.7% |
| `r1_r2_0_1_no_vicreg` | 7/9, 26/29 | trajectory MSE 0.250; mode_hold 2/8, 33.3% HQ | 6/8, 91.6% |
| `r1_r2_0_1_cover_0` | 7/9, 23/29 | cover_leftover (kept/content/both poles); mode_hold 1/8, 8.3% HQ | 6/8, 50.2% |
| `r1_r2_0_2` | 6/9, 23/29 | trajectory MSE 0.489; residual student; mode_hold 3/8, 40.6% HQ | 5/8, 65.8% |

`r1_r2_0_1_no_l2` was the untested combination called out in the 3-toy report. On all 9 toys it fails trajectory and the live ring. Cover weight does not affect the ring, so `cover_1` / `cover_0` reproduce the same-runtime `r1_r2_0_1` ring failure. Dropping cover to 0 also breaks cover/leftover.

### Wave A follow-up — particle L2 around 0.005

| Config | `particle_l2` | Live | Failed live targets | EMA ring |
| --- | ---: | --- | --- | --- |
| `r1_r2_0_1_l2_004` | 0.004 | **9/9 PASS** | None | 8/8, 100% |
| `r1_r2_0_1_l2_005` | 0.005 | **9/9 PASS** | None | 7/8, 91.9% |
| `r1_r2_0_1_l2_007` | 0.007 | **9/9 PASS** | None | 6/8, 75.2% |
| `r1_r2_0_1_l2_006` | 0.006 | 8/9 | mode_hold 6/8 (HQ 92.2%) | 7/8, 100% |
| `r1_r2_0_1_l2_003` | 0.003 | 8/9 | mode_hold 3/8, 31.9% HQ | 3/8, 25.0% |

The pass set is not a solid interval. 0.003 fails, 0.004 and 0.005 pass, 0.006 fails, 0.007 passes, and 0.01 / 0.00 / 0.02 fail.

### Wave B — b_cap arms that looked strong on 3 toys

| Config | Live | Failed live targets | EMA ring |
| --- | --- | --- | --- |
| `music_cover` | 8/9, 27/29 | mode_hold 4/8, 49.2% HQ | 8/8, 100% |
| `no_vicreg` | 7/9, 27/29 | trajectory MSE 0.0220; mode_hold HQ 82.7% (7/8 modes) | 7/8, 91.7% |
| `no_l2_no_vicreg` | 7/9, 26/29 | trajectory MSE 0.237; mode_hold 6/8, 82.4% HQ | 4/8, 51.5% |
| `base_regularization` | 7/9, 24/29 | cover_leftover; mode_hold 6/8, 58.3% HQ | 8/8, 100% |
| `no_cover` | 7/9, 23/29 | cover_leftover; mode_hold 4/8, 49.2% HQ | 8/8, 100% |

`no_l2_no_vicreg` confirms the 3-toy trajectory failure on the full suite. `no_vicreg` cleared trajectory in the 3-toy study (MSE 0.00235) and misses it here by 0.002. `music_cover`, `no_cover`, and `base_regularization` still show an 8/8 100% EMA ring and still fail the live ring.

### Wave C

Not run. Eikonal, interpolation cap, relativistic-average logistic, and Rp hinge already missed a required live target in the 3-toy study, and the simpler b_cap edits in wave B did not repair mode hold.

## Recommendation

1. Prefer `r1_r2_0_1_l2_004` when citing a config that beats published `r1_r2_0_1` on this protocol. `l2_005` is the runner-up.
2. Leave production defaults unchanged. The live ring moves after step 800 across CPUs, and particle L2 is non-monotonic on this CPU.
3. Next check: rerun `r1_r2_0_1` and `r1_r2_0_1_l2_004` on the machine that recorded the published 7/8 at 100% HQ live ring. Do not seed-sweep, and do not retune thresholds.
4. Do not treat `l2_007` as an improvement over the published margin profile.

## Reproduce

```bash
git clone https://github.com/HyperGAN/conceptmod.git /tmp/conceptmod-reference
git -C /tmp/conceptmod-reference checkout 5571213f5e8e129cfda45c785c3f30aad9c1d8c9
python -m benchmarks.locked_shared.baseline \
  --configs reports/behavioral_search/wave_a.json \
  --reference /tmp/conceptmod-reference \
  --output reports/behavioral_search/wave_a \
  > /tmp/behavioral-search-wave-a.log 2>&1
tail -f /tmp/behavioral-search-wave-a.log
```

The harness prints `START` before each toy and a JSON `DONE` line after it, then writes `results.json` and `README.md`. `--resume` only accepts the same source, runtime, and config fingerprint.

| Folder | Candidates |
| --- | --- |
| [control_r1_r2_0_1](control_r1_r2_0_1/README.md) | same-runtime `r1_r2_0_1` |
| [wave_a](wave_a/README.md) | R1+R2 coefficient, L2, VICReg, and cover neighbors |
| [wave_a_l2](wave_a_l2/README.md) | particle L2 0.003–0.007, including a repeat of 0.005 |
| [wave_b](wave_b/README.md) | b_cap VICReg / L2 / cover arms |

This is one fixed-seed CPU search. It is not evidence of GPU, LunarLander, or cross-seed transfer.
