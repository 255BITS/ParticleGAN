# Canonical re-baseline of the published continuous-learning mechanisms

Draft evidence only. This does not qualify a release, and it does not claim a
full toy-suite pass. No training rule, width, cap, or gain was changed.

Scoring used `scripts/toy100_env.sh` and
`benchmarks/toy100/canonical_env.py`. The guard accepted every run:
`MKL_CBWR=COMPATIBLE` (effective raw 3), `ATEN_CPU_CAPABILITY=avx2`,
`MKL_ENABLE_INSTRUCTIONS=AVX2`, `ONEDNN_MAX_CPU_ISA=AVX2`,
`DNNL_MAX_CPU_ISA=AVX2`, one thread, `torch.use_deterministic_algorithms(True)`,
seed 0. Torch 2.14.0+cpu, oneMKL 2024.2, Intel Xeon. Logs are one JSON line
per event: `tail -f reports/toy100/canonical-rebaseline/runs/<method>.log`.

## What was ported

All five rows were ported onto this harness branch. `mode_hold.py` and
`trajectory.py` are the same blobs as the PR heads, so a foreign checkout was
unnecessary. Each call is the published setting.

| Row | Source | Call |
| --- | --- | --- |
| Unmodified host | this branch, `baseline_adam` | no patch; constant rates are only the gate's LR switch |
| #82 alternating own-curvature | `cursor/cross-curvature-bound-c712` at `4285309d`, file copied as `pr82_alternating_curvature_scratch.py` | v10: `bound_d`, G cap .25, D bound 3. Stencil, slope gate, ratio controller, and per-particle knobs stay at their off defaults |
| #84 smoothed critic | extracted `pr84_smoothed_candidate.py` from `cursor/fence-retract-exit-clip-e514` (same blob as #107/#143) | stencil width `min(.15, .5/sharpness)`, G cap .25, D bound 3 |
| #81 error-rel gain .5 | `cross_competitive_scratch.py` from `cursor/cross-drift-replay-908b` at `5501e6e4` | `error_relative_gain=.5`. Implicit and fixed-metric files match the later branch |
| #93 fence restore | `exit_aware_step_clip.py` from the fence branch, clip armed by `pr93_fence_restore` | same stencil and caps as #84, then the published nearest-real fence |

#84 is the extracted stencil, which is the frozen recipe #93 subclasses. The
#84 branch tip also runs the stopped slope gate (`_scale_bounded_g_step`) on
every smoothed step. That gate is the variant the PR closed, so it is not in
this row.

Trajectory has no `sample_ring`. The fence reads reals from that sampler, so
the ring patch is unchanged and the trajectory host skips it. Cold trajectory
MSE for #82, #84, and #93 is the same value, which is what that host should do.

## One table

Old cells are the published receipts. Canonical cells are seed 0 under this
env. Identity warm is the scheduled host; it is 200/200, min HQ .9983, on
every canonical row. The published identity, where stated, was 200/200, min
HQ .990.

Cold ring acquisition is the unchanged rule: 8 modes and HQ ≥ .9 on a
terminal suffix of 5. **None of the five pass that rule here.**

| Method | Method warm 1001–1200 | Cold trajectory | Cold ring | Stay 1210–2400 |
| --- | --- | --- | --- | --- |
| Unmodified .00425 | old 6/200. **Canonical 6/200 FAIL**, min HQ .0002, min 1 mode. First misses are 1001–1016 | old not scored. **Canonical PASS**, MSE .004989, suffix 21 | old FAIL, terminal HQ .2029 / .6267 / .4441 / .2437 / .3694, final coverage 7/8. **Canonical FAIL**, 0/24, suffix 0. Terminal 6/.1692, 8/.6128, 6/.1943, 6/.4160, 8/.7012 | old .00425 hold not scored. **Canonical 17/120**, min 4 / .1506, final 8 / .5244 |
| #82 v10 G .25 D 3 | old 200/200, min HQ .9297. **Canonical 105/200 FAIL**, min HQ .5996, min 7 modes. Misses 1089–1182 and 1184 | old PASS .00094, suffix 18. **Canonical PASS**, MSE .000942669, suffix 18 | old FAIL, 3/5 terminal checks: 7/.916, 7/.843, 8/1, 8/.995, 8/1. **Canonical FAIL**, 0/24, suffix 0, never 8 modes. Final 5 / .6685 | old not run. **Canonical 0/120**, min 0 / 0, final 6 / .9985 |
| #84 smoothed critic | old FAIL 196/200, misses 1129–1132, min HQ .8662, 8 modes. **Canonical FAIL 193/200**, misses 1108–1114, min HQ .8542, 8 modes | old PASS .000943, suffix 18. **Canonical PASS**, MSE .000942669, suffix 18 | old **PASS**, suffix 5, terminal HQ .995 / .988 / .988 / .996 / .999, first 8 at 600. **Canonical FAIL**, 2/24, suffix 0. Checks at 1000–1200: 7/.8594, 7/.9204, 8/.9985, 3/.1829, 8/.7832 | old not run. **Canonical 104/120**, min 1 / .0825, final 8 / .9978. Misses 1210, 1220, 1280–1300, 2140, 2240–2320, 2340 |
| #81 gain .5 | old 200/200, min HQ .9802. **Canonical 200/200**, min HQ .9209, 8 modes | old FAIL, MSE .02110, 3/24 under .02, no 5-suffix. **Canonical FAIL**, MSE .02120, 0/24 under .02, suffix 0 | old not run. **Canonical FAIL**, 0/24, suffix 0, peak 3 modes. Final 3 / .3560 | old not run. **Skipped.** Ring took 245s, above the 180s cheap cutoff |
| #93 fence restore | old 200/200, min HQ .965. **Canonical FAIL 197/200**, misses 1108, 1109, 1172, min HQ .7957, 8 modes | old not run. **Canonical PASS**, MSE .000942669, suffix 18 | old not run. **Canonical FAIL**, 0/24, suffix 0. Final 4 / .4790 | old 119/120, single miss 2160, worst 8 / .891. **Canonical 0/120**, min 2 / .0889, final 5 / .5208 |

Stay lists in the JSON are capped at 40 steps. #82 and #93 fail all 120
checks, so that cap hides the later misses. #84's 16 misses are the full list.
#93's clip was on (1193 of 1200 ring updates, and 2264 of 2400 stay updates).

## Where the ring loses a mode

Checkpoints are the host's 24 cold-ring observations, every 50 updates. The
loss is first visible at that checkpoint; the previous one still had the
higher count.

| Method | First drop below the running peak | First time an 8-mode checkpoint is lost |
| --- | --- | --- |
| Unmodified | 250 (1 → 0) | 750 (8 at 700, HQ .8174 → 4, HQ .2786) |
| #82 | 200 (1 → 0) | never reaches 8. Final checkpoint is 5 / .6685 |
| #84 | 750 (8 → 7, HQ .6550). That is also the first drop of any kind | 750, after 8 modes at 600, 650, and 700 (HQ .9226). The terminal window loses it again at 1150 (8 / .9985 at 1100 → 3 / .1829) |
| #81 | 350 (1 → 0) | never reaches 8. Final checkpoint is 3 / .3560 |
| #93 | 200 (5 → 4) | 900 (8 / .6445 at 850 → 5 / .6165) |

## Leaderboard under this env

No row passes cold-ring acquisition. #84 is the only curve with an 8-mode
checkpoint at HQ ≥ .9 (700 and 1100). The suffix is still 0. That is not a
reason to prefer it over the others, and it is not the old pass.

Among stays that were cheap enough to record, #84 ends at 8 / .9978 with
104/120, the unmodified host ends at 8 / .5244 with 17/120, and #82 and #93
fail every stay check. #81's stay was not run.

## Recommendation

Keep COMPATIBLE as the scoring mode. The unmodified warm failure reproduced
at 6/200, so that local filter is stable. The published #84 cold-ring pass
and the #93 119/120 hold are not. Do not retune the cap, the stencil width,
the gain, or the fence to recover them.
