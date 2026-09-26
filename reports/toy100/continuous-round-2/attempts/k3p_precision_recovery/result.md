# k3p_precision_recovery

K3P stays the selected base. Nothing in this lane passed hold, extension, and the full shift verdict together. `current-research-base.json` was not edited. A3 was not used as a starting formulation.

Parent, reused and not rerun: 22/22 toys, ring hold 1200/1200 (min HQ 0.90723), extension 300/300 (min HQ 0.98779), target-shift deadline 28/81, delay 1130. Pinned mechanism `d2eb08ee932b288cbba25cd1e7be3a9572b129bd1baf0b79718be1eb37ba9391`.

## Leaderboard

Ranked by hold, then extension, then timely recovery. Toy gates were not opened.

| Candidate | Hold | Extension | Pre-shift continued | Deadline | Delay | Toys |
|---|---|---|---|---|---|---|
| K3P parent (published) | 1200/1200 | 300/300 | published with the 28/81 failure | 28/81 FAIL | 1130 | 22/22 |
| **p3_floor_reopen** | **1200/1200, min HQ 0.99219** | **300/300, min HQ 0.99512** | **120/120, min HQ 0.99463** | **77/81 FAIL** | **780** | NOT_RUN |
| p2_generator_collapse | 1200/1200, min HQ 0.99219 | 300/300, min HQ 0.99512 | 120/120, min HQ 0.99463 | 0/81 FAIL | none | NOT_RUN |
| p1_hysteresis_dwell | FAIL, incomplete | not a passed hold | 0/120 | 27/81 FAIL | 940 | NOT_RUN |

p3 is the best partial result. Stationary window 5/5 (min HQ 0.99976) and continued hold 120/120 both pass, and 77 of 81 deadline checks pass. The four misses are steps **2970, 3150, 3160, 3170**. Deadline minimum is 5 modes and HQ 0.32910. The passing suffix starts at step 3180, so the reported delay is 780. Final live state is 8 modes at HQ 0.98267; EMA is 8 modes at HQ 0.98828. That does not repair the four misses. Both optimizers took 3600 updates, so the final state is not a frozen copy.

## What was tested

All three candidates copy K3P's latent rule and direct response unchanged (`latent.py` `197df6350f5295f7d396f7d3c821808be1d15168d6e5586a89ebfbd403586139`, `response.py` `7e71d60a343f9f47e1c16600279364f0482863ce116c00f4657355638615987d`). Critic guard stays 5× Adam RMS after 200 steps. Anchor decay stays 0.999. Base LR stays 0.00425 with prior multiplier 2. The rate latch ignores the horizon except as a memo key so the driver's post-step check sees the applied multiplier. No seed sweep, coefficient grid, metric feedback, target center, or change time.

**p1 — critic-cosine dwell.** Mixing and the rate leave the K3P initial point only after a non-wiping reversal memory reaches 0.25. Input noise was removed and output noise was constant 0.029. By step 1200, still at full rate, the ring had 7 modes at HQ 0.999. The dwell then froze that 7-mode cover. A later reopen toward full rate erased it (0 modes at step 3600). The hold was stopped at step 4750 with 0 qualifying hold checks; that gate is an incomplete FAIL, not a finished 7500-step verdict. The completed shift is FAIL: stationary 0/5, continued 0/120, deadline 27/81.

**p2 — generator-gradient collapse, K3P noise restored.** The only pre-dwell difference from K3P on p1 was noise, and 8 modes were missing. p2 restores K3P input noise 0.5 (anneal end 0.1) and output warmup 0.2 on the driver's frozen noise horizon of 1200. The rate stays at the initial value until the 0.98 EMA of generator+prior gradient RMS stays under 0.05 of its own peak, then dwells at 0.10 of the initial rate (G/D 0.000425, prior 0.00085). After that first dwell the critic mix locks at the EMA anchor and does not follow later rate changes. Anchor gap is multiplied by `1 + relu(-cosine)`. Hold converged at step 1400 with 0 settling failures, 8 modes already at step 600. Shift pre-hold passed. After the shift, generator RMS rose from 0.00026 to 0.0036 while the rate stayed at 0.000425, because the quiet EMA chased the rise. Deadline 0/81. Final HQ 0.86426 with 8 modes.

**p3 — same acquisition as p2, asymmetric reopen floor.** While dwelling, RMS is compared with a floor that falls with the quiet regime and creeps upward at 0.999, so a rise can stay above 5× that floor. The post-lock reopen target is 0.25 of the initial rate, below the ~0.66 fraction that erased p1's cover. Hold and extension match p2 (same convergence step, same minima). The shift pre-hold matches p2. Deadline improves to 77/81, but the adapt phase aborts immediately: the return test is `quiet / lifetime peak < 0.05`, and the acquisition peak is 0.142, so post-shift RMS is already "cool." Sampled multipliers after step 2400 peak at 0.163 (step 3150) and 0.115 (step 2550, the only sample tagged `adapt`). Applied rate therefore never sits near the 0.25 target (0.00106). The 3150–3170 misses sit on that second bump.

p3 shift added 3081 EMA-critic forwards (518 pure-R1 calls, 36 blended, 3046 anchor calls). The anchor is on for the dwell and the recovery. Noise on p2 and p3 is still the K3P schedule referenced to noise horizon 1200. That is a retained scheduled component, not a horizon-free final rule. A two-horizon prefix was not run. The 200-step guard warmup and the EMA decays are estimator memory, not a declared training budget. Search time is not an input to the rate.

## Not run

Matched frozen recovery control, the sensitive four (`mode_hold`, unequal mass, unequal width, stripes), the other transfer toys, native 7000-update coverage and accuracy, the horizon-prefix audit, and the delayed/repeated-change stress. Deadline failure stops that ladder. None of those are inherited from K3P.

## Replay

```sh
export CUDA_VISIBLE_DEVICES=GPU-72c1b506-891d-b8bc-b353-e020585e1c47
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/tmp/pr38-default-env/bin/python
REPO=/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda
FIX=/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt
CAND=/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/k3p_precision_recovery/20260925T165310Z-3675734/repo/reports/toy100/k3p-precision-recovery/p3
# p3 mechanism sha256 607cdac6e1ccd4e2c080ac1083c5836ef1df2b617a4f2fde13013b0cb1799471
# p3 precision_policy sha256 57fca1ac270e3372a647918863d8d3fb3ac30475363d265d34849f71f26b0132
# p3 config sha256 408ac45184d81dd0a1c4b5f7bdea788f4d7a5ba42d666e508dfabadb277c0dc0
$PY -u $CAND/hold.py --repo $REPO --config $CAND/config.json --task mode_hold --backend cuda \
  --initial-state $FIX --output /tmp/p3-hold --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6
$PY -u $CAND/shift.py --repo $REPO --config $CAND/config.json --task mode_hold --backend cuda \
  --initial-state $FIX --output /tmp/p3-shift --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6
```

Logs and raw results: `repo/reports/toy100/k3p-precision-recovery/logs/` and `runs/`. Ledger: `tests.jsonl` beside this file.

## Next step

Do not promote p3. The acquisition and the long hold are in hand; the shift dies on four precision checks while the 0.25 reopen never actually holds. The return test should compare generator RMS with the dwell floor, not with the acquisition peak, so a triggered reopen remains at 0.25 until the gradient falls back. Do not grid that cap. Keep K3P noise labeled until a later candidate replaces it without reading the training budget.
