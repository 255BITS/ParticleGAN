# K3P constant-LR suite: AMSGrad, no instance noise, and a simpler K3P

## Question

With a constant LR and no instance noise, K3P collapses on the ring-8 shift benchmark. Adam's second moment keeps
shrinking at equilibrium, so its step size creeps up (see PR #202 and `reports/simple-critic/K3P_CONSTANT.md`).
PR #202 adds `Recipe.amsgrad`, which is off by default. This study runs the full suite to answer three questions:

1. Does constant-LR K3P with AMSGrad and no noise, or K3P + AMSGrad under the stock schedule, beat the shipped
   K3P, and does it beat the simple-critic winner `simple_B_cap3`? AMSGrad's one-way decay of the effective step
   is an implicit anneal, and we accept that.
2. Can instance noise be removed altogether?
3. Which K3P components carry weight here, and how simple can K3P get without losing a suite pass or ring stability?

Branch `k3p-constant-fix`: develop was merged at 29677200 (#203). #205–#207 landed on develop after the suite ran
and are **not** in this branch's code (#206 fixes the toy100 CUDA device-policy leak; #207 keeps toy100's xavier
critic init). No `particlegan/` code changed in this study. Every change is in the harness in this directory.

## Suite (26 tasks, `run_suite.py`)

| group | tasks | protocol | pass |
|---|---|---|---|
| native | grid100, rotated100, staggered100 | `benchmarks.toy100`, 7000 updates, seed 1234 | coverage gate AND accuracy gate |
| transfer | 19 frozen `benchmarks.transfer_suite` hosts (9 custom-loop on CPU, 10 standard-trainer) | host budgets and seeds | sustained live gate (cell = passing suffix /24) |
| hold | hold-mode_hold | mode_hold to 7500, a 1200-update hold plus a 300-update extension | all hold and extension checks |
| shift | shift-mode_hold | mode_hold, (1,0) shift at 2400, 3600 updates | deadline recovery |
| ring | ring8-shift | 20k particles, latent 2, batch 2048, ring 3×96, Fourier 3, (1,0) shift at 2400, run to 4600, observe every 10 updates on 4096 samples | 8 modes and HQ≥.90 at every check after arrival, 0 fails outside transit |
| ring | ring8-multishift | same, with +(1,0) at 2400, −(1,0) at 4600, +(1,0) at 6800, run to 9000 | arrives after all 3 shifts, 0 fails outside transit |

Ring columns:
- prehold: passing checks out of 120 in updates 1210–2400.
- arrival: updates from a shift to the first passing check (x = never arrives).
- f: failed checks outside transit.
- d: departures (a pass followed by a fail).

Each arm is one formulation run on the benchmark's own seeds. There are no seed variants. The suite is
deterministic: the final-parameter hashes reproduce across runs and across both GPUs.

Reproduce one arm with `run_suite.py --arm <arm> --device cuda:0`. Rebuild `LEADERBOARD.md` with
`summarize.py --write`. Follow progress with `tail -f logs/<arm>.log` (one line per task) or
`tail -f logs/<arm>/<task>.log` (one line per eval).

## Arms (`arms.json`)

| arm | LR | optimizer | noise | other |
|---|---|---|---|---|
| k3p_stock | cosine decay (stock) | Adam (0,.999) | stock (input .5→0, output .029) | shipped K3P |
| k3p_stock_ams | decay | AMSGrad | stock | "k3p + ams" |
| k3p_nonoise | decay | Adam | 0 | |
| k3p_nonoise_ams | decay | AMSGrad | 0 | |
| k3p_const | constant | Adam | 0 | the constant-LR failure baseline |
| k3p_const_ams | constant | AMSGrad | 0 | "k3p constant ams" |
| k3p_const_ams_c3 | constant | AMSGrad | 0 | penalty coefficient ×3 |
| simple_B_cap3 | constant | critic Adam (0,.9), critic LR ×.5 | 0 | simple-critic winner (wgan + R1 + secant + cap-all; A2 off, no guard) |
| ab_* | constant | AMSGrad | 0 | k3p_const_ams with one component removed (see Simplification) |
| k3p_simple | constant | AMSGrad | 0 | k3p_const_ams without the EMA anchor term or the direct-particle response |
| k3p_stock_oldinit | decay | Adam | stock | calibration: constructor init on the ring (ring tasks only) |

## Leaderboard

| # | arm | passes | native | transfer /19 | ring8-shift | ring8-multishift (arrival per shift) |
|---|---|---|---|---|---|---|
| 1 | k3p_stock | **18/26** | **3/3** | 14 | P 120/120 +1220 f0 d1 | F +1990 / x / +1640, f2 d4 |
| 2 | k3p_stock_ams | 17/26 | 2/3 | 14 | P 120/120 +1460 f0 d0 | F x / x / x, f10 d1 |
| 3 | k3p_nonoise | 15/26 | 0/3 | **15** | F 110/120 +1020 f10 d2 | F +1010 / x / x, f18 d4 |
| 4 | ab_noanchor (copied) | 14/26 | 0/3 | 12 | P 120/120 +430 f0 d0 | **P +430 / +190 / +20, f0 d0** |
| 5 | ab_nodirect (copied) | 14/26 | 0/3 | 12 | P 120/120 +430 f0 d0 | **P +430 / +190 / +20, f0 d0** |
| 6 | k3p_const_ams | 14/26 | 0/3 | 12 | P 120/120 +430 f0 d0 | **P +430 / +190 / +20, f0 d0** |
| 7 | **k3p_simple** | 14/26 | 0/3 | 12 | P 120/120 +430 f0 d0 | **P +430 / +190 / +20, f0 d0** |
| 8 | ab_absunits | 13/26 | 0/3 | 11 | P 120/120 +460 f0 d1 | P +460 / +190 / +60, f0 d1 |
| 9 | k3p_const_ams_c3 | 13/26 | 0/3 | 11 | P 120/120 +420 f0 d0 | P +420 / +80 / +50, f0 d0 |
| 10 | k3p_nonoise_ams | 13/26 | 0/3 | 13 | F 120/120 x f0 | F x / x / +1840, f0 |
| 11 | ab_prior1 | 13/26 | 0/3 | 13 | F 120/120 +370 f5 d3 | F +370 / +430 / +30, f5 d3 |
| 12 | ab_noA2 | 13/26 | 0/3 | 12 | P 120/120 +420 f0 d0 | F +420 / +120 / +310, f22 d2 |
| 13 | k3p_const | 13/26 | 0/3 | 13 | F 95/120 +360 f25 d1 | F +360 / +440 / +70, f52 d3 |
| 14 | simple_B_cap3 | 13/26 | 0/3 | 13 | F 98/120 +440 f27 d13 | F +440 / +360 / +90, f93 d23 |
| 15 | ab_noguard | 12/26 | 0/3 | 12 | F 120/120 x | F x / x / x |
| 16 | ab_r1only | 12/26 | 0/3 | 12 | F 0/120 f120 | F x / x / +1420, f123 d1 |
| 17 | ab_caponly | 10/26 | 0/3 | 10 | F 6/120 +240 f188 d23 | F +240 / +160 / +110, f231 d42 |
| 18 | k3p_stock_oldinit | 1/2 (ring only) | – | – | P 120/120 +1960 f0 d3 | F x / x / +1550, f0 d1 |

Hold and shift (mode_hold) fail for every arm. See Audit caveats.

Final HQ in each multishift segment (each segment is 2200 updates):

| arm | seg 1 | seg 2 | seg 3 |
|---|---|---|---|
| k3p_const_ams / k3p_simple | .993 | .995 | .991 |
| k3p_const | .993 | .957 (27 fails) | .990 |
| k3p_stock | .918 | .887 (never arrives) | .934 |
| k3p_stock_ams | .886 | .812 | .640 |
| k3p_nonoise_ams | .878 | .741 | .949 |
| ab_noguard | .029 | .032 | .136 |

### Per task (status, key metric)

Native cells show final modes / HQ. Transfer cells show the passing suffix /24. Hold cells show passing hold checks
/ 1200. Shift cells show deadline checks /81. Ring cells show fails outside transit. Generated by
`summarize.py --write` in `LEADERBOARD.md`, which also has runtimes and receipt checks.

| task | k3p_stock | k3p_stock_ams | k3p_nonoise | ab_noanchor | ab_nodirect | k3p_const_ams | k3p_simple | ab_absunits | k3p_const_ams_c3 | k3p_nonoise_ams | ab_prior1 | ab_noA2 | k3p_const | simple_B_cap3 | ab_noguard | ab_r1only | ab_caponly | k3p_stock_oldinit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grid100 | P 100/0.987 | F 88/0.978 | F 99/0.983 | F 99/0.973 | F 99/0.973 | F 99/0.973 | F 99/0.973 | F 99/0.969 | F 99/0.961 | F 99/0.978 | F 96/0.922 | F 99/0.977 | F 86/0.958 | F 100/0.933 | F 99/0.973 | F 98/0.982 | F 100/0.977 | – |
| rotated100 | P 100/0.982 | P 100/0.977 | F 100/0.975 | F 100/0.971 | F 100/0.971 | F 100/0.971 | F 100/0.971 | F 100/0.958 | F 100/0.950 | F 100/0.974 | F 100/0.966 | F 100/0.964 | F 100/0.915 | F 100/0.930 | F 100/0.971 | F 100/0.971 | F 92/0.981 | – |
| staggered100 | P 100/0.987 | P 100/0.978 | F 100/0.982 | F 100/0.970 | F 100/0.970 | F 100/0.970 | F 100/0.970 | F 100/0.959 | F 100/0.947 | F 100/0.972 | F 100/0.970 | F 100/0.965 | F 100/0.970 | F 100/0.948 | F 100/0.970 | F 100/0.970 | F 100/0.985 | – |
| hold | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 299/1200+0/0 | F 10/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | F 0/1200+0/0 | – |
| ring8-multishift | F 2 | F 10 | F 18 | P 0 | P 0 | P 0 | P 0 | P 0 | P 0 | F 0 | F 5 | F 22 | F 52 | F 93 | F 0 | F 123 | F 231 | F 0 |
| shift | F 0/81 | F 0/81 | F 0/81 | F 0/81 | F 0/81 | F 0/81 | F 0/81 | F 71/81 | F 0/81 | F 0/81 | F 8/81 | F 0/81 | F 0/81 | F 3/81 | F 0/81 | F 0/81 | F 18/81 | – |
| ring8-shift | P 0 | P 0 | F 10 | P 0 | P 0 | P 0 | P 0 | P 0 | P 0 | F 0 | F 5 | P 0 | F 25 | F 27 | F 0 | F 120 | F 188 | P 0 |
| two_pole | P 9/24 | F 0/24 | P 10/24 | F 2/24 | F 0/24 | F 2/24 | F 0/24 | F 2/24 | F 0/24 | F 0/24 | F 0/24 | F 2/24 | P 10/24 | P 20/24 | F 2/24 | F 2/24 | P 13/24 | – |
| trajectory | P 23/24 | P 23/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | F 0/24 | F 0/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | P 5/24 | P 22/24 | P 22/24 | P 23/24 | – |
| residual_student | P 20/24 | P 20/24 | P 20/24 | P 8/24 | P 8/24 | P 8/24 | P 8/24 | F 0/24 | P 23/24 | P 18/24 | P 23/24 | P 8/24 | P 20/24 | F 0/24 | P 8/24 | P 8/24 | P 14/24 | – |
| unipolar | P 19/24 | P 19/24 | P 18/24 | P 18/24 | P 18/24 | P 18/24 | P 18/24 | P 14/24 | P 15/24 | P 18/24 | P 18/24 | P 18/24 | P 18/24 | P 18/24 | P 18/24 | P 18/24 | P 19/24 | – |
| ae_gan_hold | P 22/24 | P 13/24 | P 15/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | P 22/24 | P 15/24 | P 11/24 | P 22/24 | P 22/24 | P 22/24 | – |
| cover_leftover | P 13/24 | P 13/24 | P 11/24 | P 11/24 | P 11/24 | P 11/24 | P 11/24 | P 13/24 | P 12/24 | P 11/24 | P 9/24 | P 11/24 | P 11/24 | P 14/24 | P 11/24 | P 10/24 | P 9/24 | – |
| unused_token_hold | P 11/24 | P 11/24 | P 10/24 | P 10/24 | P 10/24 | P 10/24 | P 10/24 | P 7/24 | P 6/24 | P 10/24 | P 10/24 | P 10/24 | P 10/24 | P 12/24 | P 10/24 | P 10/24 | P 7/24 | – |
| mid_scale_identity | P 17/24 | P 17/24 | P 16/24 | P 16/24 | P 16/24 | P 16/24 | P 16/24 | P 13/24 | P 14/24 | P 16/24 | P 16/24 | P 16/24 | P 16/24 | P 15/24 | P 16/24 | P 16/24 | P 17/24 | – |
| mode_hold | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 1/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | – |
| v_two_broad | P 23/24 | P 23/24 | P 21/24 | P 21/24 | P 21/24 | P 21/24 | P 21/24 | P 22/24 | P 22/24 | P 21/24 | P 20/24 | P 23/24 | P 21/24 | P 23/24 | P 21/24 | P 23/24 | P 23/24 | – |
| v_unequal_mass | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 1/24 | F 3/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 2/24 | F 0/24 | F 0/24 | F 0/24 | – |
| v_unequal_width | F 0/24 | P 11/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | P 21/24 | F 0/24 | F 0/24 | P 8/24 | F 0/24 | F 0/24 | F 0/24 | – |
| v_anisotropic | P 6/24 | P 6/24 | P 7/24 | P 5/24 | P 5/24 | P 5/24 | P 5/24 | P 21/24 | P 22/24 | P 21/24 | P 21/24 | P 20/24 | P 7/24 | F 2/24 | P 5/24 | P 7/24 | F 0/24 | – |
| v_overlap | P 11/24 | P 9/24 | P 13/24 | P 9/24 | P 9/24 | P 9/24 | P 9/24 | F 0/24 | F 0/24 | P 9/24 | F 3/24 | P 11/24 | F 0/24 | P 7/24 | P 9/24 | P 9/24 | F 0/24 | – |
| v_spiral | P 21/24 | P 23/24 | P 23/24 | P 24/24 | P 24/24 | P 24/24 | P 24/24 | P 22/24 | P 24/24 | P 24/24 | P 23/24 | P 23/24 | P 23/24 | P 23/24 | P 24/24 | P 24/24 | P 23/24 | – |
| i_stripes2 | P 22/24 | P 8/24 | F 0/24 | P 5/24 | P 5/24 | P 5/24 | P 5/24 | P 7/24 | P 13/24 | P 17/24 | F 0/24 | P 5/24 | F 0/24 | P 12/24 | P 5/24 | P 5/24 | F 1/24 | – |
| i_bars4 | P 10/24 | P 5/24 | P 18/24 | F 3/24 | F 3/24 | F 3/24 | F 3/24 | P 20/24 | F 0/24 | P 15/24 | F 0/24 | F 3/24 | P 18/24 | F 0/24 | F 3/24 | F 3/24 | F 0/24 | – |
| i_blobs4 | F 0/24 | F 0/24 | P 18/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | P 9/24 | P 21/24 | F 0/24 | P 18/24 | F 0/24 | P 18/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | – |
| i_intensity2 | F 0/24 | F 0/24 | P 6/24 | F 0/24 | F 0/24 | F 0/24 | F 0/24 | F 4/24 | F 0/24 | F 0/24 | P 5/24 | F 0/24 | F 1/24 | P 11/24 | F 0/24 | F 0/24 | F 3/24 | – |

### Does AMSGrad's one-way decay slow repeated shifts?

**Not at constant LR.** k3p_const_ams arrives at +430, then +190, then +20 after the three shifts, so it gets
faster each time, and its HQ stays at .99 in every segment. The spike guard clips at each shift (87 clips on
ring8-shift, 163 on multishift), which keeps AMSGrad's max second moment from being inflated by the transient.

**AMSGrad on top of LR decay is a double anneal, and it does stop re-tracking:**
- k3p_stock_ams never arrives after any of the 3 shifts, and its HQ falls to .64.
- k3p_nonoise_ams arrives only after the third shift (+1840) and never arrives on ring8-shift.

Plain Adam with decay does no better:
- k3p_stock misses the second shift.
- k3p_nonoise misses the second and third.

So the shipped schedule is also too annealed for 9000-update tracking. AMSGrad belongs with a constant LR, not
with decay.

## Noise removal

- **With LR decay** (k3p_stock → k3p_nonoise):
  - loses all 3 native tasks;
  - loses ring8-shift (10 fails);
  - gains 1 transfer pass (15 vs 14), which is within init noise (see Audit caveats).
- **With constant LR** (k3p_const, k3p_const_ams, k3p_simple): no noise from step 0. These arms are the best ring
  trackers, but they still fail all 3 native tasks.
- **Why native fails.** The no-noise arms cover the modes: 99–100/100, HQ .97–.98. They fail the **accuracy
  gate** ("only 0/5 terminal postinitial checks pass"). On grid100 the per-mode covariance eigenvalue ratio runs
  .147–1.906, so the modes are there but the within-mode shape is off. Instance noise is currently what calibrates
  within-mode width and shape on toy100.
- **Verdict:**
  - Noise is **not needed** for tracking or stability. Every best ring result has no noise.
  - Noise **is needed** for the native toy100 accuracy gate.
  - It can't be removed from the default until something else closes that gap.
- **Receipts** (confirmed by the audit):
  - Noise was 0 on every step: host counters, the ring eval lines and the standard-trainer wrappers were not
    built.
  - Every LR group in the constant-LR arms had min = max.
  - The optimizer state showed AMSGrad was active (`max_exp_avg_sq` present).

## Simplification (base: k3p_const_ams)

Activity counters and final-parameter hashes were added to each task result.

| component | evidence | verdict |
|---|---|---|
| EMA anchor term (EMA critic, anchor decay, late-phase penalty) | at constant LR the anchor weight s is 1 on every call, so the anchor never engages; turning it off gives the same final hash as the base on every task checked | **inert at constant LR, removed** |
| direct-particle response | the hash changes only on two_pole (FAIL either way, suffix 2→0) | **removed** |
| spike guard | ab_noguard: the penalty hits about 9e4 on the first update after the shift, the ring never recovers (HQ .03 at 4600), both ring tasks are lost, 12/26 | load-bearing |
| A2 latent damping | ab_noA2: 22 multishift fails, 13/26 | load-bearing |
| prior LR ×2 | ab_prior1: both ring tasks lost (f5 d3); +3 / −4 tasks | load-bearing for the ring |
| R1 on reals | ab_caponly: 6/120 prehold, f188, 10/26 | load-bearing |
| one-sided cap on fakes | ab_r1only: the ring never holds its modes (0/120 prehold, f120), 12/26 | load-bearing |
| RMS units (1/d, 1/√d) | ab_absunits (plain L2): −1 net task, +1 departure on each ring task | keep; plain units are not simpler |
| penalty coefficient ×3 | k3p_const_ams_c3: 13/26 (−1 transfer), same ring stability | not needed |

**k3p_simple** = AMSGrad (0,.999), constant LR .00425, no noise, prior LR ×2, penalty
`coeff/2 · [mean‖g_real‖²/d + mean relu(‖g_fake‖/√d − 1)²]`, spike guard (5× after 200 steps) and A2 latent damping.
It has no EMA anchor term, no blend schedule and no direct-particle response. Public recipe fields:
`amsgrad=True, reg_anchor_weight=0, direct_particle_gain=False, direct_particle_betas=(0,.999)`, plus
`lr_floor = network_lr_floor = 1` and noise 0.

It scores **14/26**. Its ring and multishift results match k3p_const_ams exactly. Every task status matches too;
the only metric change is the two_pole suffix (2→0, FAIL either way). `GANTrainer` still deep-copies D for
the EMA critic, which is never updated at a constant LR. Removing it would need a trainer change.

## Audit caveats

- **Init is not what the brief asked for.**
  - The new develop init (`particlegan.init.deterministic_orthogonal_`) was applied only on the 2 ring tasks.
  - Native, transfer, hold and shift use each entry point's constructor init, the same in every arm. Since #203,
    develop has no recipe-path `batch_feature_zero`, and the toy100 and transfer entry points make no init call.
- **The calibration did not reproduce develop's 22/22.**
  - k3p_stock scores 17/22 on the 22 non-ring toys, with configs byte-identical to `gap-fill-20260925`.
  - Cause: the 7 standard-trainer transfer toys draw their constructor init on CUDA here, so it differs from the
    archived fixtures. The 9 custom-loop hosts match their fixtures exactly.
  - Re-running k3p_stock with the fixtures flips 4 of 7 toy verdicts, for an estimated ~19/22.
  - **So a 1–2 pass transfer gap between arms is within init noise.** The ring and native comparisons are not
    affected, because every arm shares the same init there.
  - k3p_stock_oldinit ran only the two ring tasks, since the other tasks already used constructor init.
- **mode_hold is broken.** hold, shift and toy-mode_hold fail for every arm, including the host run directly with
  its own penalties. With `b_cap` it holds 6 modes, and with `a_r1r2` it holds 1 mode at HQ .08. Only 23 tasks
  are informative.
- **How AMSGrad was applied.** On the custom-loop hosts, AMSGrad was forced by patching
  `torch.optim.Adam.__init__` (`suite_adapter.py`), not through the PR's `recipe.amsgrad` path. The ring,
  native and standard-trainer tasks used the recipe path.
- **The spike guard reads `exp_avg_sq`, not AMSGrad's `max_exp_avg_sq`** (`particlegan/k3p.py:122`).
- **The exact-match checks were partly at 400 steps.**
  - ab_noanchor and ab_nodirect were not trained at full budget. Their `result.json` was copied from
    k3p_const_ams after a bit-exact check at 400 steps (`runs_smoke/`), and the leaderboard marks them "copied".
  - At 400 steps, the 20 transfer, hold and shift tasks are full-budget, but native ×3, hold and both ring
    tasks are shortened.
  - For those 6 tasks, the only full-budget support is k3p_simple matching k3p_const_ams on metrics.
- **Overwritten logs.** The ring logs for k3p_const_ams, ab_noanchor and ab_nodirect were overwritten by the
  400-step smoke run. Their `result.json` files are intact.
- **Harness fixes during the run.** None of them touch `particlegan/`.
  - The 9 custom-loop hosts are forced onto CPU, to avoid generator-state and CPU-only errors under the CUDA
    policy.
  - The native verdict is re-derived from `bench/gate-<problem>.json` (`fix_native_results.py`).
  - The simple_B_cap3 betas receipt check is patched.
  - Each arm takes a launcher lock, after a duplicate launcher deleted running tasks. That arm was rerun cleanly.
  - `--runs-dir` is now resolved to an absolute path.

## Recommendation

**Keep shipped K3P (k3p_stock) as the default.** Neither constant-LR AMSGrad no-noise K3P nor k3p_simple beats
it on the suite: 14/26 against 18/26.
- They lose all 3 native toy100 tasks on the accuracy gate. That loss comes from dropping noise, not from AMSGrad
  or the constant LR: k3p_nonoise also loses them.
- The 2-task transfer gap (12 vs 14) is within init noise.

**On the constant-LR problem, k3p_const_ams and k3p_simple are the best arms tested.**
- They are the only arms with 0 fails on both ring tasks.
- They arrive after the first shift in 430 updates, against 1220 for stock.
- They re-track every multishift at HQ ≥ .99, which stock does not (it misses the second shift).
- They beat simple_B_cap3 on the ring (0 vs 27/93 fails) and pass 1 more suite task.

**Ship:**
1. PR #202's `Recipe.amsgrad` as an opt-in, documented for constant-LR use only. Do **not** combine it with LR
   decay: k3p_stock_ams and k3p_nonoise_ams lose multishift tracking.
2. k3p_simple as the documented constant-LR / tracking configuration. The anchor and direct-particle removals
   are proven only at constant LR. Under decay the anchor engages, so don't strip it from the default until an
   ablation under decay shows it is inert there too.

## Next steps (no seed runs)

1. **Test constant LR + AMSGrad with noise kept** (k3p_const_ams + stock noise), a cell the suite doesn't have.
   It decides whether native needs noise or needs decay.
   - If native passes, the constant-LR recipe is `k3p_simple + noise`.
   - If not, test `k3p_simple + output noise only` (.029, no input anneal).
2. **Fix native without noise.** Target the within-mode covariance error directly with a small noise-free
   within-mode width or shape term, then check it on grid100's cov eig ratio.
3. **Point the spike guard at `max_exp_avg_sq` when AMSGrad is on**, then rerun ab_noguard and k3p_simple on
   the ring.
4. **Pin transfer init** (archived fixtures, or CPU constructor draws) and merge develop #205–#207. Then rerun
   k3p_stock, k3p_simple and k3p_nonoise on the 19 toys, so that 1–2 pass gaps mean something.
5. **Fix the mode_hold route** (the `suite_adapter` legacy graft or the hold/shift protocol) so that hold and
   shift separate the arms.
6. **Ablate the anchor under LR decay** before simplifying the default.

