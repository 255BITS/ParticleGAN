# ParticleGAN LR-free search: cross-task evidence, failure diagnoses and combination hypotheses

2026-09-26. This is a read-only synthesis of the archived PR155 evidence. No new training was run for it. The raw numbers come from the `metrics.jsonl.gz` and `learning-rates.jsonl.gz` files under
`deterministic-init-retest/evidence/`, `deterministic-init-retest/followup-evidence/` and `continuous-api-search/evidence/`, and from `broader-results.json`, `first-results.json` and `followup-results.json`.

**Bottom line:** Only one row has ever passed mode_hold, bars4 and unequal_mass together: DV16, under the old init. It fails new-init mode_hold (7/8, 0/24) and has a late departure in its 30000 run. Under the new init, no row is known to pass all three. The gates fail for three separable reasons:

1. **Steps that never shrink (mode_hold).** With Adam β1=0, no run held 8/8 to the end at full rate. RP13 had 8/8 for six checks, then fell to 2 modes. Every stable hold needed a *state-driven* step reduction after the modes were acquired.
2. **Controls that ignore coverage.** Both existing reductions (the rp5 precision close and the DV brake) ignore coverage. They freeze a 7/8 state on mode_hold and freeze a bad particle allocation on unequal_mass.
3. **Too little early exploration (bars4, unequal_mass).** Without it, bars4 locks onto one bar orientation and unequal_mass gives the 2% component only 1–2 of 256 particles. Today that exploration comes only from the *timed* startup input noise (RP5, K3P), and that noise ends on a clock.

**New learner-side signal.** Consider the runs whose mode count is stable over the last three checks. For them, the windowed generator loss Lg at the end of mode_hold rises strictly as coverage falls (§2.1):

| Stable final coverage | Final windowed Lg |
|---|---|
| 8 modes | .77–.86 (DV1–4, whose HQ is only ≈ .85: 1.05) |
| 7/8 | 1.13–1.37 |
| 6/8 | 1.58–1.93 |
| 5/8 | 2.03 |
| 4/8 | 3.1–3.2 |

The payoff gap p = (Lg − Ld)/ln2 at each precision close separates good closes from bad ones perfectly:

- the three closes that froze 8/8 had p = .19–.29;
- the three closes that froze 7/8 had p = .84–1.20.

This gives a clock-free way to tell "done" from "missing a mode".

**Most promising directions** (all eligible):

- **H0:** DV16 + RP12's loss-balance fields.
- **H1/H3:** RP12 + exploration and closing both driven by the payoff gap.

## 1. Evidence matrix

**Legend**

- **clock column**
  - `H`: horizon LR/noise anneal over `total_steps`. Disqualified.
  - `C+T`: constant LR plus a fixed-clock startup noise (input 0.5→0 by update 360, output 0→.029 by update 720).
  - `S`: state-driven rates and constant noise. This is the only strictly clock-free class.
  - `S+T`: state-driven precision plus the startup-noise clock.
  - `S0.2+T`: the rp1-variant precision (open gain .2, so rates run at 0.21×) plus the startup-noise clock.
- **Cells:** `P`/`F` followed by the number of passing checks out of 24. For mode_hold (MH), the parentheses give (final modes/final HQ%). `n:` marks the new `batch_feature_zero` initialization; `o:` marks the old historical initialization. `—` means not measured.
- **Ring (old init)** means acquire the target and recover after a target change at update 2400, over 4600 updates. Each entry reads `arrival delay:passing/observed since arrival`, first for the original target, then (after `→`) for the shifted target. `NA` means the target was never reached.
- **stat7500** gives the same arrival:retained/observed for an unchanged target over 7500 updates.
- **30000** gives four segments; the target changes at 6000, 7800 and 27000.

| cand | clock | MH new | MH old | int2 | blobs4 | stripes2 | bars4 | uneq_mass | uneq_width | other vec | new-init follow-up | ring (old) | stat7500 (old) | 30000 (old) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PUBLIC-K3P | H | F0 (6/97) | o:F0 (6/98) | — | — | — | — | o:P21 | — | — | — | 670:157/174 → NA (0/220 by 4600; released 4600 schedule) | — | — |
| PUBLIC-KA2 | H | F0 (6/100) | — | — | — | — | — | — | — | — | — | — | — | — |
| PUBLIC-KA2-CONSTANT | C+T | F0 (4/100) | — | — | — | — | — | — | — | — | — | — | — | — |
| C1 | C+T | F0 (4/100) | — | — | — | — | — | — | — | — | — | 580:172/183 → 570:164/164 | — | — |
| C2 | C+T | F0 (8/66) | — | — | — | — | — | — | — | — | — | 1960:45/45 → 1890:32/32 | 1960:296/555 | — |
| C3 | C+T | F0 (1/6) | — | — | — | — | — | — | — | — | — | NA → NA | — | — |
| C4 | C+T | F0 (0/0) | — | — | — | — | — | — | — | — | — | NA → NA | — | — |
| C5 | C+T | F0 (7/100) | — | — | — | — | — | — | — | — | — | 560:185/185 → 270:194/194 | — | — |
| C6 | C+T | F1 (8/100) | — | o:P6 | — | — | — | — | — | — | cont→2400 21/25 | 590:182/182 → 300:191/191 | 590:638/692 | — |
| C7 | C+T | F0 (7/100) | — | o:P6 | — | — | — | — | — | — | — | 590:182/182 → 270:194/194 | 590:641/692 | — |
| C8 | C+T | F0 (3/57) | — | — | — | — | — | — | — | — | — | NA → NA | — | — |
| C9 | C+T | F0 (2/49) | — | o:F0 | — | — | — | — | — | — | — | NA → NA | — | — |
| C10 | C+T | F0 (7/98) | — | o:F4 | — | — | — | — | — | — | — | 540:187/187 → 250:196/196 | — | — |
| C11 | C+T | F0 (4/92) | — | o:F0 | — | — | — | — | — | — | — | 660:175/175 → 280:193/193 | — | — |
| C12 | C+T | F0 (0/0) | — | o:F0 | — | — | — | — | — | — | — | NA → NA | — | — |
| C13 | C+T | F0 (7/100) | — | o:P11 | — | — | — | — | — | — | — | — | — | — |
| C13-R1 | C+T | F0 (7/100) | o:F0 (7/100) | o:P11 | — | — | — | — | — | — | — | 230:210/218 → 130:208/208 | — | — |
| DV1 | S | F1 (8/91) | — | — | — | — | — | — | — | — | cont→2400 24/25 | 790:162/162 → 430:178/178 | 790:655/672 | — |
| DV2 | S | F1 (8/91) | — | — | — | — | — | — | — | — | cont→2400 25/25; ring orig NA@2400, shift +380 183/183 | 790:159/162 → 400:181/181 | — | — |
| DV3 | S | F2 (8/91) | — | — | — | — | — | — | — | — | cont→2400 26/26; ring orig NA@2400, shift +500 171/171 | 790:153/162 → 520:169/169 | — | — |
| DV4 | S | F3 (8/91) | — | — | — | — | — | — | — | — | cont→2400 26/27 | NA → 460:175/175 | — | — |
| DV5 | S | F0 (6/100) | — | — | — | — | — | — | — | — | — | 550:186/186 → 350:184/186 | — | — |
| DV6 | S | F0 (6/100) | — | o:P14 | o:F0 | o:P23 | o:F0 | — | — | — | — | 550:186/186 → 390:182/182 | 550:696/696 | 550:546/546 → 410:140/140 → 420:1877/1879 → 370:263/264 |
| DV7 | S | F0 (7/100) | — | o:P13 | o:P15 | o:P23 | o:P20 | o:F0 | o:P20 | o:two_broad P18 | — | 640:177/177 → 330:183/188 | 640:687/687 | 640:537/537 → 310:146/150 → 270:1894/1894 → 250:276/276 |
| DV8 | S | F0 (7/100) | — | — | — | — | — | — | — | — | — | 640:175/177 → 310:171/190 | — | — |
| DV9 | S | F0 (5/64) | — | — | — | — | — | o:F0 | — | — | — | 580:181/183 → 320:189/189 | — | — |
| DV10 | S | F0 (8/71) | — | — | — | — | o:F8 | o:P7 | — | — | — | 620:179/179 → 320:184/189 | — | — |
| DV11 | S | F0 (5/22) | — | — | — | — | — | o:F0 | — | — | — | 560:185/185 → 290:190/192 | — | — |
| DV12 | S | P12 (8/98) | — | — | — | — | — | n:F0 / o:F2 | — | — | — | 590:182/182 → 290:191/192 | — | — |
| DV13 | S | F0 (8/87) | — | — | — | — | o:F10 | o:P14 | — | — | — | 610:180/180 → 350:181/186 | — | — |
| DV14 | S | F0 (7/94) | — | — | — | — | o:F8 | o:P14 | — | — | — | 590:182/182 → 290:191/192 | — | — |
| DV15 | S | F0 (5/62) | o:F0 (7/96) | o:F7 | o:P15 | o:P22 | o:P11 | o:F4 (+cont. 1200→2400 24/24) | o:P18 | o:anisotropic P21, overlap P24, spiral P24, two_broad P16 | — | 520:189/189 → 330:185/188 | 520:699/699 | — |
| DV16 | S | F0 (7/91) | o:P11 (8/94) | o:P8 | o:P14 | o:P21 | o:P19 | o:P15 | o:P19 | o:anisotropic P21, overlap P24, spiral P24, two_broad P21 | — | 580:183/183 → 290:191/192 | 580:693/693 | 580:543/543 → 420:139/139 → 270:1884/1894 → 270:274/274 |
| RP1-PUBLIC-PARTIAL | S0.2+T | F0 (6/100) | — | — | — | — | — | — | — | — | — | 640:146/166 (partial) | — | — |
| RP1-CUDA-EAGER | S0.2+T | F0 (7/100) | — | — | — | — | — | — | — | — | — | 640:177/177 → 500:171/171 | 640:687/687 | — |
| RP2 | S0.2+T | F0 (7/100) | — | o:F2 | — | — | — | — | — | — | — | 640:177/177 → 500:171/171 | 640:687/687 | 640:537/537 → 360:145/145 → 420:1879/1879 → 320:269/269 |
| RP3 | S0.2+T | F0 (7/100) | — | o:F0 | — | — | — | — | — | — | — | 610:180/180 → 460:175/175 | — | — |
| RP4 | S0.2+T | F0 (7/84) | — | o:F0 | — | — | — | — | — | — | — | 1250:108/116 → 350:170/186 | — | — |
| RP5 | S+T | F0 (5/75) | o:F0 (5/100) | o:P6 | o:P19 | o:P22 | o:P18 | o:P18 | o:P19 | o:two_broad P23, anisotropic P20, overlap P24, spiral P23 | — | 570:184/184 → 270:194/194 | 570:694/694 | 570:544/544 → 350:146/146 → 450:1876/1876 → 250:276/276 |
| RP6 | S+T | F0 (7/100) | o:F0 (7/93) | o:F1 | — | — | — | — | — | — | — | 660:173/175 → 380:182/183 | — | — |
| RP7 | S+T | F0 (7/100) | o:F0 (0/0) | o:F3 | — | — | — | — | — | — | — | 600:181/181 → 340:187/187 | — | — |
| RP8 | S+T | F0 (6/100) | o:F0 (6/100) | o:F10 | — | — | — | — | — | — | — | 520:189/189 → 290:192/192 | — | — |
| RP9 | S+T | F0 (3/33) | o:F0 (6/100) | o:F7 | — | — | — | — | — | — | — | — | — | — |
| RP10 | S | F0 (7/100) | o:F0 (7/100) | o:F4 | — | — | — | — | — | — | — | — | — | — |
| RP11 | S | F0 (5/58) | o:F0 (6/99) | o:F4 | — | — | — | — | — | — | — | — | — | — |
| RP12 | S | P19 (8/100) | o:F0 (7/100) | n:P12 / o:F4 | n:P17 | n:P17 | n:F0 | — | — | — | — | — | — | — |
| RP13 | S | F6 (7/84) | o:F0 (7/92) | o:F1 | — | — | — | — | — | — | — | — | — | — |
| RP14 | S | P12 (8/100) | o:P11 (8/100) | n:P9 / o:F0 | n:P6 | n:P13 | n:F0 | — | — | — | — | — | — | — |
| RP15 | S | P14 (8/100) | o:interrupted@322 | n:P5 | n:P17 | n:P21 | n:F0 | — | — | — | — | — | — | — |

**Matrix notes.**

- **New-init coverage is thin.** Beyond mode_hold, the new init has been measured only for:
  - RP12/14/15 on the four images;
  - DV12 on unequal_mass;
  - DV1–4 and C6 as unchanged checkpoint continuations to 2400;
  - DV2 and DV3 on the ring.
- **No RP10+ config has ever been tested beyond mode_hold and the images.** That means no vectors, ring, stationary or long run. RP12/14/15 and every other `S`-class RP row are unmeasured on vectors and the ring.
- **Long-run notes (old init):**
  - DV16 30000 has late departures at 16910–16990 (min 6 modes, HQ .604), which rejects it.
  - DV6 30000 has early settling misses (8280/8290, min HQ .635) but no late loss.
  - DV7 has four early misses after the first change.
- **DV15 unequal_mass:** the original run fails 4/24. An unchanged continuation from 1200 to 2400 passes 24/24 (supplementary).
- **Some mechanism differences never activate on mode_hold.** These runs have identical traces:
  - C5≡C7 (24/24 observations) and C13≡C13-R1 (24/24);
  - RP1-CUDA≡RP2≡RP3 (24/24) and DV5≡DV6 (24/24);
  - DV1–4 through observation 17;
  - C1≡ka2-constant and k3p≡ka2 through observation 16.

### 1b. Which rows pass each gate (any init)

| Gate | Passing rows | Informative near-misses |
|---|---|---|
| mode_hold | n: RP12 19, RP15 14, RP14 12, DV12 12 · o: RP14 11, DV16 11 | DV4 3/24 (late arrival at 1100), RP13 6/24 (suffix 0) |
| img_intensity2 | n: RP12, RP14, RP15 · o: RP5, C6, C7, C13, C13-R1, DV6, DV7, DV16 | o: RP8 10/24 (suffix 3), DV15 7/24, RP12 4/24 (suffix 4) |
| img_blobs4 | n: RP12, RP14, RP15 · o: RP5, DV7, DV15, DV16 | o: DV6 F0 (2 modes) |
| img_stripes2 | n: RP12, RP14, RP15 · o: RP5, DV6, DV7, DV15, DV16 | — |
| img_bars4 | **n: none** · o: RP5 18, DV7 20, DV16 19, DV15 11 | o: DV13 10 (suffix 2), DV10 8, DV14 8; n: RP12/14/15 0 |
| vector_unequal_mass | o: K3P 21, RP5 18, DV16 15, DV13 14, DV14 14, DV10 7 | o: DV15 4 (continuation 24/24), DV12 2; n: DV12 0 |
| vector_unequal_width | o: RP5, DV7, DV15, DV16 | — |
| other vectors (two_broad, anisotropic, overlap, spiral) | o: RP5 4/4, DV15 4/4, DV16 4/4; DV7 two_broad | — |
| ring (orig → shift) | o: most rows that acquire. Fastest recovery: C13-R1 +130, C10 +250, RP5/C5/C7 +270 | o: K3P never recovers (0/220); DV8 shifted collapse (min HQ .076) |
| stationary 7500 | o: RP5 694/694, DV15 699/699, DV6 696/696, DV16 693/693, DV7 687/687, RP2 687/687 | o: DV1 655/672, C6 638/692, C7 641/692, C2 296/555 |
| 30000 | o: RP5, RP2 (no departures), DV7, DV6 (early transients only) | o: DV16 late loss 16910–16990 |

**Rows closest to covering the three hard gates:**

- **DV16 (`S`-class)** is the only row that ever passed all three (old init: MH 11, bars4 19, unequal_mass 15). It also passes all images, all 6 vectors, the ring and stationary 7500. It fails new-init MH (7/8 at 91%, Lg 1.47) and has a late departure in its 30000 run.
- **RP5** passes everything except mode_hold. It is `T`-class.
- **RP12** is the strongest new-init MH result and fails bars4.

## 2. Failure diagnoses (raw metrics)

### 2.1 mode_hold: what separates a hold from a failure (all 49 new-init runs)

Measured over the late window (updates 650–1200). "G-rate/base" is the mean applied generator rate divided by its starting value. Lg is the 50-update windowed generator loss; ln2≈.69 is the balanced value.

| Group | G-rate/base | Mode-count changes | Min HQ | Final Lg | Rows |
|---|---:|---:|---:|---:|---|
| Stable 8/8 hold | .01–.42 | 0 | .89–.99 | .78–.84 | RP12 (.010), DV12 (.20), RP15 (.26), RP14 (.42) |
| Stable 7/8 lock | .34 (C10: 1.0) | 0 | .84–.99 | 1.19–1.37 | RP10, RP6 (precision closed at 819/833 with one mode missing); C10 at full rate |
| 8 modes, slow quality | .13–.69 | 1–5 | .09–.57 | .77–1.06 | DV1–4 (HQ 83–91%), DV10, DV13 |
| Oscillating at full rate | .76–1.0 | 1–8 | .00–.62 | 1.02–3.17 | C1, C2, C5–C7, C11, C13, RP1–RP5, RP8, RP9, RP11, RP13, DV5–7, DV9, DV11, DV14–16, ka2-constant |
| Dead / never acquires | 1.0 | 0–2 | 0 | 1.2–3.0 | C3, C4, C8, C9, C12 |

**Final windowed Lg (updates 1100–1200)**, for runs with the same mode count at the last three checks:

| Stable final coverage | Rows | Lg | p = (Lg − Ld)/ln2 |
|---|---|---|---|
| 8 modes | DV13, RP14, DV12, RP15, RP12, DV10 | .77–.86 | .17–.35 |
| 8 modes, HQ ≈ .85 | DV1–4 | 1.05 | .65 |
| 7/8 | DV14, RP4, RP10, RP7, RP2/3, DV8, DV7, C10, RP6 | 1.13–1.37 | .79–1.21 |
| 6/8 | RP8, DV5/6, KA2, K3P | 1.58–1.93 | 1.36–2.05 |
| 5/8 | RP5 | 2.03 | 2.15 |
| 4/8 | C11, ka2-constant | 3.11–3.17 | 3.9–4.1 |

- Oscillating runs fall in between (DV9 and DV15 end at 5 modes with Lg 1.14–1.17).
- The dead numerics (C8/C9 at 2–3 modes) sit at 2.5–3.0.

**At the precision close** (Lg averaged over the 50 updates before it):

| Close | Coverage frozen | p at close |
|---|---|---|
| RP12 at 551 | 8/8 | .29 |
| RP14 at 857 | 8/8 | .19 |
| RP15 at 788 | 8/8 | .20 |
| RP10 at 819 | 7/8 | .92 |
| RP6 at 833 | 7/8 | 1.20 |
| RP7 at 1120 | 7/8 | .84 |

A missing mode leaves real mass that the critic can always separate from every fake. That keeps the payoff gap open. The learner can measure this gap itself; it needs no evaluator.

1. **Every stable 8/8 hold comes after a state-driven rate cut.**
   - The precision closes (rates drop to 1%) at update 551 for RP12, 857 for RP14 and 788 for RP15. DV12's mobility falls to 0.07 of base by 1200.
   - The same holds under the old init: RP14 closed at 1083 and DV16's mobility fell to 0.07.
   - RP13 shows the counter-case. It reaches 8/8 for six checks at full rate (450–700), then drops to 2 modes at update 800, because its precision never closes.
2. **Both reductions ignore coverage.** A close freezes whatever state exists at that moment.
   - RP10, RP6 and RP7 closed at 7/8 and froze the hole. So did RP12 under the old init (close at 1147).
   - As a result, a pass depends on the arrival order, which is sensitive to initialization:
     - RP12 went from o:7/8 to n:8/8;
     - DV16 went from o:8/8 to n:7/8;
     - RP14 passes under both.
3. **Game balance is what makes the plateau.**
   - RP12 sits at Lg≈.84 from update 300 on, and is already 8/8 at full rate between 300 and 550, before the close. RP10, which differs only by the two RP12 loss changes, sits at Lg≈1.2.
   - The same pattern holds under the old init: RP12 held a flat 7/8 at full rate from 850 to 1150.
   - Runs that oscillate show critic domination. RP5's Lg climbs 1.1→2.1, RP9's reaches 2.5, and ka2-constant's goes 1.56→3.19 while it loses modes 6→4.
   - all_pairs pairing alone (RP11) does not do this: 5/8, oscillating. The balance comes from all_pairs together with `uniform_sampled`, which gives equal loss mass to each distinct particle in the batch.

### 2.2 Why RP5 fails mode_hold

Under the old init, RP5 passed all 4 images, 6 vectors, the ring, stationary 7500 and 30000.

- **The precision never closes.** It records 0 closings in 1200 updates. The smoothed gap velocity stays positive, so the required 50-update contraction streak never starts. All groups run at full rate the whole time: G .00425, prior .0085, D .00425.
- **The end of the timed noise destabilizes the game.** Input noise reaches 0 at update 360, and output noise ramps to .029 by 720. After 360:
  - windowed Lg goes .94 (update 250) → 1.49 (450) → 1.66 (500) → 1.69 (800) → 2.09 (1150);
  - HQ collapses to 0/0 at update 500 and to 1/0 at 800;
  - the run ends at 5/8 (HQ 75%).
- **RP10 isolates the cause.** RP10 is RP5 with constant noise (input 0, output .029). It shows Lg 1.1–1.3 and a flat 7/8 from update 450.
- **Why only mode_hold.** It has 12 particles for 8 modes, about 1.5 particles per mode, so one particle hopping deletes a mode. The images have 8–16 particles per mode and the vectors have 256 particles in total, so the same jitter does not remove a mode. On images the precision also closes: RP12 blobs at 383, RP14 blobs/stripes at 521/474, RP15 blobs at 308.
- **Old init shows the same pattern:** RP5 ended at 5/8 with HQ 100, precision open, full rates.

### 2.3 Why constant-LR rows oscillate

- **Adam with β1=0 turns noise into a random walk.** Every G, D and prior group uses Adam with β1=0 (betas [0, .999]; particle betas [0, .9]). Near a noisy equilibrium the mean gradient goes to 0, but g/√v stays of order 1. Each update therefore moves each coordinate by about lr in a random direction. This random walk does not decay.
- **The logged activity confirms it.** Precision "activity" is the RMS of Δθ/lr. At the end of open runs it stays at .05–.34 against peaks of .68–.83 (RP12 bars4 .05, RP5 mode_hold .21, RP15/RP14 bars4 .29/.34).
- **The walk moves particles between modes.** With 12 particles and a 2× prior rate, HQ swings accordingly. In the 96-row constant-LR wave, c006 shows HQ .81/.22/.50/.66/.64 over updates 1000–1200, and c036 drops .98→.53 in 50 updates.
- **Changing the optimizer does not fix it.** The 79 constant-rate optimistic/AMSGrad rows all fail. The best, bg016, had 4 of 5 terminal checks at HQ .92/1.0/.96/.92/.84.
- **Constant rows also drift toward critic domination.** ka2-constant's Lg rises from 1.6 to 3.2.
- **Conclusion:** one fixed step is either too large to hold 12 particles or too small to acquire and recover in time. Every pass on record, in both the API and research tables, includes a reduction:
  - the research passes pnb3, jt2 and ra use clocks, and sn3 failed its long hold;
  - the API passes use a state-driven reduction.

### 2.4 bars4: why RP12, RP14 and RP15 score 0/24

The task has 32 particles and 600 updates. Its four modes are vertical bars at columns 1 and 5 and horizontal bars at rows 1 and 5. A pass needs each bar to hold at least 4/32 HQ particles and HQ ≥ .90 (rmse ≤ .10) over the final 5 checks.

**RP12: a coverage lock, not a quality problem.**

- By update 50, all particles sit on the two vertical bars (mode fractions .47/.53/0/0).
- Horizontal-bar mass never exceeds .06 through update 600.
- Quality on the two bars it holds is excellent: HQ .97, rmse .036.
- The precision stays open at full rates for all 600 updates, so this is not a rate lock.
- The lock forms during very fast early fitting (rmse .10 by update 50), before the critic separates the horizontal orientation. After that, no gradient pulls particles toward it.

**RP14 and RP15: a quality floor with coverage intact.**

- Each bar keeps 12–41% of the particles from update 75 to 600.
- rmse falls slowly: RP14 .38→.165 (HQ .34); RP15 .38→.116 (HQ .72, still improving at 600).
- **The cause is the tangent-support repulsion.** Its bandwidth is the current median nearest-neighbor distance. So mean crowding stays at .37–.47 at every scale, and the added force stays at:
  - .37–.47× the adversarial norm for RP15;
  - .58–.98× the adversarial norm for RP14 (mean of per-row max ratio).
- The force stays that large even though the correct solution is 8 near-identical images per bar.
- Median nearest-neighbor distance at update 600 is still .23 (RP15) and .94 (RP14). On blobs and stripes it reaches .02–.07 by update 200–300.
- This same always-on force explains their late image arrivals elsewhere: RP14 blobs at 475 and intensity at 375; RP15 intensity at 500.

**Old-init references:**

- RP5 also sat on 2 bars at updates 100–125, then found all 4 by update 150 while input noise was still ≈.29. It then passed 18/24.
- DV7 passes (20/24) without any input noise. Its payoff-scaled critic rate slows D early (D rate .0022 vs G .0031 at update 100).
- DV6 fails through a 3/4 coverage lock, similar to RP12.
- DV10, DV13 and DV14 fail through HQ jitter (.81–.91) caused by latent smoothing.

**Open attribution question.** RP12's orientation lock could come from all_pairs/uniform, from removing the startup noise, or from the new init. The runs that would decide it are unmeasured: RP5, RP10 and RP11 on bars4 under the new init.

### 2.5 unequal_mass: why DV12, DV7 and DV9 fail

The task has 256 particles and target masses .55/.30/.13/.02. It passes when all of these hold over the final 5 checks:

- mean per-component covariance error ≤ .85;
- min mass ratio ≥ .25;
- min eigenvalue ratio ≥ .15;
- sw1 ≤ .18, mass TV ≤ .15 and HQ ≥ .85.

1. **Particle allocation freezes early, even at full rate.** Component masses stop changing once they are set, even at full rate: RP5 is fixed to 3 decimals from update 400, and K3P stays within ±.004 from update 100. The rare components keep whatever particles reach them during the discovery window.
2. **Timed early input noise discovers every component in 100–200 updates.** RP5 reaches .093/.015 (mass ratio .72) and K3P .092/.017 (.71). Every DV row runs without input noise and sits on 2 components for 200–1000 updates before finding components 3 and 4.
3. **DV12 (new init) fails on both metrics.**
   - Component 3 appears at update 200 and component 4 at 400. By then the DV brake is already cutting rates: G .0042 → .0015 at 400 → .00018 at 1200.
   - The allocation freezes at .602/.339/.055/.005. The 2% component holds about one particle (ratio .005/.02 = **.232 < .25**). Component 3 holds 42% of its target.
   - That single particle has isotropic smoothing clipped to half the nearest-particle distance, so it cannot take the component's shape. The per-component covariance errors are .14/.12/.73/**2.52**, which gives a mean of **.879 > .85**.
   - Under the old init, the rare component got .007 (ratio .33, a pass by roughly one particle's sampling multiplicity). The covariance still failed (.884, rare-component error ≈3).
4. **DV7 and DV9 have no latent width at all.** The 1–2 rare particles form a nearly flat cloud:
   - min eigenvalue ratio .021 (DV7) and .027 (DV9);
   - rare-component covariance error 2.39 and 4.14;
   - DV9's mean covariance error is 1.17.
5. **Learned widths fix the covariance but not the allocation.** With the same 1–2 particles, DV16 reaches a rare-component error of .31 (mean .20) and DV14 .49 (mean .28); both pass. Their mass ratio is still .33, one particle away from failing.

## 3. Complementary mechanisms and combination hypotheses

### 3.1 Mechanism → gate effect

| Mechanism (where) | Helps | Hurts / limit |
|---|---|---|
| all_pairs + uniform_sampled (RP12 vs RP10/RP11) | MH hold (n:19/24); Lg≈.84 even at full rate; 3 images fast (arrival 200–325) | bars4 orientation lock (cause not attributed); vectors unmeasured |
| rp5 reversible precision (binary: 1% network / 5% prior when calm, reopens on shock) | Makes holds possible; ring recovery and stationary/30000 with no departures (RP2, RP5, old init) | Ignores coverage (locks 7/8); never closes on a noisy 12-particle game (RP5, RP8, RP9, RP11, RP13); 1% freeze relies on the shock detector to adapt |
| rp1 precision (0.21× open rates, RP1–RP4) | Ring and stationary (RP2) | Too slow for images (intensity 0–2/24); MH 7/8 |
| secant_resolvent game update (RP4+) | Fast shifted recovery (+270); 30000 with no departures (RP5) | 2 field evaluations per update |
| Timed startup noise (RP5–9, C rows, K3P/KA2) | Early discovery: unequal_mass allocation, bars4 (RP5) | A clock (ineligible); its end triggers critic domination on MH |
| DV mobility brake (data drift + payoff) | DV12 hold; stationary retention (DV6/7/15/16) | Ignores coverage: freezes vector allocation (DV12); slow HQ (DV1–4 at 91%) |
| DV7 payoff-scaled critic rate 1/(1+payoff²) | bars4/blobs4 without noise (DV7 old 20/15) | Alone gives no latent spread for vectors (eigen .02) |
| Fixed-width latent smoothing (DV10/11; DV12 clipped) | Early 8-mode coverage on MH; DV12 HQ 98% | HQ jitter (DV10 71%, DV11 22%); rare-component covariance error |
| Learned per-particle width/shear (DV13–16) | unequal_mass/width, 4 other vectors, bars4 (DV15/16) | New-init MH fails (DV13 8/87%, DV14 7/94, DV15 5/62, DV16 7/91); DV16 late 30000 loss |
| Tangent-support repulsion (RP14/RP15) | MH coverage push (n:12/14 per 24) | bars4 quality floor; slow image arrival |
| Activation metric (RP13), two-direction resolvent (RP7), particle exploration (RP9) | — | MH oscillation / collapse (o:RP7 0/8) |

### 3.2 Configuration and code deltas of the complementary rows

All rows share lr .00425, D×1, prior×2, betas [0, .999], the ka2 critic penalty, EMA .995, `total_steps=None` and `batch_feature_zero`.

| Field | RP5 | RP10 | RP12 | RP15 | DV12 | DV16 |
|---|---|---|---|---|---|---|
| Rate control | rp5 precision | = | = | = | DV mobility | DV mobility |
| Game update | secant_resolvent | = | = | = | plain | plain |
| Noise | `initialization`: input .5→0 at 360, output →.029 at 720 | `constant`: input 0, output .029 | = | = | input 0, output .029 | = |
| Relativistic pairing | row | row | **all_pairs** | row | row | row |
| Particle loss weights | frequency | frequency | **uniform_sampled** | frequency | frequency | frequency |
| Generator update | plain | plain | plain | **shared_tangent_support** | plain | plain |
| Latent support | — | — | — | — | isotropic N(0, bw), clipped to ½ nearest-neighbor | **learned width + rank-one shear**, uniform ±, paired gradients |
| Critic rate scale | 1 | 1 | 1 | 1 | 1/(1+payoff²) | = |

**Code distances between packages** (under `deterministic-init-retest/port-source/api-*/package/particlegan`):

- RP12 vs RP10: about 30 lines (`gan_loss.py` pairing, `recipes.particle_weights`, two call sites in `training.py`).
- RP15: RP10 plus `generator_support.py` and about 10 lines in `training.py`.
- DV16 latent support: prior `log_width`/`shape_shear` parameters, `perturb_latent`, and the paired-width gradient block (about 80 lines). It can be ported without the DV rate controller.

The packages share no feature flags, so every combination below needs a small merge rather than overrides only.

### 3.3 Combination hypotheses

All hypotheses use only state-driven controls: no clock, no horizon, no per-task LR. They are ranked by expected value per unit of cost. "Payoff gap" means p = EMA of max(0, (Lg − Ld)/ln2), which DV5+ already compute (`observe_generator`).

**H0: DV16 + RP12's loss balance.**
- Base package: `api-dv16`.
- Change: port RP12's `relativistic_pairing="all_pairs"` and `particle_loss_weighting="uniform_sampled"` (about 30 lines: `gan_loss.py`, `recipes.particle_weights`, two call sites in `training.py`).
- Overrides: DV16's recipe plus `relativistic_pairing="all_pairs"` and `particle_loss_weighting="uniform_sampled"`.
- Rationale: DV16 is the only row that ever passed MH, bars4 and unequal_mass together (old init). It fails new-init MH in a critic-dominated 7/8 state (Lg 1.47; brake open at mean rate .96). The same two fields moved RP10 from 7/8 at Lg 1.19 to RP12 at 8/8, Lg .84.
- Target gates: new-init MH, while keeping bars4 and the vectors.
- Risks:
  - uniform weighting under DV16's per-draw latent perturbation is untested;
  - RP12's bars4 orientation lock may transfer (probe P1 decides);
  - DV16's late 30000 departure (16910–16990) is not addressed.

**H1: RP12 + exploration noise driven by the payoff gap.**
- Base package: `api-rp12`.
- Change: add `noise_policy="adaptive"`: σ_in = σ_max·clip((p − p0)/(p1 − p0), 0, 1), smoothed with a fixed EMA. Use σ_max = `input_noise_std` = .5, p0 ≈ .3, p1 ≈ 1.0, and keep output noise at .029.
- Rationale: exploration turns on exactly when a mode is missing or the critic dominates (p ≈ .8–2 at 7/8 and below; RP5 mode_hold's failure, §2.2), and turns off at 8/8 balance (p ≤ .27). This reproduces RP5/K3P early discovery (bars4 all 4 by update 150; unequal_mass allocation) without a clock, and reopens after a target change.
- Target gates: bars4, unequal_mass, MH robustness, ring.
- Risks:
  - p0 and p1 must suit every task. Check the equilibrium p in the image and vector logs; RP image runs do not log losses, so this needs a logging-only rerun or a probe.
  - Noise keeps precision activity high. Measure activity on the noise-free field, or close only when σ_in = 0.

**H1b: RP12 + DV7's payoff-scaled critic rate.**
- Base package: `api-rp12`.
- Change: the critic LR is multiplied by 1/(1+p²) (port `critic_scale`, about 15 lines).
- Rationale: DV7 passes bars4 and blobs4 with no noise by slowing D while it dominates (D rate .0022 vs G .0031 at update 100). This is the cheapest state-driven fix for the early lock and composes with H1.
- Target gates: bars4.
- Risk: RP12's MH p is already low, so the MH effect is small. It is untested on images for RP12.

**H2: H1 + DV16 learned latent width and shear.**
- Base package: `api-rp12`.
- Change: port the DV16 prior `learnable_width`/`learnable_shear`, `perturb_latent` (bounded uniform, rank-one shear) and the paired ± width gradients. Keep rp5 precision as the only rate control.
- Rationale: learned width is the only fix for the rare-component covariance (DV16 .31, DV14 .49 vs ≈3 without it), and DV16 passed bars4 19/24. H1 supplies the allocation that learned widths cannot.
- Target gates: unequal_mass, unequal_width, bars4.
- Risk: learned widths cost new-init MH HQ (DV13–16 end at 5–8 modes, HQ 62–94%).

**H3: precision close gated by the payoff gap.**
- Base package: `api-rp12` (+H1).
- Change: allow a close only while p < ~.4, and reopen when p stays above ~.8 for about 25 updates. The existing gap and activity rules stay.
- Rationale: the new-init 7/8 locks closed at p = .92 (RP10), 1.20 (RP6) and .84 (RP7). The 8/8 closes came at p = .29 (RP12), .19 (RP14) and .20 (RP15). A gate at p < .4–.5 would have blocked all three bad closes and allowed all three good ones. Old-init RP12 (closed at 7/8, update 1147) has no loss logs to check.
- Target gates: MH robustness to initialization; unequal_mass (prevents freezing the allocation).
- Risk: an open gate alone just oscillates at 7/8, as DV14–16 do with full mobility. It needs H1's exploration to fill the hole.

**H4: RP15 with repulsion that vanishes when duplicates are justified.**
- Base package: `api-rp15` (optionally with RP12's fields merged in).
- Change: scale the tangent force by min(1, p/p1). Alternatively, measure crowding against a fixed, data-scaled radius instead of the median nearest-neighbor distance.
- Rationale: keeps RP15's MH coverage push (14/24) and removes the bars4 quality floor (§2.4).
- Target gates: bars4, MH.
- Risk: the MH benefit may shrink once the force is gated.

**H5: continuous step multiplier instead of the binary 1% close.**
- Base package: `api-rp12`.
- Change: for each role, m = clip(|EMA Δθ| / EMA|Δθ|, .01, 1) (the step signal-to-noise ratio).
- Rationale: attacks the β1=0 random walk directly (§2.3). Adaptation scales with evidence instead of a 5-step shock detector.
- Target gates: indefinite operation, ring.
- Risk: rotational dynamics shrink EMA Δθ far from equilibrium, which can freeze the game early. Untested.

### 3.4 Cheap attribution probes before building H1–H4

These fill the empty matrix cells that decide which merge is needed. All use the new init. An image run takes about 30 s and a vector run about 25 s on one A6000. They are distinct configs, not seed repeats.

- **P1: RP10 and RP11 on img_bars4.** Does RP12's vertical-only lock come from all_pairs, from uniform_sampled, or from the RP10 base (removed noise)?
- **P2: RP5 on img_bars4 and vector_unequal_mass.** Does startup-noise exploration still pass under the new init? This is a diagnostic only; RP5 is `T`-class.
- **P3: RP12 on vector_unequal_mass and vector_unequal_width.** These are unmeasured. The prediction is an eigenvalue/allocation failure like DV7, because RP12 has no latent width and no noise.
- **P4: RP12 with `noise_policy="initialization"`, `input_noise_std=.5`** (the RP5 noise) on img_bars4 and mode_hold. This tests H1's premise directly, before writing the adaptive controller.
- **P5: DV12 on img_bars4.** DV12 is the only `S`-class mode_hold pass outside the RP family, and it is unmeasured on images.
- **P6: loss logging on RP12 img_bars4 and blobs4** (logging only, identical config and seed). Measures the equilibrium payoff gap p on images, to set H1/H3 thresholds that are not tuned per task.

### 3.5 Mechanisms that consistently hurt (drop them)

| Mechanism (rows) | Evidence |
|---|---|
| Horizon schedules (public K3P/KA2) | Disqualified. They also fail mode_hold (6/8), and K3P never recovers after a change (0/220). |
| Unmodified constant rates (all C rows, ka2-constant, 96 + 79 constant screens) | Oscillation on mode_hold; retention losses: C6 638/692, C7 641/692, C2 296/555, C1 172/183. |
| Bounded, optimistic and extragradient numerics (C2, C3, C4, C8, C9, C12) | Never acquire or acquire far too late (mode_hold 0–3/8); C12 needs 80k field evaluations. |
| rp1 open gain .2 (RP1–RP4) | Image quality too slow (intensity 0–2/24). |
| RP7 two-direction resolvent, RP9 particle exploration, RP13 activation metric | mode_hold collapse (o:0/8) or oscillation (3/8; 6 checks, then loss). |
| all_pairs without uniform weighting (RP11) | 5/8, oscillating. |
| Unbounded or evidence-gated latent smoothing (DV10, DV11) | mode_hold HQ 71% / 22%; minor-component covariance errors of 1.97 (DV10, component 4) and 3.8 (DV11, component 3). |
| DV8 moment-discrepancy mobility | Shifted-ring collapse (min HQ .076, 2 modes). |
| Always-on median-bandwidth repulsion (RP14/RP15) | bars4 quality floor. |
| Coverage-blind locks (precision close, DV brake) as the *only* control | Freeze 7/8 on mode_hold; freeze a bad allocation on vectors. |
| Startup-noise clocks | Ineligible, and their end triggers critic domination on mode_hold (Lg 1.5–3.2). |
| Inert differences within 1200 updates | C7 vs C5, C13-R1 vs C13, RP3 vs RP2, DV6 vs DV5, DV2–4 vs DV1. |
