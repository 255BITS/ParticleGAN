# AMSGrad on DV12 (PR202 port): results and diagnosis

*2026-09-27, harness `/ml2/hypergan/lrfree-20260926`. Analysis script: `scratchpad/dv12amsgrad-an/{dvtraj,ringsum}.py` (not part of the harness).*

## 0. Answers

- **Did amsgrad fix `vector_unequal_mass` without breaking mode_hold? No.**
  - **Unequal mass is unchanged.** `dv12-ams` has the same applied-LR trace as API-DV12 (G<.5 @322 vs 321, <.1 @720 vs 717). The particle allocation is also the same: frozen at [.602,.339,.055,.005] from update 450.
    - min mass ratio is .2319 in both runs.
    - Covariance error got slightly worse (.920 vs .879).
  - **mode_hold still passes, but it is weaker.** 6/24 @950 with final HQ .909, vs 12/24 @650 with HQ .983.
  - **Two new failures appear:** img_bars4 (2/4 modes) and img_intensity2 (a one-check HQ blip). The net result is **10/13 vs 12/13** for API-DV12.
  - **`dv12-ams-rc3` does pass v.mass and mode_hold, but reg_coeff=3 is what does it.** The plain-Adam control `dv12-rc3` gives the same particle allocation and the same v.mass PASS 18/24 @350. It also passes mode_hold (6/24 @950).
  - Both rc3 arms fail img_intensity2, so each is **12/13**, the same count as API-DV12. They swap the v.mass failure for an intensity2 failure.
- **Can amsgrad replace the LR controller? No.** In the constant-LR DV12 arm `dv12-const-ams`, the controller still runs but never scales an LR. In the 4 tasks that ran, it failed mode_hold, img_intensity2 and img_stripes2, and passed img_blobs4.
  - PR202's constant-LR K3P arms also fail mode_hold (0/8 and 5/8).
  - What amsgrad actually buys at constant LR is **retention**:
    - PR202 stationary: 696/696 with amsgrad vs 574/701 without (9 departures).
    - PR202 ring: 186/186 vs 132/191.
  - That retention does not help the HQ/coverage gates, which need the step to shrink by 2-10x near equilibrium.
- **Next steps:** drop amsgrad from the DV12 line and attack the v.mass allocation freeze directly (§5).

## 1. Matrix

Cells show harness results: PASS/FAIL, passing checks/24 and first arrival. Ring cells show `arrival checks | +post-shift delay checks`.

| task | API-DV12 | dv12-ams | dv12-ams-rc3 | dv12-rc3 (control, plain Adam) | dv12-const-ams (partial) | pr202-const-ams | pr202-const-ams-rc3 | pr202-const-noams |
|---|---|---|---|---|---|---|---|---|
| mode_hold | PASS 12 @650 (8m .983) | PASS 6 @950 (8m .909) | PASS 9 @800 (8m .955) | PASS 6 @950 (8m .94) | **FAIL** 1 @1200 (8m .91, sfx1) | FAIL 0 (0m) | FAIL 0 (5m .92) | FAIL 0 (3m .33) |
| intens2 | PASS 6 @425 | **FAIL** 8 @375 sfx4 (2m .97) | **FAIL** 0 (2m .84) | **FAIL** 2 @475 (2m .88) | **FAIL** 5 @450 (2m .81) | PASS 12 @325 | PASS 15 @200 | PASS 12 @275 |
| blobs4 | PASS 18 @175 | PASS 19 @150 | PASS 17 @200 | PASS 17 @175 | PASS 17 @125 | PASS 11 @350 | PASS 9 @375 | PASS 10 @375 |
| stripes2 | PASS 21 @100 | PASS 21 @100 | PASS 22 @75 | PASS 22 @75 | **FAIL** 20 @100 sfx3 | PASS 22 @75 | PASS 18 @175 | PASS 22 @75 |
| bars4 | PASS 5 @500 | **FAIL** 0 (2m .94) | PASS 15 @250 | PASS 15 @200 | parked | FAIL 0 (1m .5) | FAIL 0 (3m .94) | FAIL 0 (2m .81) |
| v.broad | PASS 22 @150 | PASS 23 @100 | PASS 22 @150 | PASS 22 @150 | parked | PASS 23 @100 | PASS 24 @50 | PASS 22 @100 |
| v.mass | **FAIL** 0 [cov .879, mmr .232] | **FAIL** 0 [cov .920, mmr .232] | PASS 18 @350 | PASS 18 @350 | parked | PASS 8 @650 | FAIL 15 [eig] | FAIL 0 [cov, eig] |
| v.width | PASS 21 @200 | PASS 21 @200 | PASS 19 @300 | PASS 19 @300 | parked | FAIL 8 [eig] | FAIL 4 [eig] | FAIL 6 [eig] |
| v.aniso | PASS 21 @200 | PASS 21 @200 | PASS 22 @150 | PASS 22 @150 | parked | FAIL 20 sfx4 | FAIL 8 sfx1 | FAIL 19 sfx4 |
| v.overlap | PASS 23 @50 | PASS 23 @50 | PASS 23 @50 | PASS 21 @50 | parked | PASS 22 @50 | PASS 22 @50 | PASS 20 @50 |
| v.spiral | PASS 24 @67 | PASS 24 @67 | PASS 24 @67 | PASS 24 @67 | parked | PASS 24 @67 | PASS 24 @67 | PASS 24 @67 |
| ring_shift | PASS @600 181/181, +440 177/177 | PASS @580 183/183, +320 188/189 | PASS @660 175/175, +300 190/191 | PASS @650 175/176, +380 181/183 | parked | PASS @550 186/186, +480 170/173 | PASS @520 189/189, +530 168/168 | PASS @500 132/191, +230 145/198 |
| stationary | PASS @600 686/691 | PASS @580 673/693 | PASS @660 685/685 | PASS @650 652/686 | parked | ERROR budget (inf: 696/696) | ERROR budget (inf: 699/699) | ERROR budget (inf: 574/701) |
| **gates** | **12/13** | **10/13** | **12/13** | **12/13** | 1/4 done | 8/12 | 7/12 | 7/12 |

Reference row, API-RP15: 10/11 quick gates (fails only bars4, 3m .72). Ring: PASS @930 143/148, +370 184/184. Stationary: PASS @930 653/658.

**Identity checks.**
- dv12-ams0 (amsgrad=false) is bitwise equal to API-DV12 on 3 tasks (review).
- dv12-const-id (new `continuous_lr_scales` flag left True, with amsgrad) is bitwise equal to dv12-ams on mode_hold: 24/24 observation rows and 1200/1200 LR rows.

**New candidates from this analysis:**

| candidate | purpose | package / change |
|---|---|---|
| `API-DV12` ring/stationary | baseline, so the ring cells above can be attributed | — |
| `dv12-rc3` | attribution control | API-DV12 package with reg_coeff=3 |
| `dv12-const-ams` / `dv12-const` | constant-LR DV12 arms | `candidates/dv12-const/package` = dv12-ams package plus one flag. `Recipe.continuous_lr_scales` (default True, bitwise). When False, the DataDriftController still observes (latent perturbation, KA2 alpha·gt, diagnostics) but G/D/prior stay at .00425/.00425/.0085 and critic_scale is skipped. |

**Parked jobs.** At 00:46 another process moved 9 dv12-const-ams jobs and all 13 dv12-const jobs into `queue-parked/`. They have not run. To run them, move the files back into `queue/`.

## 2. How DV12's controller behaved per task (and what amsgrad changed)

DV12 target: m ← max(data_drive, min(1, pe²)), rising at .05 and falling at .005. The applied rates are:
- G = (.01+.99m)·gt
- D = G-scale/(1+pe²)

G LR crossings (fraction of .00425):

| task | API-DV12 G<.5 / <.1 / final | dv12-ams | dv12-ams-rc3 | dv12-rc3 |
|---|---|---|---|---|
| mode_hold | 648 / 917 / .066 | **871 / never / .273** | 542 / 828 / .060 | 696 / 888 / .083 |
| intens2 | 244 / – / .221 | 241 / – / .214 | 237 / – / .118 | 238 / – / .124 |
| bars4 | never / – / .723 | **never / – / 1.000** | 346 / – / .157 | 328 / – / .148 |
| v.mass | 321 / 717 / .043 | 322 / 720 / .043 | 162 / 508 / .015 | 162 / 508 / .015 |
| v.broad/width/aniso/overlap/spiral | 141-163 / 479-503 / .005-.013 | identical ±1 update | ~same | ~same |

**On the vectors DV12 behaves like a clock.**
- data_drive is 0 and pe² is about .03, so mobility falls at the fixed .005 rate. The cut lands at the same step on every card: G<.5 at 141-163, half-life about 140 updates.
- amsgrad cannot move that timing because pe stays small.

**amsgrad raises the payoff error only where the game is not yet settled:**
- mode_hold: pe plateau .44-.48 vs .15-.17.
- bars4: pe 1.3-2.1 for the whole run vs a drop to .30 after update 475.

A higher pe raises mobility's floor (pe² ≈ .2), so the controller cuts **later and less** on mode_hold and **never** on bars4. There, D stays throttled to .19-.37 by 1/(1+pe²) while G runs at full LR.

## 3. Failing gates, diagnosed

### 3.1 v.mass, API-DV12 and dv12-ams: allocation freeze (amsgrad-neutral)

Both runs have identical trajectories.
- **Allocation.** Components 1 and 2 are occupied from update 50. Component 3 is reached at about 250 (mass .055) and component 4 at about 450 (mass .005).
- **Freeze.** After 450 the masses never change: [.602,.339,.055,.005]. That is about 1 of 256 particles in the 2% component; the gate needs ≥1.28. G LR is already .28 at 450, then .09 at 750 and .04 at 1200.
- **What fails.**
  - min_mass_ratio: .232, below .25.
  - Covariance error: the mean over components is dominated by the under-filled ones. Component 4 is at 2.5-2.9, component 3 at .73-.79 (ams .79). The mean is .879 in the baseline and .920 with amsgrad.

The comparison runs show what fixes it:
- **RP15** fills all 4 components by update 50 ([.58,.25,.14,.03]), before its LR cut at 339.
- **PR202 constant LR** keeps migrating at full LR: component 4 goes .005 → .012 → .022 by 850.
- **rc3 (with or without amsgrad)** fills them by 150 ([.548,.357,.080,.014]), before mobility decays.

So the failure is **particle reallocation being cut off by a clock-like mobility decay**, not second-moment creep.

### 3.2 bars4, dv12-ams: D-dominant lock (amsgrad-caused)

- **Both runs start the same way.** Until about 425, pe is 1.2-2.3, m is pinned at 1, G runs at 1.0 and D at about .2. Only 2 bars are covered.
- **Baseline.** A reshuffle at 475-525 recovers D (.34 → .66, pe 2.1 → .66) and brings in modes 3 and 4. It passes 5/24 @500, a late and fragile pass.
- **dv12-ams.** pe never drops (1.78 at 600), so there is no reshuffle. It stays at 2 modes (fractions .47/.41/.09/.03) with HQ .94.
- **rc3.** pe falls to .77 by 175 and all 4 modes are covered by 275: PASS 15/24 in both rc3 arms.

### 3.3 intens2, dv12-ams: one-check blip (borderline)

The LR trace is identical to the baseline. The run arrives earlier (375 vs 425) and has more passing checks (8 vs 6). It fails only because HQ is .81 at update 500 (6 of 32 particles above rmse .1), which leaves a suffix of 4 instead of 5. This is 1/32 quantization, not a mechanism.

### 3.4 intens2, rc3 arms: LR cut before quality converges (reg_coeff-caused)

With rc3, pe drops to about .19 early, so mobility decays and G is .12 at 600. HQ stalls at .84-.88. The plain-Adam dv12-rc3 fails the same way (.88).

### 3.5 mode_hold, dv12-ams: still passes, but the cut arrives late

HQ only crosses .9 once G LR is at or below about .45, in every DV12 variant:
- baseline: 650 at G .50
- ams: 950 at G .43
- rc3-ams: 800 at G .19
- rc3: 950 at G .23

amsgrad delays that point by 300 updates and ends with effective modes 6.26 vs 7.54. With rc3 it goes the other way: the cut comes earlier (542 vs 696) and there are 9 vs 6 passing checks. At rc3, amsgrad is noise-level.

### 3.6 Ring and stationary: all pass

- **Data-drive reopen works in every arm.** G goes from .03 to .43 in 10 updates after the shift and reaches 1.0 by 2600.
- **ring_shift.** Post-shift arrival is +320 with amsgrad vs +440 in the baseline, with one post-shift departure (min HQ .69).
- **stationary, dv12-ams.** 673/693, with one slow drift: HQ .89 → .81 → .90 over 2620-2810 at G .01-.04. The baseline has 686/691, also with one departure (min HQ .75).
- **stationary, rc3.** 685/685 with amsgrad; plain dv12-rc3 has 652/686.
- **Runtime cost.** DV12 ring jobs take 1.2-2.3 h under load. dv12-ams stationary took 8104 s against the 9000 s timeout, so ring runs for DV12 variants should be given more timeout or scheduled when the pool is idle.

### 3.7 Constant LR: dv12-const-ams

- **mode_hold.** pe is 1.1-2.2 for all 1200 updates, so D dominates. It reaches 8/8 with HQ .91 only at the final observation, with effective modes 4.9. For comparison, the controller would have cut D by about 3x.
- **intens2.** HQ oscillates between .38 and .97 at full LR.
- **stripes2.** A single dropout at 525 (1 mode, HQ .62) leaves a suffix of 3.

amsgrad stops the step from growing but does not make it shrink. The HQ gates need the shrink.

## 4. Caveat from the port review

With amsgrad on, dv12-ams's spike guard measures against `max_exp_avg_sq`, so it clips less often than PR202's guard, which reads `exp_avg_sq`. This is an untested difference from PR202. It does not affect the conclusions: amsgrad is either inert (vectors, rc3) or harmful through pe (mode_hold, bars4), and the guard can only clip less.

## 5. Next steps (no seed variants)

1. **Drop amsgrad from the DV12 line.** It is net -2 gates at rc1 and changes nothing at rc3. The t2-dv12* sweeps built on dv12-ams-rc3 carry over to plain dv12-rc3; rebase them to avoid depending on PR202.
2. **Do not rely on reg_coeff.** The rc probes from another job show a narrow window:
   - rc2.0 and rc2.5 fail mode_hold (both with and without amsgrad).
   - rc2.5 and t2 rc3.5 fail v.mass covariance.
   - rc3 fails intens2.
3. **Fix v.mass at rc1 (API-DV12, which passes everything else including ring and stationary) by keeping the prior mobile while the networks anneal.** These are state-driven, with no clock:
   - **Code, `dv12-pf{25,50}`.** Add `Recipe.continuous_prior_floor` (default .05, bitwise). In `DataDriftController.current_scales`, compute the prior scale as (f + (1-f)·m)·gt with f = .25 or .5. Run all gates plus ring.
   - **Config only, `dv12-plm4`.** API-DV12 with prior_lr_mult 2 → 4. This also speeds up early prior motion, so watch mode_hold HQ.
4. **If 3 is not enough, remove the clock-like decay.** Add a state term to the mobility target: target = max(dd, pe², prior_motion), where prior_motion is an EMA of the mean particle displacement per update divided by the DV12 latent bandwidth, clipped to [0,1]. The LR then stays up while particles are still moving between components and falls only once the allocation has settled.
5. **Constant LR is not viable with amsgrad alone.** Every passing mode_hold config cuts its LR, including the survivors in this report. amsgrad is only worth keeping as a retention aid on top of a state-driven cut.
