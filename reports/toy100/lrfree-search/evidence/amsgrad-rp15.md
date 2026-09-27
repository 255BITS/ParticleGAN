# AMSGrad on the RP15 package (PR202 port): results

Date: 2026-09-27. Harness: `/ml2/hypergan/lrfree-20260926` (frozen PR155 new-init hosts, bitwise-validated).

Package: `candidates/rp15-ams/package`. This is API-RP15 plus `Recipe.amsgrad`, passed to every recipe optimizer: G, prior and critic.
- Port diff: `candidates/rp15-ams/amsgrad.diff` (7 files).
- `package_sha256` 5168d04f…, identical for every run below.
- Overrides: `candidates/rp15-ams/overrides-*.json`.
- Review verdict: CORRECT.
  - `amsgrad=false` is bitwise equal to API-RP15.
  - The applied step equals a manual AMSGrad step.
  - shared_tangent_support and the spike guard read `max_exp_avg_sq`.

## Candidates

| cand | config | purpose |
|---|---|---|
| API-RP15 | PR155 RP15 as declared (rp5 precision, secant_resolvent, shared_tangent_support, eager Adam) | baseline. Its ring rows were submitted in this round. |
| rp15-ams0 | port + `amsgrad=false` | identity check |
| rp15-ams | API-RP15 + `amsgrad=true` | main question |
| rp15-ams-rc3 | rp15-ams + `reg_coeff=3.0` | PR202's second arm |
| rp15-const-ams | constant LR: `continuous_precision=null, total_steps=2**62, lr_floor=1, network_lr_floor=1`, noise constant/0, secant + shared_tangent + eager, `amsgrad=true` | can amsgrad replace the controller? |
| rp15-const | the same, `amsgrad=false` | control |
| rp15-rc3 | API-RP15 + `reg_coeff=3.0`, plain Adam | attribution control for rp15-ams-rc3. **Added in this round.** |
| rp15-const-ams-rc3 | rp15-const-ams + `reg_coeff=3.0` (mode_hold only) | does the rc3 mode_hold pass survive without the rp5 close? **Added in this round.** |

The constant-LR recipes validated in this package unchanged. `network_lr_horizon_cap=1600` is inert because both floors are 1.0, and `rates.jsonl` shows G, D and prior at the full .00425/.0085 for every update of every task. No adjustment was needed.

## Matrix

`gates (11)` = the 11 quick gates only. The LEADERBOARD `gates` column also counts ring rows. PR202 stationary cells come from its `-inf` reruns (`total_steps 2**62`); the listed rows hit `ERROR budget`.

| cand | gates (11) | mode_hold | intens2 | blobs4 | stripes2 | bars4 | v.broad | v.mass | v.width | v.aniso | v.overlap | v.spiral | ring_shift | stationary |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| API-RP15 | 10/11 | PASS 14/24 @550 | PASS 5/24 @500 | PASS 17/24 @200 | PASS 21/24 @75 | FAIL 0/24 (3m hq0.72) | PASS 23/24 @100 | PASS 21/24 @200 | PASS 20/24 @250 | PASS 21/24 @200 | PASS 23/24 @50 | PASS 23/24 @67 | PASS @930 143/148 / +370 184/184 | PASS @930 653/658 |
| rp15-ams0 | 2/2 | PASS 14/24 @550 |  | PASS 17/24 @200 |  |  |  |  |  |  |  |  |  |  |
| rp15-ams | 6/11 | FAIL 0/24 (7m hq1) | FAIL 6/24 @425 (2m hq1) sfx4 | FAIL 2/24 @225 (4m hq0.88) | PASS 20/24 @100 | FAIL 0/24 (4m hq0.78) | PASS 22/24 @150 | PASS 21/24 @200 | PASS 20/24 @250 | PASS 21/24 @200 | FAIL 22/24 @50 sfx4 | PASS 24/24 @67 | PASS @870 154/154 / +530 168/168 | PASS @870 664/664 |
| rp15-ams-rc3 | 9/11 | PASS 6/24 @950 | FAIL 6/24 @425 (2m hq0.97) sfx1 | PASS 15/24 @250 | PASS 17/24 @200 | PASS 7/24 @450 | PASS 22/24 @150 | PASS 21/24 @200 | PASS 19/24 @300 | PASS 21/24 @200 | FAIL 21/24 @50 sfx2 | PASS 24/24 @67 | FAIL no arrival / +500 159/171 | PASS @4020 349/349 |
| rp15-rc3 | 7/11 | FAIL 4/24 @800 (8m hq0.89) | FAIL 2/24 @450 (2m hq0.94) sfx1 | PASS 17/24 @200 | PASS 21/24 @75 | FAIL 0/24 (4m hq0.81) | PASS 21/24 @200 | FAIL 0/24 [c.min_eigen_ratio] | PASS 19/24 @300 | PASS 21/24 @200 | PASS 23/24 @50 | PASS 24/24 @67 |  |  |
| rp15-const | 9/11 | FAIL 9/24 @550 (8m hq0.81) | PASS 5/24 @500 | PASS 17/24 @200 | PASS 21/24 @75 | FAIL 0/24 (3m hq0.72) | PASS 23/24 @100 | PASS 21/24 @200 | PASS 20/24 @250 | PASS 21/24 @200 | PASS 23/24 @50 | PASS 23/24 @67 | PASS @930 140/148 / +370 182/184 | PASS @930 587/658 |
| rp15-const-ams | 6/11 | FAIL 0/24 (6m hq0.52) | FAIL 6/24 @425 (2m hq1) sfx4 | PASS 15/24 @225 | PASS 20/24 @100 | FAIL 0/24 (4m hq0.78) | PASS 22/24 @150 | FAIL 20/24 @200 sfx4 | PASS 20/24 @250 | PASS 21/24 @200 | FAIL 22/24 @50 sfx4 | PASS 24/24 @67 | PASS @870 154/154 / +30 218/218 | PASS @870 664/664 |
| rp15-const-ams-rc3 | 0/1 | FAIL 4/24 @1050 (8m hq1) sfx4 |  |  |  |  |  |  |  |  |  |  |  |  |
| pr202-const-noams | 6/11 | FAIL 0/24 (3m hq0.33) | PASS 12/24 @275 | PASS 10/24 @375 | PASS 22/24 @75 | FAIL 0/24 (2m hq0.81) | PASS 22/24 @100 | FAIL 0/24 [c.covariance_error,c.min_eigen_ratio] | FAIL 6/24 @150 [c.min_eigen_ratio] | FAIL 19/24 @200 sfx4 | PASS 20/24 @50 | PASS 24/24 @67 | PASS @500 132/191 / +230 145/198 | ERROR budget |
| pr202-const-ams | 7/11 | FAIL 0/24 (0m hq0) | PASS 12/24 @325 | PASS 11/24 @350 | PASS 22/24 @75 | FAIL 0/24 (1m hq0.5) | PASS 23/24 @100 | PASS 8/24 @650 | FAIL 8/24 @150 [c.min_eigen_ratio] | FAIL 20/24 @200 sfx4 | PASS 22/24 @50 | PASS 24/24 @67 | PASS @550 186/186 / +480 170/173 | ERROR budget |
| pr202-const-ams-rc3 | 6/11 | FAIL 0/24 (5m hq0.92) | PASS 15/24 @200 | PASS 9/24 @375 | PASS 18/24 @175 | FAIL 0/24 (3m hq0.94) | PASS 24/24 @50 | FAIL 15/24 @300 [c.min_eigen_ratio] | FAIL 4/24 @200 [c.min_eigen_ratio] | FAIL 8/24 @150 sfx1 | PASS 22/24 @50 | PASS 24/24 @67 | PASS @520 189/189 / +530 168/168 | ERROR budget |

PR202 stationary from its `-inf` reruns:
- pr202-const-noams-inf: PASS @500 574/701
- pr202-const-ams-inf: PASS @550 696/696
- pr202-const-ams-rc3-inf: PASS @520 699/699

Not run: rp15-rc3 ring (skipped) and the rest of rp15-const-ams-rc3 (mode_hold only).

## Key structural fact: rp5 does nothing until it closes

rp15-ams and rp15-const-ams are **bitwise identical until the first precision close**, as the per-task comparisons show:

| task | close in rp15-ams | where the two runs diverge |
|---|---|---|
| bars4, intensity2, overlap | never closes | identical over all 600/1200 updates |
| mode_hold | 1109 | first differing observation 1150 |
| blobs4 | 285 | 300 |
| stripes2 | 331 | 350 |
| v.broad | 908 | 950 |

So the rp5 controller is observation-only while open. `ams` vs `const-ams` isolates exactly the effect of the 100x cut, and `API-RP15` vs `rp15-const` isolates it for plain Adam.

Precision close step (first G LR < .5 x lr0, from `rates.jsonl`):

| task | API-RP15 | rp15-ams | rp15-ams-rc3 |
|---|---:|---:|---:|
| mode_hold | 788 | **1109** | 942 |
| blobs4 | 309 | 285 | 303 |
| stripes2 | never | 331 | 369 |
| intensity2, bars4, overlap | never | never | never |
| v.broad / v.mass / v.width / v.aniso / v.spiral | 887 / 339 / 407 / 396 / 996 | 908 / 348 / 411 / 386 / 985 | 921 / 348 / 402 / 377 / 1004 |
| ring_shift (close, reopen, close) | 993, 2465, 3162 | 1116, 2453, 3275 | **755**, 2475, 3323 |

amsgrad barely moves the close timing on the vectors (±20 updates). It moves it on mode_hold (+321) because arrival there is slower. rc3 makes the ring close early (755).

## Did amsgrad fix img_bars4 without breaking mode_hold?

**No.** With the controller kept (rp15-ams) the score drops from 10/11 to 6/11.

- **bars4 improves but still fails:** 3m/HQ .72 → 4m/HQ .78, 0/24.
  - The rp5 controller never closes on bars4 in any variant, so the run is at full LR for all 600 updates. The bars failure is still not an LR-cut problem.
  - HQ is still rising at 600: .69 @425, .75 @475, .78 @575. Secant `norm_ratio` stays at .35-.45, a 2-3x step shrink. The run is simply too slow.
- **mode_hold breaks:** PASS 14/24 @550 → FAIL 0/24, final 7/8 HQ 1.0.
  - The chaotic search phase lasts until 650 (0-3 modes, G applied step .2-.4) instead of settling at 500.
  - It then locks into 7 modes at 700. rp5 closes at 1109 instead of 788 and freezes the 7-mode state.
  - During 700-1100 the secant still sees strong rotation: `rotation_squared` 1-6.5 against .2-.5 for plain Adam, and D's preview step is 2-3x larger.
  - The G step is small (applied L2 .005-.014). The G has no push left to capture the 8th mode.
- **blobs4 breaks:** PASS 17/24 → FAIL 2/24.
  - rp5 closed at 285 while HQ was oscillating at 28-29 of 32 particles (.88/.91). The state froze at .88 at 1% LR.
  - Baseline: closed at 309 at HQ .97. With constant LR (rp15-const-ams), the same trajectory keeps refining and passes 15/24.
- **intensity2 is a marginal fail:** 6/24, suffix 4. There is a dip to .84 at 475-500, then 1.0 from 525. The run is at full LR throughout.
- **overlap is a marginal fail:** 22/24, suffix 4. One `mean_error` excursion of .213 > .15 at 1000, at full LR. rp5 never closes on overlap, and the baseline's excursions peak at .142.
- ring_shift passes: @870 154/154, change delay +530 at 168/168. API-RP15: @930 143/148, +370 at 184/184.
- stationary passes: @870 664/664 with no reopen after the close at 1116. API-RP15: @930 653/658.

**With rc3 (rp15-ams-rc3): 9/11, bars4 and mode_hold both pass.**
- bars4: PASS 7/24 @450, 4m HQ .94.
- mode_hold: PASS 6/24 @950. The pass is fragile. At full LR the run reaches 8m at 600-650, collapses to 1m at 700 and 800, then returns to 7m at 900. rp5 closed at 942 and happened to catch 8/8, which then holds at HQ .92→1.0.
- Remaining quick-gate failures:
  - intensity2: 6/24, suffix 1, HQ oscillating .84-.97 at full LR.
  - overlap: 21/24, suffix 2, one `mean_error` .225 at 1100.
- ring_shift **FAILS**: no first-segment arrival.
  - rc3 slows the ring (HQ .50 @600 vs .64 without rc3), and rp5 closes at **755 while HQ ≈ .70**.
  - At 1% LR, HQ creeps .72 → .84 by 2400 and never reaches .90.
  - The changed target arrives at +500 with 5 departures (159/171).
- stationary: PASS only at **@4020**, 349/349. After the close at 755 it takes about 3300 updates at 1% LR to reach HQ .90.
- **The bars4 and mode_hold passes need amsgrad and rc3 together.** rc3 alone (rp15-rc3) fails both; see the next section.

## Can amsgrad replace the LR controller? (constant LR)

**Only on the ring tasks.** On the quick gates it does not: rp15-const-ams is 6/11 vs 9/11 for the plain-Adam constant control.

| | rp15-const (plain) | rp15-const-ams |
|---|---|---|
| mode_hold | 8/8 @500-950 (9 passes), then the equilibrium **breaks at 1000** (6m .66 → 2m .10 @1100 → 8m .81 @1200) | never 8 modes; 7m 700-1150, then 6m .52 @1200 |
| G applied L2 at 850 / 900 / 950 (snapshots) | .011 / .021 / .030, rising ~3x just before the break (consistent with the beta2 creep) | .009 / .014 / .005, flat (creep suppressed) |
| ring_shift | PASS @930 140/148 (2 departures), +370 182/184 (1 departure) | **PASS @870 154/154, +30 218/218**, 0 departures |
| stationary | PASS @930 587/658; **5 departures** (950, 1090, 2720, 5830, 6730); min HQ .001 after arrival (full collapses) | **PASS @870 664/664**, 0 departures in 6630 updates at full LR |
| v.mass | PASS 21/24 | FAIL 20/24 suffix 4: component min-eigen ratio dips to .078 @1000 (hovering .17-.37 vs .57-.92 plain) |
| blobs4 | PASS 17/24 @200 (same score as API-RP15) | PASS 15/24 @225 (better than rp15-ams, which froze at .88 after its close at 285) |

Reading:
- **amsgrad does what PR202 claims on the ring.**
  - Plain Adam's decayed `v` gives large steps when the target moves; recovery takes 370 updates.
  - The AMSGrad max keeps the step bounded, and the generator tracks the +[1,0] shift within 30 updates, with zero departures over 6630 stationary updates.
  - This matches PR202 K3P (ring 186/186, stationary 696/696 vs plain 132/191, 574/701).
- **On the 12-particle mode_hold and on component shape, amsgrad at full LR is worse, not better.**
  - The permanent max over the early, large-gradient transient cuts the G step afterward: 7 modes captured at 700 vs 8 at 500 for plain Adam.
  - This is an irreversible, history-set LR cut, not an equilibrium stabilizer.
  - The run still loses modes at 1150-1200, with D rotation high throughout.
  - PR202 K3P (0/8) and dv12-ams (mode_hold 6/24 @950 vs 12/24) show the same slowdown.
- Plain constant LR in RP15 is already 9/11 (mode_hold and bars4 fail). The controller's only quick-gate value is mode_hold, where it freezes the 8/8 state at 788, before the break at 1000.

## rp15-rc3 attribution control and rp15-const-ams-rc3

**rp15-rc3 (API-RP15 + reg_coeff 3, plain Adam): 7/11. rc3 alone does not fix bars4 and breaks mode_hold.**
- bars4: FAIL 4m HQ .81.
  - rp5 closes at 507 while HQ = .81, and the state freezes there.
  - With amsgrad added (rp15-ams-rc3), rp5 never closes on bars4, and HQ keeps rising to .94 by 450.
  - The rc3 run is faster than rc1 (4 modes by 275 vs ~425-575). amsgrad keeps the controller open long enough to finish the job.
- mode_hold: FAIL 4/24 @800, final 8m .89.
  - rp5 **never closes** here, so the run stays at full LR for all 1200 updates.
  - It oscillates: 8m .91 @800-850, 6m @900, 4m .22 @1000, 8m .89 @1200.
  - With amsgrad, the gap contraction and activity drop that rp5 needs occur, and it closes at 942.
- v.mass: FAIL, `component_min_eigen_ratio` ≈ .05 from update 150 on (a flattened component). amsgrad + rc3 passes 21/24.
- intensity2: FAIL sfx1 (2/24). This is the same full-LR wobble as the amsgrad variants.
- Close steps: blobs4 324, stripes2 307, bars4 507, v.broad 855, v.mass 374, v.width 422, v.aniso 385, overlap 328, spiral 985. mode_hold and intensity2 never close.

**Synergy.** amsgrad + rc3 (9/11) beats rc3 alone (7/11) and amsgrad alone (6/11). Both are below API-RP15 (10/11), which fails only bars4. The two knobs interact through the rp5 close decision, not through a stable fixed point:
- mode_hold closes only with both knobs, at 942.
- On bars4, rc3 alone closes too early, while with both knobs it never closes.

**rp15-const-ams-rc3 (constant LR, mode_hold only): FAIL 4/24 @1050, sfx4, final 8/8 HQ 1.0.**
- The run is bitwise identical to rp15-ams-rc3 up to 942.
- At full LR it goes 8m .70 @950, 8m .58 @1000, then 8m .96-1.0 from 1050 to 1200.
- It is one observation short of the 5-check suffix. This is the closest any constant-LR config has come on mode_hold here.
  - rp15-const: 8/8 from 500, broke at 1000.
  - rp15-const-ams: never 8.
  - PR202 const: 0-5/8.
- Whether 8/8 holds after 1200 is unknown, because mode_hold is fixed at 1200 updates.

**Diagnosis of the residual failures of rp15-ams-rc3**
- **intensity2 and overlap never close in any RP15 variant.**
  - intensity2: G activity stays at 22-35% of its decaying peak, but `gap_s` keeps rising (.0065 → .027, velocity > 0), so the contraction condition never holds.
  - overlap: `gap_s` contracts, but activity stays at 55-70% of its peak (.27-.34 vs peak .45-.59). The 50%-of-peak activity condition never holds.
  - Both runs stay at full LR, and the final-suffix rule fails on one or two dips: intensity2 HQ .84 at 475-500 and 575; overlap `mean_error` .22 at 1100.
  - The baseline passes intensity2 with exactly the minimum suffix (5/24), so this gate is marginal for every full-LR RP15 variant.
- **ring_shift closes too early.** rc3 slows ring learning, and rp5 closes at 755 while HQ ≈ .70. The binary 100x cut then needs 3300 updates to arrive (stationary @4020) and misses the 2400 window.
- In both cases the binary, quality-blind rp5 switch is the failure point, not amsgrad. The same premature-close failure appears in rp15-ams blobs4 (285, frozen at .88) and rp15-rc3 bars4 (507, frozen at .81).

## What to try next (concrete, no seed variants)

Ordered by expected value. All are single deterministic configs.

1. **Make the rp5 cut soft (code, small).**
   - Add `Recipe.precision_closed_scale` (default `.01`; prior closed level = `5 * scale` capped at 1) and wire it into `precision.py:23-25`, which currently hardcodes .01/.05.
   - Run `rp15-ams-rc3` + `{"precision_closed_scale": 0.1}`, and a second run at `0.03`.
   - Rationale:
     - Every rp5 failure here is a premature freeze: ring 755 @ HQ .70, blobs4 285 @ .88, rc3 bars4 507 @ .81.
     - At full LR amsgrad already keeps the ring stable (rp15-const-ams: 0 departures in 7500 updates, +30 delay), so the cut no longer has to be 100x.
     - A 10x cut should still damp the mode_hold oscillation while letting frozen states finish refining.
   - Run `all` (gates + ring). With `amsgrad=false` and default scale the port stays bitwise equal to API-RP15.
2. **Finish rp15-const-ams-rc3:** `--tasks all` (only mode_hold was run).
   - This is the cleanest "no LR adjustment" candidate: constant LR, no controller, no horizon.
   - Known in advance, because rp5 is inert until it closes:
     - intensity2 FAIL sfx1, overlap FAIL sfx2, bars4 PASS, all identical to rp15-ams-rc3.
     - mode_hold 8/8 HQ 1.0 from 1050, one check short.
   - Unknowns: ring (full LR without the premature close), blobs4, stripes2, and the vectors (v.mass eigen ratio at constant LR).
3. **Per-player amsgrad (code, small).** Let `Recipe.amsgrad` accept `"critic"` / `"generator"` in addition to bool, and try `amsgrad="critic"` on rp15-ams-rc3 and rp15-const-ams-rc3.
   - Evidence that the G-side max costs mode capture:
     - After the 250-650 transient, G's applied step is 1.5-3x smaller.
     - 7 modes by 700 vs 8/8 by 500 for plain Adam.
     - dv12-ams shows it too: mode_hold 6/24 @950 vs 12/24 @650.
   - The ring benefit is the D/G creep, which PR202 attributes to both players. This test says which player carries it.
4. **Change the close rule for noisy plateaus (code, rp5 variant `rp6`).**
   - intensity2 and overlap never close, for opposite reasons: rising gap in one, activity at 55-70% of peak in the other.
   - Replace "activity < 50% of decaying peak" with a plateau test: slow EMA of activity flat, |velocity_s| below a fraction of the quiet reference for N updates.
   - Allow closing at `gap_s` velocity ≈ 0 as well as < 0.
   - Do this only after (1). With a soft cut, closing more often is cheap.
5. **Not recommended:**
   - more `reg_coeff` values (t2-rp15reg-* already covers rc1.5/rc2 on plain Adam);
   - amsgrad with the default rp5 on its own (6/11);
   - amsgrad at rc1 constant LR (6/11; mode_hold never reaches 8).

## Files

- Runs: `runs/{API-RP15,rp15-ams0,rp15-ams,rp15-ams-rc3,rp15-const-ams,rp15-const,rp15-rc3,rp15-const-ams-rc3}/<task>/` (`metrics.jsonl` includes `diag.precision`, `diag.game_stats.secant`, and the generator metric, which reads `native_amsgrad_existing_uncorrected_max_second_moment`).
- Overrides: `candidates/rp15-ams/overrides-{rp15-ams0,rp15-ams,rp15-ams-rc3,rp15-const-ams,rp15-const,rp15-rc3,rp15-const-ams-rc3}.json`.
