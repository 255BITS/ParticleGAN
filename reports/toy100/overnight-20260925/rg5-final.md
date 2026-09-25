The coin flip is understood, and a guard called rg5 removes it: grid100 acquisition goes from 6/16 to 16/16 on the four session arms, over the fixed seeds 1234–1237. It also holds ring stability, ring plasticity and 16/16 toys. The base rates come from the retained session-findings runs; three of those runs I re-ran here and they matched bitwise. No jobs are still running. The full write-up is in `result.md`, and all 80 executed gates are in `tests.jsonl`.

**What decides acquisition**
- **Common oscillation.** Every seed enters the same period-2 oscillation of the generator's affine map. This happens as the critic's input noise anneals to zero at step 700, while the network learning rate is still at its peak.
- **The stall event.** A run stalls if and only if one oscillation spike runs away in Adam's updates (runaway = critic penalty above 100× its early median). The runaway reaches the critic within 0–3 steps. The critic's second-moment estimate is then inflated by the spike, so its steps are 25–100× smaller for the rest of the run.
- **Retained runs.** This splits all 16 retained runs cleanly. Stalling runs start their runaway at steps 653–693 (a_r1r2) or 417–439 (b_cap); acquiring runs never exceed 100× their median.
- **No early warning.** Nothing before the event predicts the outcome: losses, penalty, oscillation structure, update size and critic step size all overlap. The acquiring seed 1234 actually had the larger early spike (update ratio 9.3), which turned around on its own.

**The fix (rg5).** It clips each tensor's gradient so its Adam update is at most 5 learning-rate units, after 200 optimizer steps. The critic is clipped unconditionally. A generator tensor is clipped only when its gradient reverses direction from the previous step. The guard reads only the optimizer's own state: no targets, labels or evaluation feedback.

| | base | rg5 |
|---|---|---|
| grid100 acquisition, 4 arms | 6/16 | **16/16** |
| b_cap grid100 full native gate | 2/4 | **4/4 PASS** (live precision .983–.986, center error .122–.127) |
| b_cap staggered100 full native gate | not run | 4/4 PASS |
| ring hold / target shift at floors .1/.1 | 120/120, 81/81 | 120/120, **81/81**, re-acquires in 110 updates |
| 16 focused toys | 16/16 | **16/16**, 14 bitwise identical to the reference |

Live and EMA values are recorded in `tests.jsonl`. Every clip happens before step 670.

**Comparison and ablation**
- **Unconditional version (sg5).** Clipping the generator on every step also acquires 8/8, but fails the ring shift test 80/81. The cause is one 2.2% clip on a steady post-shift step, followed by a single dip 774 updates later; re-acquisition was still 110 updates. So a single run of that ring protocol can flip on a tiny perturbation, and rg5's pass there is one run too.
- **Both halves are needed.** Clipping only the critic on the three stalling control seeds gives [93, 79, 100] modes. Clipping only the generator on the two stalling b_cap seeds gives [48, 68], the base rate.

**What didn't go cleanly**
- **Toys first errored in the harness.** My frozen copy of the code was missing a `reports/` file, so the first toy launch failed in about 3 s each. This was a harness error, not a candidate failure. I fixed it and reran all 16; the failed launches are kept as 17 ERROR rows (mode_hold was logged twice).
- **Two queue restarts.** I stopped one queue to fix the guard's receipt output. I stopped another because the guard wasn't identifying generator vs critic on the ring and toy runs; that fix doesn't change any grid100 result. Both are logged in `logs/queue.log`.

**Not established:** rotated100, the no-guard staggered100 control, the ring shift over several seeds, and rg5 with the b_cap toy config. b_cap's three toy losses come from the reg_arm, not from acquisition, so rg5 doesn't touch them.

**Recommendations**
1. Put rg5 in the base before ranking any native mechanism.
2. Run rotated100 over 4 seeds and the no-guard staggered100 control.
3. Repeat the ring shift test over at least 4 seeds.
4. As a non-guard alternative, test ending the input noise after the learning-rate anneal starts at step 960. It needs its own 4-seed control family.

Everything is in `/ml2/hypergan/gan-attempts/claude-pool-20260925T062531Z/native_acquisition/20260925T131236Z-3408013/`:
- `result.md`
- `tests.jsonl`
- `repo/reports/toy100/native-acquisition-3408013/` — code, `declarations.json`, `out/`, `logs/`, and the replay commands (`run1.sh`, `run_shift.sh`, `run_toy.sh`)
