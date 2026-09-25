Yes, one critic rule can carry both ends, but it needs a critic-only guard as well. The penalty handles precision; the guard handles acquisition. P1 passes 16/16 focused toys and 11 of 12 native runs over the declared seeds 1234-1237. The one native failure is an acquisition failure.

**The rule (P1):** penalty = s·a_r1r2 + (1−s)·b_cap. s is set only by the critic's own applied learning rate: it stays at 1 until that rate falls to half its peak, then goes to 0 at the .01 network floor. At s=1 it is exactly a_r1r2 and at s=0 exactly b_cap. The guard clips any critic gradient more than 5× Adam's running scale, but only on tensors with at least 200 prior steps. The rule does not look at task names and uses the same arm on every host.

**What each arm does:**
- **a_r1r2 saves the three toys.** Its zero-centered real-side term keeps the critic flat at the data, which damps allocation while the critic's learning rate is high. P1 is bit-identical to a_r1r2 for the first ~80% of every toy budget, and all three b_cap-lost toys pass with a_r1r2's exact passing suffixes. So those losses happen early, and a b_cap tail at the end doesn't bring them back.
- **b_cap lifts precision.** Once the critic's learning rate is small, it lets the critic keep a slope at real data, which pulls samples onto mode centers. On grid100, P1 is exact a_r1r2 for 1,284 critic steps and pure b_cap from step 1,601. It ends at .9870-.9886 precision, against .9587/.9607 for a2 + a_r1r2 on the seeds that acquire.
- **Acquisition is decided before the handover, and neither penalty fixes it.** Without the guard, seeds 1234 and 1236 collapse to 25 and 31 modes. Up to step 1,250 those runs are bit-identical to the session a2 + a_r1r2 runs. The guard fires only on those two seeds, at the step the runaway starts (~655).

| | three lost toys | 16 toys | grid100 | staggered100 | rotated100 | ring hold | ring shift recovery |
|---|---|---|---|---|---|---|---|
| **P1 (rule + guard)** | 3/3 | 16/16 | **4/4** | 3/4 | 4/4 | PASS 120/120 | FAIL 22/81 |
| P2 (rule only) | same as P1 (no clips) | same as P1 on 15 (img_intensity2 not run) | 2/4 | – | – | – | – |
| a_r1r2 control | 3/3 | 16/16 | 0/4 | – | – | PASS 120/120 | FAIL 0/81 |
| b_cap control | 0/3 | 12/15 | 1/4 with a2, 2/4 with no-op latent | – | – | – | – |

I ran the a_r1r2 ring control myself. The other control rows are the session-findings results for the same seeds. My copy of the harness reproduces those references bit-for-bit.

**What P1 holds and loses:**
- **Toys:** holds all 16.
- **Natives:** every run that acquires all 100 modes passes precision. It loses staggered100 seed 1235 at 96/100 modes: the guard clipped the critic at steps 643-647 but couldn't stop the excursion. An earlier attempt's notes place its start in the generator, which is outside this lane; I did not measure that here.
- **Stability:** holds the ring's post-convergence hold. However, every native run dips mid-run to .85-.90 live precision at steps 3000-4500, while the prior learning rate is still at full rate. It recovers during the anneal, so the terminal gate passes. Neither b_cap alone nor a2 + b_cap dips.
- **Plasticity:** fails at the base floors, though better than the parent (22/81 checks against 0/81).

**Recommendations:**
- The generator-side runaway on staggered100 seed 1235 needs a generator-lane fix; a critic guard does not reach it.
- Combining P1 with the .1/.1 floors (the plasticity point) needs its own ring measurement first. P1's zero point is tied to the .01 floor, so at .1 about 18% of the R1 term would remain at the floor. Earlier notes found that any leftover R1 there makes the centers drift.
- If the next gate checks at a constant prior learning rate, the mid-run dip needs attention.

Code is in `repo/reports/toy100/critic-both-arms-3248837/` (not committed). tests.jsonl has 34 rows, including two ERROR rows from a first launch that failed on a hook-import bug and was re-run. result.md has the per-run metrics, code hashes and replay commands. I also added this replication to the existing K3 note in memory.

Files are in `/ml2/hypergan/gan-attempts/claude-pool-20260925T063704Z/critic_both_arms/20260925T113805Z-3248837/`:
- result.md
- tests.jsonl
