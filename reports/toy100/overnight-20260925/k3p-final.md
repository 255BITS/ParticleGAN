One critic rule does carry both ends, measured here. The best candidate, k3p, also fixes a stability failure I found in the plain handover rule (k3). Plasticity still fails for every rule. Everything was measured in this checkout, all 4-seed native gates used the same declared seeds (1234–1237), and nothing is running now.

**What each arm does**
- **b_cap's precision:** mostly comes from having no zero-centered R1 on real samples once the rates anneal. The critic keeps a slope at the data, which pulls samples to the mode centers.
- **a_r1r2's three toys:** R1 damps the critic while its learning rate is high, which prevents the toys' allocation failures.
- **The caps:** near equilibrium they are almost always inactive. The r1s ablation (R1 scaled down, no b_cap terms) matches k3 bit-for-bit on vector_unequal_mass. It matches three grid100 seeds within .001 but fails seed 1236 (live .968–.973), so the caps still matter on some runs.

| candidate | 3 lost toys | 16 toys | grid100 | live precision / center_rms | mid-run live min | ring hold | ring checks < .90 by step 2900 | ring shift |
|---|---|---|---|---|---|---|---|---|
| **k3p** | 3/3 | 13 not run | **4/4** | .980–.988 / .15–.18 | .956–.968 | PASS | **0** (min .988) | FAIL 28/81 |
| k3 | 3/3 | **16/16** | **4/4** | .982–.989 / .10–.12 | .847–.920 | PASS | 237 (min .719) | FAIL 22/81 |
| r1s (ablation) | 3/3 | 13 not run | 3/4 | .968–.989 | .847–.894 | not run | not run | not run |

For reference, the session arms on the same seeds scored 0/4 (a_r1r2) and 1/4 (b_cap) on grid100.

- **k3:** uses a_r1r2 while the critic's applied learning rate is high and hands over to b_cap exactly at the rate floor, plus a spike guard on the critic that only acts after 200 updates. Every toy's passing suffix equals a_r1r2's, and before the handover it is bit-identical to the a_r1r2 arm. Its weakness: at the floor the critic is effectively unregularized and has no damping. The ring hold passes as the protocol is written, but 44 updates after the hold window ends live precision falls below .90 on 237 of 300 checks; the a_r1r2 parent has 0. On grid100, live precision dips to .85–.91 mid-run, though the final gate still passes.
- **k3p:** k3 plus an R1 penalty that pulls the critic's input gradient toward that of a slow running average of its own weights (decay .999, chosen once, not tuned), instead of toward zero. It is zero when the critic is steady and restoring when it swings. It removes both the ring failure and the mid-run dip. The cost is live center_rms of .15–.18, closer to the .20 limit than k3's .10–.12.

Of stability, plasticity, toys and natives:
- **k3p** holds stability, the three lost toys and grid100; it loses plasticity. Its other 13 toys were not run.
- **k3** holds all 16 toys and grid100; stability is marginal; it loses plasticity.

**Not measured:**
- staggered100 and rotated100 for any candidate: I dropped k3's staggered100 run to test k3p.
- k3p's other 13 toys.
- the r1s ring gates.

**Recommendations:**
1. Qualify k3p on the remaining 13 toys and on staggered100/rotated100 at 4 seeds.
2. Test k3p on the ring at .1/.1 floors. Earlier handover rules lost the ring there because b_cap at a 10% critic rate has no damping, and k3p supplies that damping.
3. If center accuracy becomes the binding limit, try a faster average (decay .99) as one declared change.
4. Grade the 300 updates after the ring hold window, not just the window itself: that is where k3's failure appeared.

Files are in `/ml2/hypergan/gan-attempts/claude-pool-20260925T063704Z/critic_both_arms/20260925T125747Z-3385271`:
- `result.md`: full results and replay commands
- `tests.jsonl`: 41 gates (38 PASS, 3 FAIL, no errors)
- `runs/`: logs and outputs
- code: `repo/reports/toy100/critic-both-arms-3385271/{k3,r1s,k3p}/` with runners, queues and `code-hashes.txt`
