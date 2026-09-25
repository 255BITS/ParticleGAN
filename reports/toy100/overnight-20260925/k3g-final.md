Yes: one critic rule carries both ends on the frozen base schedule. I call it K3. It keeps all 16 focused toys and passes grid100 on all four seeds with b_cap-level precision. Its one native failure, staggered100 seed 1235, starts in the generator, not the critic. It holds the ring, but it doesn't re-learn a shifted target.

**K3 in one line:** the penalty is `s·a_r1r2 + (1−s)·b_cap`, where `s` depends only on the learning rate the critic's optimizer last applied (no task names, no per-host arm). It is exactly a_r1r2 while that rate is at least half its maximum, and exactly b_cap once it reaches the frozen .01 network floor. A guard also clips the critic's gradient when one update is more than 5× its usual size, but only after a tensor has had 200 updates.

**What each arm does.** a_r1r2 forces the critic to be flat on the real data. While learning rates are high, that stops the critic locking in a partial set of modes, which is exactly how b_cap loses its three toys. b_cap lets the critic keep a slope at the real modes, and late in training that slope pulls samples tight enough for precision. The two are needed in different phases, so a handover keyed on the critic's own learning rate works.

**Results** (seeds 1234–1237, fixed in advance and identical for both candidates):

| candidate | 16 toys | grid100 | staggered100 | ring hold (.01/.05) | ring shift |
|---|---|---|---|---|---|
| **K3** (critic only) | **16/16** | **4/4 PASS**, live precision .981–.989 | 3/4 (seed 1235 stops at 94/100 modes) | **PASS**, 1200/1200 | **FAIL**, 22/81 |
| K3G (same guard on the generator too) | 16/16 | not run | **4/4 PASS** (seed 1235 at .9708) | not run | not run |

- **Toys:** K3's passing suffixes equal the a_r1r2 reference on 15 of 16 toys. Before the handover, its logged rows are bit-identical to the reference.
- **Natives:** every run that acquired all 100 modes passes, 7 of 7. On grid100, center_rms is at most .134; a_r1r2's was .19–.23.
- **Seed 1234 on grid100:** a2 + a_r1r2 collapsed to 17 modes, while K3 acquired all 100. K3 is identical to a_r1r2 until its handover at step 1285, so the critic guard is the only difference.
- **Where the staggered failure starts:** the generator's 2×2 output map runs away at steps 639–642, and the critic follows at 643. A critic-only rule can't stop that.
- **K3G is a cross-lane test.** It fixes seed 1235, but it also clips normal generator updates on seeds where K3 clipped nothing. img_stripes2's passing suffix drops from 9 to 5, and precision on seed 1235 is only just above the floor.
- **Plasticity:** I don't think a critic change can supply it at these floors. At a 1% learning rate the networks move too slowly to follow a unit shift within the 400-update deadline. The floor-frontier study had floors .1/.1 re-learning in 110 updates.

**Not measured:** rotated100 for either candidate, grid100 and the ring protocols for K3G, and anything at floors .1/.1. I stopped one K3 rotated100 run for the hard stop; it isn't recorded.

**Tests:** 46 executed gates, 44 PASS and 2 FAIL (K3's staggered seed 1235 and the ring shift), about 78 GPU-minutes. A first launch crashed on an import bug before any training ran; I moved its 7 ERROR rows to `tests-setup-errors.jsonl`.

**Recommendations:**
1. Adopt K3's penalty as the round's critic for the base schedule.
2. Hand K3G to the acquisition lane: run grid100 and rotated100 on the same four seeds, then the ring protocols. It may need a higher threshold for the generator.
3. If the round moves to floors .1/.1, K3's handover point must be re-set to the new floor. An earlier attempt found pure b_cap at a 10% critic learning rate breaks the ring hold, so the next thing to test is a critic-only floor of .01.

The code, harness and hashes are in `repo/reports/toy100/critic-both-arms-3111873/` and are not committed. Replay commands are in `result.md`.

Files are in `/ml2/hypergan/gan-attempts/claude-pool-20260925T063704Z/critic_both_arms/20260925T101301Z-3111873/`:
- `result.md`
- `tests.jsonl`
- `runs/queue.log`
