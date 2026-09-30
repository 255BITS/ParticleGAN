# Critic-value spare step: warm pass, delayed hold still fails

No no-decay replacement. On this machine the scheduled control holds through update 2400. A critic-value spare-particle pull passes the dense warm window and, in a separate cold run, the 1,200-update ring. The same warm branch then fails the 2,400 continuation. Final HQ hides those failures. This is not a production candidate.

Torch is 2.13.0+cpu, one thread, AVX2 caps. The archived cu126 warm-state hash does not reproduce here (`57c10b57…` versus `6cc79b6e…`). Local PR84 therefore is not the archived 200/200 then 1390 episode. Both machines still show the same kind of failure: a clean final after missed checks.

## What was searched

Closed and not reused: rest-damping from slope, curvature, or width; Monge / Sinkhorn drift; path-crossing; χ²-KDE; Chamfer and one-sided coverage pullbacks; target-center oracles; critic-weight EMA; time-boxed loosening; HQ latches.

Read for a different lever: [Gradient Flow Drifting](https://arxiv.org/abs/2603.10592) (KDE score difference, bandwidth left open) and [drifting as score matching](https://arxiv.org/abs/2603.09936) (Laplacian tails, stop-grad, no clock annealing). The ring diagnosis still stands: the missing mode has the highest critic value while the local generator gradient points away. Chamfer's cold failure was a huge latent pseudoinverse, not a wrong target label.

## Rule that was run

After the frozen PR84 step, on the unconditional ring only: if `max D(real) - mean D(fake)` exceeds two standard deviations of `D(real)`, move one spare particle at most 0.1 in output space toward the softmax centroid of those high-scoring reals. A particle is eligible only if a neighbor lies inside half that distance, or it is already clearly closer than every other particle. The step is a prior-only Jacobian pull. It is kept only when that particle lands closer and the others do not move. Trajectory is untouched PR84. No mode centers and no elapsed-time gain.

## Leaderboard (this CPU, seed 0)

| Run | Warm 1001–1200 | Later checks | Worst later | Final | Verdict |
| --- | ---: | ---: | --- | --- | --- |
| Scheduled identity | 200/200, min HQ .990 | 120/120 to 2400 | 8 / .998 | 8 / .998 | Holds |
| Constant Adam .00425 | 4/200 | 10/120 | 3 / .028 | 8 / .828 | Fails; final hides it |
| PR84 stencil, correction off | 196/200 (1129–1132) | 114/120 | 7 / .829 | 8 / 1 | Fails; final hides it |
| Value spare step | **200/200, min HQ .919** | **112/120** | **7 / .830** | **8 / .999** | **Fails; final hides it** |

Value-transport failures: 1270 (8 / .830), 1280 (8 / .893), 1520 (8 / .889), 1820 (7 / .841), 2090 (8 / .899), 2100 (8 / .839), 2180 (8 / .868), 2390 (7 / .871). The pull fired twice in 1,400 updates and both were accepted. It was closed during the dips.

Local PR84 failures after 1200: 1720, 1840, 1900, 2110, 2290, 2370. Not the archived list 1390, 1540, 1550, 1560, 1570, 1910, 2060, 2150. Same lesson: do not trust the final checkpoint.

Separate cold run of the same rule, before this hold: trajectory MSE .000943 (suffix 18) and ring 8 modes / HQ .970, passing suffix 8, first pass at 850. That 1,200-update pass does not survive the continuation standard above. No production 22/22 claim.

## Why the lever missed

The instability shows up as particles leaving the HQ ball while the critic-value gap stays inside two real-score standard deviations. A gate that waits for the critic to prefer some reals rests exactly when the matched cloud starts to drift. Two accepted pulls did not change that. Lowering the gap threshold would be a sweep of a gate that is already blind at the failing checks.

## Recommendation

Keep the scheduled control as the only hold that passed. Attribute the generator proposal and output motion at the first failing check on each machine (1270 here, 1390 on the archived cu126 continuation) before adding another controller. The value-gap rule should not be tuned further.
