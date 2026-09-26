# hid_q ring amplifier

CPU screening on this machine only. Seed 0 repeated bit-identical (`hq` 0.999755859375, modes 8). These pass rates are not an A6000 result.

`--init hid_q` fixes the weights. The offset changes samples and noise only. On the ring the live loop is relativistic paired logistic, `a_r1r2` (coeff 1, kappa 1), and Adam with betas `(0, 0.999)`. The EMA-critic pull does not run (the probe patches a different penalty class; calls 0). A2 latent damping enters and then no-ops, because the 12-row table is dense under batch 128 (`scoped_calls` 0). The generator EMA (decay 0.995) is evaluation-only. Unequal mass is the real `GANTrainer` K3P path, not this loop.

## Amplification map

Passing seed 0, one coordinate `+1e-7`, joint L2 of `(D, G, prior)` against the unperturbed run. Growth below is the step-to-step ratio of that joint L2. "Geo" is the geometric mean while the joint L2 sits between `1e-6` and `1e-2`.

| nudge | where it starts | burst | geo in window | max step | ring outcome |
| --- | --- | --- | --- | --- | --- |
| G weight | t0 `1.2e-7` in G | t3→t4 ×392, then ×8.8, ×9.8 | 13.2× over 4 steps | 392 | FAIL, 7/8 modes |
| D weight | t0 `1.2e-7` in D | t0→t1 ×18.6 into G; t4→t5 ×129 | 7.4× over 5 steps | 129 | PASS |
| one real sample | t1 `7.7e-6` mostly G | t3→t4 ×71, then ×9.5, ×6.4 | 8.8× over 4 steps | 71 | PASS |
| prior row | t0 `9.7e-8` in the prior | t0→t1 ×129 into G; t2→t3 ×82 | 13.9× over 3 steps | 129 | FAIL, 0 modes |

The prior nudge is the most outcome-sensitive place to put `1e-7`. A D nudge of the same size still passes: direction matters, not just the norm.

Phases, all four twins:

- Updates 1–3 move the error, mostly into G, at a few times to ~100×.
- Updates 4–6 are the burst (tens to a few hundred times) and land in G and D together. The prior's share of the L2 is ~30× smaller at the burst, then catches up a step later.
- Updates 7–20 are a shoulder of about 1–2×. The gap is already O(1), so later training cannot pull the twins back together.
- The learning-rate anneal (update ≥ 720) does not start a second burst. Final joint L2 is O(10), with D the largest share.
- EMA-critic state and the latent-anchor state stay at 0. They are not the amplifier on this path.

## Jacobian

Replaying one `mode_hold` step from the saved frame is bit-exact (`replay_check` max abs 0). The hooked seed-0 run still passes at the same `hq`.

A random 20-d subspace of the one-step map underestimates the burst (spectral radius about 1.4–1.7 early, about 1.0 after update 20, about 2.1 at update 800 and that late mode is in the prior block). Early on, the joint radius is larger than any diagonal block, and by update 10 the top pair is complex (`1.677 ± 0.363i`). Coupling adds gain, and there is a rotational piece, but these radii are not the observed stretch.

Power iteration of the same one-step map (6 iterations), at the burst. "Step" is the 1-based update index.

| update | eps | mask | singular value | where the image goes |
| --- | --- | --- | --- | --- |
| 4 | 1e-4 | joint | 109 | G 100, prior 35, D 28 |
| 4 | 1e-5 | joint | 566 | G 480, D 298, prior ~1 |
| 4 | 1e-5 | G only | 565 | same split, G and D |
| 4 | 1e-5 | D only | 1.8 | stays in D |
| 4 | 1e-5 | prior only | 1.0 | stays in the prior |
| 5 | 1e-5 | G only | 447 | G 416, D 150, prior 66 |
| 5 | 1e-5 | prior only | 465 | almost all into G |

The huge gain is a nonlinear kink: shrinking the probe from `1e-4` to `1e-5` moves the update-4 singular value from ~100 to ~570, and at the smaller probe only a G perturbation excites it. D's linear response at `1e-5` is mild, but a D perturbation at `1e-4` is mapped mostly into G (singular value 159). A finite `1e-7` prior kick still collapses the full trajectory, because the kick is handed to G on the next update and then rides the kink.

The historical "~10× per step" is the shoulder after that kink, not a uniform linear factor from step 0. A fresh `1e-7` error is O(1) by update 6–8. Kernel noise reinjected every step would ride the same burst.

## Reliefs

Each candidate is one fixed change. No coefficient sweep, and no coverage, anchor, or forward-KL term. They are off unless `K3P_RELIEF` is set.

- **optimistic.** Daskalakis et al. 2018, Algorithm 1, alpha = 1, after every Adam update (D, G, and the prior). The paper rule, no extra forward.
- **extragradient.** Simultaneous extragradient on the current minibatch. A raw Adam lookahead predicts D and G; the committed host step writes gradients from that predicted point onto the base point. The existing Adam step is the only extrapolation length.
- **ema_fake.** The critic's fake forward (and the particles it is drawn from) uses the generator EMA already maintained at decay 0.995. The generator step stays on the live weights.
- **k3p_pull.** The legacy `a_r1r2` penalty, which is the one this host actually calls, is sent through `mechanism.scaled_penalty`. Decay 0.999, floor 0.01, spike guard 5 after step 200. Those are the constants already in that file.
- **row_damp.** The same A2 rule as `latent.py`: `rho = 0.75 + 0.25 cos`, at most half the row step removed, parent second moment from the raw gradient, `rho = 1` with no history. The sparse gate is unchanged. On a fully dense table, where that gate cannot fire, the same rewrite runs anyway.

Screen: 8 offsets × ring, hold (PASS means 1200 checks), stay (120/120 and `pass_all`, not the shift terminal status), unequal. Same `hid_q` baseline on this CPU.

| candidate | ring | hold 1200 | stay | unequal |
| --- | ---: | ---: | ---: | ---: |
| baseline hid_q | 5/8 | 3/8 | 3/8 | 4/8 |
| optimistic α=1 | 2/8 | 1/8 | 1/8 | 2/8 |
| extragradient | 0/8 | 0/8 | 0/8 | 0/8 |
| ema fake | 0/8 | 0/8 | 0/8 | 0/8 |
| k3p_pull | 1/8 | 0/8 | 0/8 | 1/8 |
| row_damp | 4/8 | 1/8 | 1/8 | 4/8 |

Passing offsets:

| candidate | ring | hold | stay | unequal |
| --- | --- | --- | --- | --- |
| baseline | 0, 303, 404, 606, 707 | 0, 606, 707 | 0, 606, 707 | 101, 202, 303, 505 |
| optimistic | 404, 606 | 404 | 404 | 202, 404 |
| extragradient | none | none | none | none |
| ema fake | none | none | none | none |
| k3p_pull | 606 | none | none | 101 |
| row_damp | 0, 101, 202, 606 | 0 | 0 | 101, 202, 303, 505 |

Optimistic does not stabilize the ring. It drops 5/8 to 2/8 and moves the surviving seeds; 404 becomes the only hold/stay pass, which the baseline did not hold. Extragradient and the EMA fake miss every seed on every gate. Extragradient's ring runs are about 2.5× the baseline wall time, which is the two-pass update actually running. The EMA fake collapses the ring to 0 modes on 7 of 8 offsets.

## What the ring actually executes

Checked on this machine, not only from the trajectory note. Baseline ring seed 0 writes `regularizer_receipt.calls = 0` (pure-a, blend, pure-b, and critic steps all 0, anchor never started) and `latent_receipt.calls = 1200` with `scoped_calls = 0` and empty tables. The probe assigns `scaled_penalty` onto `particlegan.grad_regularizers.GradientPenalty`. The ring calls `benchmarks.legacy.grad_regularizers`, arm `a_r1r2`. A2's `begin` counts every generator step, then skips the rewrite unless some row was missed and the cumulative hit rate is under 1/2. Batch 128 over 12 particles hits every row, so the rate is 1 and the rewrite never runs.

The trajectory report's neighbour-hop (particles climbing toward a mode the critic scores higher, gradient ~10× and D/G ~23 about 40 steps before the Voronoi edge, then a stuck empty mode) is the failure this follow-up is aimed at. The hop steps they logged (101 at 743, 202 at 444 and 665, 505 at 595 and 698) are all before the handover weight can leave 1. On this schedule `s` stays 1 until the critic's applied LR scale drops below 1/2, which the wired run records as the first blend at call 964.

## Follow-up: wire the idle pieces

`k3p_pull` is in the executed graph. Every ring run records 1200 penalty calls, 963 of them pure `a` and 237 blended, critic steps 1200, anchor started at call 964. The spike guard does clip, including on the hop seeds (ring steps 568 and 884 on offset 101; 324–618 on 202; 316, 388, and 1031 on 505) and not at all on passing baseline offset 0. It does not stop the empty mode: those three offsets still fail, at 7 modes rather than the baseline's 7, 6, and 6. Offsets the baseline passed (0, 303, 404, 707) fail. Hold is 0/8, stay is 0/8 (offset 606 reaches 66/120). Unequal drops from 4/8 to 1/8. While `s == 1` this penalty is the mechanism's RMS R1 plus one-sided fake cap, which is the formula that file treats as its `a_r1r2`, so the early steps are not the host's symmetric squared R1/R2. The prox term itself turns on only at step 964, after the logged hops.

`row_damp` fires. Ring dense-calls are 1200 on the fully covered seeds and 1198 on offset 101 (two steps had a silent row, so the dense branch correctly stood down). First step `rho_mean` is 1 with no history. Later steps sit around 0.72–0.99, so the cosine rule is cutting rows, not passing them through. Scoped calls stay 0 on ring, hold, and stay. Unequal stays on the sparse gate (`scoped_calls` 1200, `dense_calls` 0) and keeps the baseline's four passing offsets.

On the ring, damping that can fire does stop the two cleanest hops: offsets 101 and 202 go from FAIL to PASS (8 modes, hq 1.0 and 0.998). Offset 505 stays FAIL (6 modes, hq 0.67). The same rule drops baseline passes 303, 404, and 707, so the ring rate is 4/8 rather than 5/8. The new ring passes do not survive the longer budget: 101 holds 89 checks then fails, stay 24/120; 202 never converges, stay 15/120. Baseline hold/stay passes 606 and 707 are lost (606 holds 171 checks, stay 40/120).

## Recommendation

Keep the unchanged hid_q baseline. Do not turn on any of these five.

The early amplifier is the joint alternating step, a nonlinear kink in G and in D's answer to G. Alpha = 1 optimism doubles the first Adam step. A full Adam lookahead evaluates that kink. The 0.995 generator EMA averages over ~200 steps while the burst is over by update 8.

The neighbour-hop is a later, separate failure, and the two mechanisms that were supposed to be present are idle on this host. Putting them in the graph does not remove the seed dependence. The EMA-critic pull cannot see a hop before step ~964; the guard clips a few pre-hop steps and the empty mode remains. Dense A2 damping does cancel two of the three ring hops and then opens three new ones, and the cancelled hops come back before a 1200-check hold. Both settings are the ones already written down. Changing rho, the guard ratio, or the handover floor would be a sweep.

An A6000 run is still required before treating the pass rates, or the location of the kink, as confirmed. The useful confirmation target is the amplification map, plus the fact that the ring's executed penalty and latent step are not the K3P pull and not A2. Another sweep of these settings is not the useful next run.

## Learning-rate anneal

The cosine anneal is active on the ring, hold, and stay probes. It is not the `training_recipe` branch inside `mode_hold` (that argument is null, so that block does not run). `probe.py` / `hold.py` / `shift.py` install `RecipeControl`, and `control_host_schedules` replaces `schedule_optimizer`. Each optimizer step calls `policy_multipliers` with the recipe's `lr_anneal_start` 0.6, prior floor 0.05, network floor 0.01, and horizon cap 1600. Those numbers come from `reports/toy100/gap-fill-20260925/sources/k3p/config.json`. Hold and shift pass the same floors and the 0.6 start on the command line.

The controller's horizon is the 1,200-step ring budget (`min(1200, 1600)`). `learning_rate_scale` stays at 1 through completed update 720 and first drops at update 721. The saved ring action trace matches that: update 720 is lr 0.00425 / prior 0.0085, update 721 is multiplier 0.999989. By update 1200 the network multiplier is 0.01 and the prior multiplier is 0.05. Hold and stay keep using that 1,200-step horizon, so after update 1200 the rate is already on the floor for the rest of the run. The stay window (updates 1201–2400) is entirely at the floor. The shift rate trace spans 4.25e-5 to 0.00425 for D and 4.25e-4 to 0.0085 for the prior.

Offset 0's mode curve sits on that start. Step 700 is still full LR and already 7 modes, hq 0.50. Step 750 (multiplier 0.990) is 1 mode. Step 800 (multiplier 0.934) is 1 mode. Step 900 (multiplier about 0.69) is 8 modes again, and the run finishes PASS. The 1e-7 sensitivity twins do not re-amplify there: joint-gap growth from update 680 to 1200 is about 1.00× (max about 1.01), aside from the prior twin, which had already collapsed and moves 1.15× at update 728. The baseline weights do move: median parameter step over 600–720 is 0.22, and update 750 is 2.55, mostly in D.

One constant-LR arm, pre-anneal rates held for the whole run (anneal start 0, both floors 1, cap left at 1600). Multipliers measured at 1 and the only rates are 0.00425 and 0.0085, including every hold step and all 3,600 shift steps. Curves match the baseline through step 700 on all 8 offsets. CPU only.

| offset | ring baseline | ring constant | hold baseline | hold constant | stay baseline | stay constant |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | PASS 8 / 0.9998 | FAIL 8 / 0.878 | PASS | NOT_CONVERGED | 120/120 | 33/120 |
| 101 | FAIL 7 / 1.000 | FAIL 7 / 0.659 | NOT_CONVERGED | NOT_CONVERGED | 0/120 | 0/120 |
| 202 | FAIL 6 / 0.998 | FAIL 7 / 0.985 | NOT_CONVERGED | NOT_CONVERGED | 0/120 | 0/120 |
| 303 | PASS 8 / 1.000 | FAIL 6 / 0.670 | POST_CONVERGENCE_FAIL | NOT_CONVERGED | 112/120 | 9/120 |
| 404 | PASS 8 / 1.000 | FAIL 6 / 0.400 | POST_CONVERGENCE_FAIL | NOT_CONVERGED | 116/120 | 17/120 |
| 505 | FAIL 6 / 1.000 | FAIL 4 / 0.263 | NOT_CONVERGED | NOT_CONVERGED | 0/120 | 0/120 |
| 606 | PASS 8 / 1.000 | FAIL 7 / 0.694 | PASS | NOT_CONVERGED | 120/120 | 53/120 |
| 707 | PASS 8 / 0.999 | FAIL 0 / 0.000 | PASS | NOT_CONVERGED | 120/120 | 0/120 |

Constant LR is 0/8 ring, 0/8 hold, 0/8 stay. On offset 0 the 1-mode hole at 750–800 is gone (8 modes at 750, 7 at 800), then step 900 is 2 modes and the final hq is 0.878. Offsets that the anneal carried to a pass (303, 404, 606, 707) fail at the full rate. Continuous learning still needs a schedule that stays stable without decay; this constant pre-anneal rate is not that schedule. No floor or start was varied.
