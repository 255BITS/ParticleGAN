# k3p_adversarial_progress — round 2

K3P stays the selected base. Nothing here is promoted. `current-research-base.json` was not edited.

The parent scores were not rerun: 22/22 GPU toys, ring hold 1200/1200 (min HQ 0.90723), extension 300/300 (min HQ 0.98779), target-shift deadline **FAIL 28/81**, delay 1130. Parent mechanism `d2eb08ee932b288cbba25cd1e7be3a9572b129bd1baf0b79718be1eb37ba9391`.

Three proposals, one GPU worker. Critic cosine was not used. It is negative while this ring is still going from 4 modes to 8, and it is still negative during a healthy 8-mode hold. The phase signal is critic parameter-gradient RMS against a peak that is set after a 200-step estimator warmup and then decays at 0.9997 per critic step.

## Leaderboard

Ranked by hold, extension, and timely recovery, then by how close recovery got.

| Candidate | Own hold | Extension | Shift pre-hold | Deadline recovery | Final live | Toys |
|---|---|---|---|---|---|---|
| K3P parent (not rerun) | 1200/1200 | 300/300 | (parent evidence) | FAIL 28/81, delay 1130 | — | 22/22 |
| **ap3_partial_reopen** | **PASS 1200/1200**, min HQ 0.96851, converged step 1400, 0 settling failures | **PASS 300/300**, steps 2601–2900, min HQ 0.96802, min modes 8 | **120/120**, min HQ 0.97559, min modes 8 | **FAIL 72/81**, delay 780, stable from step 3180 | 8 modes, HQ 0.99194 (EMA 8 / 0.87573) | NOT_RUN |
| ap1_magnitude_phase | NOT_RUN | NOT_RUN | 120/120, min HQ 0.97559 | FAIL, delay none, passing suffix 0, min modes 0 | 4 modes, HQ 0.91675 | NOT_RUN |
| ap2_anchored_reopen | NOT_RUN | NOT_RUN | 120/120, min HQ 0.97559 | FAIL, delay none, passing suffix 0, min modes 0 | 7 modes, HQ 0.73608 (EMA 5 / 0.74927) | NOT_RUN |

ap3 is the strongest result in this lane and still fails the raw shift verdict. 72/81 is not a pass. No frozen control, no sensitive screen, no 22-toy qualification, no delayed or repeated change.

## What the runs showed

Cold acquisition matches across all three and matches the parent's pace: 8 modes and HQ 1.0 at step 500, with the early R1 penalty and full base rates (generator/critic 0.00425, prior 0.0085). The phase closes after a quiet stretch, mixing weight `s` reaches 0 near step 1200, and rates sit on the K3P floors (network about 4.25e-5, prior about 4.25e-4). On the uninterrupted hold that state never reopened.

ap1 also restored `s` to 1 when RMS jumped. By step 2449 the rate was back to 0.00425 and the early penalty was back. The ring reached a precise 4-mode state (step 3000: 4 modes, HQ 0.99927) and the phase closed again. Deadline recovery did not start.

ap2 left `s` at 0, so the gradient cap and the 0.999 EMA anchor stayed on, but put the rate back to full. Modes moved (2, then 5, 7, back to 5, 7) and never locked 8. Final HQ 0.73608. Optimizer steps stayed active; this was not a frozen generator.

ap3 uses the same cold path, then on a later RMS reopen sets the rate gain to 0.2 and leaves `s` at 0. Applied rates during that window: generator and critic **0.000884**, prior **0.00204** (multipliers 0.208 and 0.24). Eight modes were back by step 2600 (HQ 0.99756). The deadline window is steps ≥ 2800 and needs all 81 checks. **72 passed.** The 9 failures are steps 3070, 3080, 3110, 3120, 3130, 3140, 3150, 3160, and 3170. Worst deadline point is step 3120: 5 modes, HQ 0.34839. Sustained recovery starts at step 3180 (delay 780, deadline is 400). From 3180 through 3600 every check passed. Both roles and the prior took 3600 Adam updates; post-shift displacements stayed positive.

The fixed 800-step hold at gain 0.2 outlasted the reacquisition. Eight good modes were already present at step 2600, and the later wobble happened while that partial rate was still forced on (`refit_until` 3201).

## Rules

`config.json` `a1475108…`, `latent.py` `197df635…`, and `response.py` `7e71d60a…` are byte copies of pinned K3P. Learned particle prior, direct response, and bounded sparse-latent damping are unchanged. On these ring runs latent calls equal the update count and scoped calls are 0; direct-response calls are 0.

Mechanism hashes:

- ap1 `694a40d6cb11c110abf36f7e9852c7ef88c44b244c66843101e4c3af1eb8b667`
- ap2 `ede2c615fc9d0170562fa983577dc9ebac5b155370787989bdcebd66ae42c86f`
- ap3 `8d65d1d690c1a1ccfe4e2cc82a2f6982fb31c50f67cba4e8665623a7f3f627de`

Mixing weight uses the K3P floor map on a mixing gain, not on last/max critic LR. Until that mixing gain first hits 0, it tracks the rate gain: full early penalty, then 0.99 decay per critic step after 250 consecutive steps with RMS below a quarter of the peak. After that latch, ap1 still reopens the mixing gain; ap2 and ap3 do not.

Host cosine arguments are ignored. Step count, horizon, anneal fraction, quality scores, target centers, and the shift time are not read. Estimator memories, not a training budget: RMS warmup 200, quiet dwell 250, peak decay 0.9997, fast decay 0.9, slow decay 0.99, guard 5× after 200 Adam steps, anchor decay 0.999. ap2/ap3 also hold a reopen for 800 critic steps and reset the peak to the slow RMS estimate 200 steps into that window.

**Remaining budget dependence.** Input noise 0.5 over the first 0.1 of `noise_horizon=1200`, and output noise warming to 0.029 over 0.2 of that horizon, are still the driver schedule. The probe rejects any other noise horizon. That inherited schedule is an intermediate ablation. A paired prefix under two declared horizons was **NOT_RUN**.

Extra EMA-critic forwards (anchor evaluations): ap3 shift 2795, ap3 hold 2095, ap1 shift 2137, ap2 shift 2795. Anchor starts at penalty call 805 on the cold path. Parent K3P's saved counter is not a comparable zero; its anchor starts once the LR ratio leaves the early penalty.

## NOT_RUN

- ap1 and ap2 canonical convergence-gated hold and 300-update extension. Their shift pre-hold is the same 120/120 cold path as ap3, and ap3's own hold passed, but that is not a substitute score for those hashes.
- Matched frozen recovery control.
- Sensitive four (mode_hold toy, unequal mass, unequal width, stripes) and the other 18 frozen toys. Native 7000-update coverage and accuracy were not run.
- Delayed change, second change, and any stress protocol.
- Two-horizon prefix equality (model, optimizer, controller, EMA, RNG, rates, noise).

No seed sweeps, no coefficient grid, no metric feedback, no edits to old attempts, the pinned parent, or the frozen runtime. No pushes or comments.

## Replay

Environment on every benchmark process: `CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `/tmp/pr38-default-env/bin/python`. Runtime repo and mode-hold fixture are the gap-fill ones below. Logs: `reports/toy100/k3p-adversarial-progress/logs/`.

```bash
BASE=/ml2/hypergan/gan-attempts/formulations-20260925T165310Z/k3p_adversarial_progress/20260925T165310Z-3675713/repo/reports/toy100/k3p-adversarial-progress
RUNTIME=/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda
FIXTURE=/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt

# ap3 shift (FAIL 72/81). Same shape for ap1 and ap2.
/tmp/pr38-default-env/bin/python -u "$BASE/candidates/ap3/shift.py" \
  --repo "$RUNTIME" --config "$BASE/candidates/ap3/config.json" \
  --task mode_hold --backend cuda --initial-state "$FIXTURE" \
  --output "$BASE/runs/ap3-shift-replay" \
  --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6

# ap3 hold + 300 extension (PASS)
/tmp/pr38-default-env/bin/python -u "$BASE/candidates/ap3/hold.py" \
  --repo "$RUNTIME" --config "$BASE/candidates/ap3/config.json" \
  --task mode_hold --backend cuda --initial-state "$FIXTURE" \
  --output "$BASE/runs/ap3-hold-replay" \
  --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6 --post-window 300
```

## Next mechanism

Keep ap3's cold path and the anchor-on reopen. Drop the fixed 800-step dwell at gain 0.2. That dwell is what was still forcing generator/critic LR 0.000884 after eight modes had already returned, and the deadline misses are exactly the wobble that followed (3070–3170). Let the post-reset RMS peak decay the partial rate once the new activity fades. Do not restore the early penalty, and do not reopen all the way to the base rate: those were ap1 and ap2.
