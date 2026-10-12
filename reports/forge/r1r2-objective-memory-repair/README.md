# R1/R2 objective and optimizer-memory readout

**No complete Tier 1 repair.** Global sliced transport and AMSGrad on m110 each
pass five of six ordinary tasks plus their own clock audit. Both fail Ring16.
This finite comparison is closed. The [technique inventory](../technique-inventory.md)
remains the sole current generated leaderboard; this table preserves the frozen
comparison and does not change selection, defaults or qualification.

Each row is one global recipe. Gaussian and words require independently confirmed
acquisition; pole, token, AE and Ring retain their original sustained gates.
Every cell links its actual-training GIF and has a provenance sidecar.

| Recipe | Gaussian | Pole | Token | AE | Ring16 | Words | Clock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Scheduled reference | [PASS](publication/media/baseline-gaussian1d_smoke.gif) | [PASS](publication/media/baseline-two_pole.gif) | [PASS](publication/media/baseline-unused_token_hold.gif) | [PASS](publication/media/baseline-ae_gan_hold.gif) | [FAIL](publication/media/baseline-ring16_acquisition.gif) | [PASS](publication/media/baseline-five_word_joint_smoke.gif) | [FAIL](publication/media/baseline-clockfree_audit_measurement_v1.gif) |
| Constant Adam parent | [PASS](publication/media/adam-gaussian1d_smoke.gif) | [PASS](publication/media/adam-two_pole.gif) | [PASS](publication/media/adam-unused_token_hold.gif) | [PASS](publication/media/adam-ae_gan_hold.gif) | [FAIL](publication/media/adam-ring16_acquisition.gif) | [FAIL](publication/media/adam-five_word_joint_smoke.gif) | [PASS](publication/media/adam-clockfree_audit_measurement_v1.gif) |
| D/G momentum, zero prior momentum (m110) | [FAIL](publication/media/m110-gaussian1d_smoke.gif) | [PASS](publication/media/m110-two_pole.gif) | [PASS](publication/media/m110-unused_token_hold.gif) | [PASS](publication/media/m110-ae_gan_hold.gif) | [FAIL](publication/media/m110-ring16_acquisition.gif) | [FAIL](publication/media/m110-five_word_joint_smoke.gif) | [PASS](publication/media/m110-clockfree_audit_measurement_v1.gif) |
| Global sliced transport | [PASS](publication/media/global_transport-gaussian1d_smoke.gif) | [PASS](publication/media/global_transport-two_pole.gif) | [PASS](publication/media/global_transport-unused_token_hold.gif) | [PASS](publication/media/global_transport-ae_gan_hold.gif) | [FAIL](publication/media/global_transport-ring16_acquisition.gif) | [PASS](publication/media/global_transport-five_word_joint_smoke.gif) | [PASS](publication/media/global_transport-clockfree_audit_measurement_v1.gif) |
| AMSGrad on m110 | [PASS](publication/media/ams_m110-gaussian1d_smoke.gif) | [PASS](publication/media/ams_m110-two_pole.gif) | [PASS](publication/media/ams_m110-unused_token_hold.gif) | [PASS](publication/media/ams_m110-ae_gan_hold.gif) | [FAIL](publication/media/ams_m110-ring16_acquisition.gif) | [PASS](publication/media/ams_m110-five_word_joint_smoke.gif) | [PASS](publication/media/ams_m110-clockfree_audit_measurement_v1.gif) |

## Ring density remains the failure

| New global recipe | Best covariance error | Final covariance error | Final minimum eigen ratio | Final HQ |
| --- | ---: | ---: | ---: | ---: |
| Global sliced transport | 2.072274 at 1550 | 2.352042 | 1.164918 | .760498 |
| AMSGrad on m110 | 1.535922 at 1467 | 1.679120 | .159521 | .879150 |

The original covariance bound is .85, minimum eigen ratio .15 and HQ .85.
Both recipes have **zero covariance passes in 96 evaluations and zero full
terminal passes**. All 16 modes occur at both endpoints. Global transport also
fails HQ at every evaluation; a sufficient minimum spread does not bound excess
spread. AMSGrad passes the other endpoint bounds, but covariance fails at all five
terminal checks and minimum spread fails at four. Neither endpoint satisfies the
original sustained gate.

Relative to its own exact parent, global transport changes final covariance from
2.386427 to 2.352042 and lowers HQ from .842285 to .760498. AMSGrad repairs m110's
Gaussian and word acquisitions while worsening final covariance from 1.083997 to
1.679120. Those distinct comparisons do not support composing their mechanisms
or choosing a different recipe for each task.

The independent [saved-panel audit](density-readout.md) reconstructs all six
original Ring metrics across 192 new panels with exact agreement. Its endpoint
geometry distinguishes global transport's broad radial and tangent variance from
AMSGrad's compressed radial and stretched tangent variance. The Jacobian-based
noise proxy is first order and supplies no causal or exact covariance partition.

## Acquisition and endpoint evidence

Both word runs first acquire at update 834 under the original independent
confirmation gate. Global transport has only 1/24 confirmed passing checks and
ends with two modes, quality .395508 and failed inverse reconstruction. Its
acquisition PASS remains valid, but does not establish retention. AMSGrad has
24/24 confirmed passing checks and also passes the endpoint: five modes, quality
1 and exact inverse reconstruction. This endpoint is useful evidence, not a
separately executed Tier 2 word hold. Both Gaussian endpoints satisfy their
numerical bounds; their smoke PASS likewise supplies acquisition rather than
continuation qualification. Pole, token and AE retain their sustained passes.

## Recipe, evidence and cost

The [plan](PLAN.md), [registration](registration.json), [independent preflight](preflight-review.json)
and [certified results](publication/results.json) bind the experiment. Global
transport changes only `kinetic_transport_weight: 0→1` on the exact constant Adam
parent, with local weight 0 and 32 fixed projections. AMSGrad changes only
`amsgrad: false→true` on exact m110. Its D/G+E/prior beta1 values remain .5/.5/0;
beta2 remains .99. Both retain constant LR .0085, positive paired R1/R2 coefficient
.1 every update, learned prior updates, zero extra prior regularization and
zero noise/EMA. Transport adds a declared generator/prior objective; AMSGrad
changes maximum second-moment memory globally in every active optimizer group.

All tasks retain seed 0, public deterministic initialization, architecture,
target/data law, actual initial tensors and ordered target batches, initial prior
law/capacity, sampling, update budgets, evaluation cadence and gates. Every
consumed named RNG stream is checkpointed. Direct pole remains generator-owned;
token has no sampled prior. Word's explicit zero-width exception and the other
positive-width MoG laws remain unchanged.

Execution commit: `2b0ad9ec7c5b2c1e91ab0d70751da0b7e36a7b42`. Scientific source and
runtime remain those in registration. All 35 cases are certified and have training
GIFs. Twenty-one exact cached control attempts add no training. Fourteen new
attempts cost **379.57483000901993 seconds**. The shared queue totals **203 unique
attempts / 6,222.199867211952 seconds**, preserving all 189 earlier charges and
their owners. Full reservation remains 12,600 seconds; the aggregate paid cap
remains 8,000. No retries, seed changes, execution errors or shortened allowances.

The [collection audit](collection-audit-readout.md) verifies stored original
grades, checkpoints, actual optimizer histories, source/runtime, caches, media,
costs and initialization/data/RNG identities. Actual component transport counters
and frozen same-batch public source bind consumption. Scalar hosts retain no
separate transport counters or paired force decomposition; momentum buffers
cannot recover instantaneous gradients for beta1 .5. Read-only collection and
diagnosis add no updates or sampling draws. Bulk logs, panels and checkpoints
remain in `/mnt/ml7tb/ParticleGAN-forge/r1r2-objective-memory-repair-20261012`;
[validation](validation.json) binds software and memory checks.

## Recommendation

Close both hypotheses without adoption. Preserve AMSGrad on m110 as an acquisition
control, while retaining its density regression. Further paid work needs a
distinct mechanism that addresses component shape and the same complete suite.
The [Jacobian-conditioning review](research/jacobian-conditioning.md) identifies
unresolved dimensional, discrete-host, scale and API requirements; it supplies no
ready global recipe or justified clamp coefficient. No automatic refinement,
longer training, seed study, Tier 2 or new round follows this registration.
