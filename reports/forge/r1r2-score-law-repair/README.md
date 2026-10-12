# R1/R2 native score-law readout

**No complete Tier 1 repair.** Native non-saturating loss on exact AMSGrad m110
passes five of six ordinary tasks plus its own clock proof, but fails Ring16.
It preserves the parent's acquisition passes and does not supersede that parent.
This finite comparison is closed. The [technique inventory](../technique-inventory.md)
remains the sole current generated leaderboard; the table below is an unranked,
frozen comparison of whole global recipes.

| Recipe | Gaussian | Pole | Token | AE | Ring16 | Words | Clock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Scheduled reference, cached | [PASS](publication/media/baseline-gaussian1d_smoke.gif) | [PASS](publication/media/baseline-two_pole.gif) | [PASS](publication/media/baseline-unused_token_hold.gif) | [PASS](publication/media/baseline-ae_gan_hold.gif) | [FAIL](publication/media/baseline-ring16_acquisition.gif) | [PASS](publication/media/baseline-five_word_joint_smoke.gif) | [FAIL](publication/media/baseline-clockfree_audit_measurement_v1.gif) |
| AMSGrad m110 parent, cached | [PASS](publication/media/ams_m110-gaussian1d_smoke.gif) | [PASS](publication/media/ams_m110-two_pole.gif) | [PASS](publication/media/ams_m110-unused_token_hold.gif) | [PASS](publication/media/ams_m110-ae_gan_hold.gif) | [FAIL](publication/media/ams_m110-ring16_acquisition.gif) | [PASS](publication/media/ams_m110-five_word_joint_smoke.gif) | [PASS](publication/media/ams_m110-clockfree_audit_measurement_v1.gif) |
| Non-saturating AMSGrad m110 | [PASS](publication/media/non_saturating-gaussian1d_smoke.gif) | [PASS](publication/media/non_saturating-two_pole.gif) | [PASS](publication/media/non_saturating-unused_token_hold.gif) | [PASS](publication/media/non_saturating-ae_gan_hold.gif) | [FAIL](publication/media/non_saturating-ring16_acquisition.gif) | [PASS](publication/media/non_saturating-five_word_joint_smoke.gif) | [PASS](publication/media/non_saturating-clockfree_audit_measurement_v1.gif) |

Gaussian and words use independently confirmed acquisition. Pole, token, AE and
Ring retain their original sustained gates; clock remains a separate proof.
Every cell has a certified result, actual-training GIF and provenance sidecar.

## Ring density still fails

| Recipe | Endpoint covariance error (≤.85) | Endpoint minimum eigen ratio (≥.15) | Endpoint HQ (≥.85) |
| --- | ---: | ---: | ---: |
| AMSGrad parent | 1.679120 | .159521 | .879150 |
| Non-saturating | 1.556546 | .094075 | .888672 |

The new score law slightly lowers endpoint covariance error while reducing minimum
spread below its bound. It reaches all 16 modes with mass TV .049805 and 4,096
samples, but has **zero full passes in 96 checks and zero terminal passing suffix**.
Its best covariance error is 1.501881 at update 1584; covariance fails every check.
Both covariance and minimum spread fail all five terminal checks; the first also
fails HQ. The original requirement remains five consecutive full terminal checks.
The [saved-panel audit](density-audit.json) reproduces all six gate metrics across
96 retained panels exactly. Its first-order Jacobian geometry supplies no causal
or exact additive covariance partition.

## Acquisition is separate from endpoint quality

Non-saturating words first confirm at update 834 and pass all 24/24 confirmations,
as does the AMSGrad parent. Both endpoints have five modes, quality 1 and exact
inverse reconstruction; minimum reconstruction probability is .999085 for the new
recipe versus .998372 for the parent. The scheduled reference keeps its original
acquisition PASS (one confirmed check at 2501), despite ending with four modes and
failed inverse reconstruction. None supplies a separately executed Tier 2 hold.

Non-saturating Gaussian acquires at 292 with 15/24 passing checks, versus the
parent's acquisition at 375 with 6/24. **Both fail the endpoint KS bound .05**:
.057647 and .069604 respectively. Their original smoke PASS grades remain valid.
The scheduled reference also acquires at 292, but its endpoint fails KS and mean
error bounds. Acquisition does not establish continued retention.

The new pole, token and AE runs retain their original sustained passes, with
terminal passing suffixes 12, 19 and 24, each above the required five. Their
endpoints satisfy the original transport/gradient, used/unused-slot and
reconstruction/hold bounds. Its own clock comparison passes all four parity checks.

## Fixed recipe, evidence and cost

The [plan](PLAN.md), [registration](registration.json), [preflight review](preflight-review.json)
and [certified publication](publication/results.json) bind exactly one delta:
`Recipe.loss: relativistic → non_saturating` on measured AMSGrad m110. Both positive
R1/R2 penalties remain coefficient .1 every update; global AMSGrad, beta1
D/G+E/prior .5/.5/0, beta2 .99, constant LR .0085, epsilon 1e-8, learned prior,
zero extra prior regularization/transport/noise/EMA and all other fields stay fixed.
The [research review](../r1r2-objective-memory-repair/research/score-law-literature.md)
and [saved-score diagnostic](../r1r2-objective-memory-repair/research/score-law/README.md)
motivated a weak-confidence trajectory hypothesis. This result rejects the
unchanged score-only repair; it does not isolate critic anchoring from changed
generator/encoder weighting or establish an endpoint gradient-starvation cause.

Seed 0, public deterministic initialization, task architecture, target/data law,
captured initial-tensor and ordered target-batch identities, initial prior law and
capacity, sampling, updates, evaluation cadence and gates remain fixed. Direct
pole consumes generator optimizer settings; token has no sampled prior. Words
retains the native joint encoder objective and explicit zero-width prior exception.
Named streams are checkpointed. The [collection audit](collection-audit.json)
passes certificates, caches, optimizer/penalty histories, source, streams, GIFs and
costs. It verifies captured initialization/data receipts without reconstructing
initial tensors, historical score pairs, forces or sampled-row ownership.

Execution commit: `517257ff3879516ae2da64009c375ffffe9b001a`. Scientific source:
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`;
runtime: `640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc`.
There are **21 certified cases/GIFs: 14 exact cached controls and seven new runs**.
New work costs **182.87967788797687 seconds**; cached work adds zero current spend.
The shared queue totals **210 unique attempts / 6,405.0795450999285 seconds**,
preserving all 203 original charges and owners. Full reservation stays 7,560
seconds, each recipe allowance 2,520, and the aggregate paid cap 8,000.
Bulk logs and checkpoints remain in
`/mnt/ml7tb/ParticleGAN-forge/r1r2-score-law-repair-20261012`.
Collection and diagnosis add no updates or sampling draws.

Reject this repair without adoption or qualification reuse. Any further paid
comparison needs a separately justified mechanism and finite registration under
the original full suite. No automatic loss grid, seed study, longer training,
added budget, Tier 2 or default change follows this result. The
[full-transport review](../r1r2-objective-memory-repair/research/full-transport.md)
selects no additional arm.

The [saved regularizer-balance fixture](research/regularizer-balance/README.md)
adds separate endpoint parameter-gradient evidence without training or qualification.
A [separately registered weaker-penalty contrast](../r1r2-weak-penalty-repair/PLAN.md)
retains all original tasks; its low-confidence hypothesis is distinct from this
closed loss-field comparison.
