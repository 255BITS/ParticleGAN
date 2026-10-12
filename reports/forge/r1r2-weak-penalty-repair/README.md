# R1/R2 weaker-penalty readout

**Reject the weaker-penalty repair.** Reducing paired R1/R2 gamma from .1 to .01
on exact non-saturating AMSGrad m110 leaves **2/6 ordinary passes plus clock PASS**.
Gaussian, pole, Ring16 and words fail. Its cached parent retains 5/6 plus clock.
This finite comparison is closed, without qualification reuse or default adoption.
The [technique inventory](../technique-inventory.md) remains the only current
generated leaderboard; this is an unranked, frozen comparison of global recipes.

| Recipe | Gaussian | Pole | Token | AE | Ring16 | Words | Clock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Scheduled reference, cached | [PASS](publication/media/baseline-gaussian1d_smoke.gif) | [PASS](publication/media/baseline-two_pole.gif) | [PASS](publication/media/baseline-unused_token_hold.gif) | [PASS](publication/media/baseline-ae_gan_hold.gif) | [FAIL](publication/media/baseline-ring16_acquisition.gif) | [PASS](publication/media/baseline-five_word_joint_smoke.gif) | [FAIL](publication/media/baseline-clockfree_audit_measurement_v1.gif) |
| Non-saturating AMSGrad parent, gamma .1, cached | [PASS](publication/media/non_saturating-gaussian1d_smoke.gif) | [PASS](publication/media/non_saturating-two_pole.gif) | [PASS](publication/media/non_saturating-unused_token_hold.gif) | [PASS](publication/media/non_saturating-ae_gan_hold.gif) | [FAIL](publication/media/non_saturating-ring16_acquisition.gif) | [PASS](publication/media/non_saturating-five_word_joint_smoke.gif) | [PASS](publication/media/non_saturating-clockfree_audit_measurement_v1.gif) |
| Same global recipe, gamma .01 | [FAIL](publication/media/weak_gamma-gaussian1d_smoke.gif) | [FAIL](publication/media/weak_gamma-two_pole.gif) | [PASS](publication/media/weak_gamma-unused_token_hold.gif) | [PASS](publication/media/weak_gamma-ae_gan_hold.gif) | [FAIL](publication/media/weak_gamma-ring16_acquisition.gif) | [FAIL](publication/media/weak_gamma-five_word_joint_smoke.gif) | [PASS](publication/media/weak_gamma-clockfree_audit_measurement_v1.gif) |

Every cell has a certified saved grade, actual-training GIF and provenance sidecar.
Gaussian/word gates require independently confirmed acquisition; pole, token, AE
and Ring retain their original sustained gates. Clock is a separate proof.

## Failures and retained passes

Gaussian has **0/24 passing checks and no confirmed acquisition**. Its endpoint
KS .373359 exceeds .05, mean error .991625 sigma exceeds .2, and standard-deviation
ratio 1.949833 exceeds 1.2. The cached non-saturating parent retains acquisition
PASS at update 292 (15/24 passing checks), despite endpoint KS .057647 >.05.
The earlier [AMSGrad parent](../r1r2-score-law-repair/README.md) likewise retains
acquisition PASS while endpoint KS .069604 fails. These original grades do not
establish retention. The scheduled reference's endpoint also fails.

Pole's endpoint movement **1.033388 ≥.3** and gradient median **.592705 ≤1** both
pass their bounds. Its original gate still **FAILS**: only 4/24 observations pass,
with a terminal suffix **2/5** and no confirmed sustained convergence. A passing
endpoint cannot replace the five-check requirement; the parent retains suffix 12.

| Ring16 endpoint | Gamma .1 parent | Gamma .01 | Original bound |
| --- | ---: | ---: | ---: |
| Component covariance error | 1.556546 | 4.625256 | ≤.85 |
| Minimum eigen ratio | .094075 | .189267 | ≥.15 |
| Modes | 16 | 15 | ≥16 |
| Mass TV | .049805 | .180664 | ≤.15 |
| HQ | .888672 | .899902 | ≥.85 |

Weak gamma worsens covariance and occupancy despite passing endpoint spread/HQ.
Covariance and mode coverage fail all 96 evaluations; full passes and terminal
suffix remain zero. Best covariance is 4.198951 at update 1484. All five terminal
checks fail covariance, modes and mass TV. The [density audit](density-audit.json)
reproduces all six original gate metrics across 96 retained panels exactly.

Words completes the original **20,001 updates**, with **0/24 independent
confirmations** and a failing endpoint: three modes, mass TV .401758, exact inverse
reconstruction 0 and minimum correct-token probability about 7.06e-24. Quality 1
does not establish full vocabulary coverage or inversion. The parent confirms
24/24 from update 834 and passes its endpoint. The scheduled reference retains
its acquisition PASS (1/24 confirmations at 2501), despite failed endpoint
coverage/inversion. No word continuation qualification follows.

Token and AE retain sustained passes with terminal suffixes 8 and 20, exceeding
five. Token concept movement .909217 and unused hold .994165 exceed .85; AE
reconstruction MSE .026328 and hold .024521 meet ≤.05/≤.35. Its own clock passes
all four parity comparisons.

## Recipe, provenance and cost

The [plan](PLAN.md), [research rationale](research/weak-penalty.md),
[registration](registration.json), [preflight](preflight-review.json) and
[certified publication](publication/results.json) bind exactly one
`floor_constant` delta: **`reg_coeff: .1 → .01`**. Both positive real/fake
zero-centered penalties remain every update, with gamma/2 multiplying their summed
mean squared input gradients. Native non-saturating loss, global Adam AMSGrad,
D/G+E/prior beta1 .5/.5/0, beta2 .99, constant LR .0085, multipliers 1, epsilon
1e-8, learned prior and zero extra prior regularization/transport/noise/EMA remain
fixed. This treatment rejects the proposed global repair; it does not establish
one historical force or causal instability explanation.

All tasks retain seed 0/public deterministic initialization, architecture,
target/data law, initial prior distribution/capacity, sampling, updates, cadence
and gates. Captured initial-tensor and ordered-batch receipts match the parent;
every consumed named stream is checkpointed. Direct pole remains generator-owned,
token has no sampled prior, and words retains its zero-width exception and joint
encoder objective. The passing [collection audit](collection-audit.json) verifies
certificates, exact caches, optimizer/penalty histories, streams, media and costs.
It verifies recorded initialization/data identities without reconstructing initial
tensors, historical paired forces or sampled-row ownership. Diagnosis and
collection add zero updates or sampling draws.

Execution commit: `3cb46b91121f42f8b171a77f17776365aebad1d8`. Scientific source
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507` and runtime
`640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc` are unchanged.
There are **21 certified cases/GIFs: 14 exact cached controls and seven new runs**.
New work costs **181.08289201298612 seconds**, with zero current cached spend.
The shared queue totals **217 unique attempts / 6,586.162437112915 seconds**,
preserving all 210 prior charges and owners. Full reservation stays 7,560 seconds,
each recipe allowance 2,520, and aggregate paid cap 8,000. Bulk logs and states
remain in `/mnt/ml7tb/ParticleGAN-forge/r1r2-weak-penalty-repair-20261012`.

Close this contrast without adoption. Preserve the stronger parent's acquisition
passes and unresolved Ring failure. Further work requires a distinct justified
mechanism and separate finite registration under the original complete suite;
no automatic penalty grid, seed study, longer training, added budget or Tier 2.
