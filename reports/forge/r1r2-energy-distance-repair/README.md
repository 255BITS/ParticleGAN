# R1/R2 raw energy-distance readout

**Reject this global repair.** Adding raw signed energy-U at weight 1 to
non-saturating AMSGrad m110 preserves **5/6 ordinary passes and clock PASS**, but
fails Ring16 density. The fresh-source control has the same gate count. Energy
restores component spread while worsening covariance error and high-quality
mass. The finite contrast is closed; no qualification or default changes follow.
The [technique inventory](../technique-inventory.md) remains the only current
generated leaderboard. This table compares two frozen global recipes.

| Recipe | Gaussian | Pole | Token | AE | Ring16 | Words | Clock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fresh non-saturating AMSGrad control | [PASS](publication/media/non_saturating-gaussian1d_smoke.gif) | [PASS](publication/media/non_saturating-two_pole.gif) | [PASS](publication/media/non_saturating-unused_token_hold.gif) | [PASS](publication/media/non_saturating-ae_gan_hold.gif) | [FAIL](publication/media/non_saturating-ring16_acquisition.gif) | [PASS](publication/media/non_saturating-five_word_joint_smoke.gif) | [PASS](publication/media/non_saturating-clockfree_audit_measurement_v1.gif) |
| Same recipe, energy-U weight 1 | [PASS](publication/media/energy_u-gaussian1d_smoke.gif) | [PASS](publication/media/energy_u-two_pole.gif) | [PASS](publication/media/energy_u-unused_token_hold.gif) | [PASS](publication/media/energy_u-ae_gan_hold.gif) | [FAIL](publication/media/energy_u-ring16_acquisition.gif) | [PASS](publication/media/energy_u-five_word_joint_smoke.gif) | [PASS](publication/media/energy_u-clockfree_audit_measurement_v1.gif) |

Every cell has its own fresh certified result, actual-training GIF and provenance
sidecar in the [publication](publication/results.json). Historical baseline
grades fill no new-source cell.

## Remaining failure

| Ring16 endpoint metric | Fresh control | Energy-U | Original bound |
| --- | ---: | ---: | ---: |
| Component covariance error | 1.556546 | 6.371506 | ≤.85 |
| Minimum eigen ratio | .094075 | 1.027582 | ≥.15 |
| HQ | .888672 | .747559 | ≥.85 |
| Modes | 16 | 16 | ≥16 |
| Mass TV | .049805 | .030518 | ≤.15 |
| Samples | 4096 | 4096 | ≥4096 |

Both recipes have zero full passes among 96 observations and terminal suffix
zero against five required checks. The control fails covariance and minimum
spread at the endpoint; energy fails covariance and HQ. Balanced occupancy and
restored spread do not establish component density. The
[density audit](density-audit.json) independently reproduces all six gate
metrics on both recipes' 192 saved panels without new sampling or updates.
The [density readout](density-readout.md) retains the full terminal comparison
and distinguishes mapped-center dispersion from local noise geometry.

Words completes the original 20,001 updates in both recipes. The control confirms
24/24 checks, first at update 834; energy confirms 22/24, first at 2501. Both
endpoints pass all original generation and inverse checks, with five modes,
quality 1, exact inverse reconstruction and mass TV .018945. Energy's minimum
correct-token probability is .999996, versus .999085 for the control. These
acquisition passes grant no word-hold qualification.

Gaussian confirms acquisition at update 292 for the control (15/24 passing
checks) and 209 for energy (13/24). Both endpoints fail KS: .057647 and .068194
exceed .05. Preserve their original acquisition grades without claiming
retention. Pole/token/AE retain sustained passes: terminal suffixes are 12/19/24
for energy and 12/19/24 for the control, above five. Energy's AE reconstruction
MSE .003729 and hold .012141 meet ≤.05/≤.35. Each recipe's own clock passes all
four parity comparisons.

## Delta and evidence

The [plan](PLAN.md), [API documentation](API.md), [registration](registration.json)
and [preflight](preflight-review.json) declare only
`energy_distance_weight: 0 → 1`. The added generator loss is
`2 mean ||fake-real|| - off-diagonal mean ||fake-fake|| - off-diagonal mean ||real-real||`.
It uses existing G-phase outputs and targets, detaches targets, retains negative
finite estimates and chooses zero derivative at exact collisions. It adds no
normalizer, bandwidth, labels, target draws or evaluation samples. Component
hosts inherit their original loops and objectives; signed accounting is separate
from nonnegative transport diagnostics. The public default remains zero.

Both positive real/fake R1/R2 penalties remain gamma .1 every update. Native
non-saturating loss, Adam AMSGrad, D/G+E/prior beta1 .5/.5/0, beta2 .99,
constant LR .0085, multipliers 1, epsilon 1e-8, learned prior and zero extra prior
regularization/transport/noise/EMA remain fixed. Tasks retain architecture,
target law, initial prior distribution/capacity, sampling, update budget,
evaluation cadence and all gates. Seed 0/public initialization, actual initial
tensors and ordered batch receipts match across recipes. Every consumed named
stream is checkpointed. Direct pole remains generator-owned, token has no sampled
prior, and words keeps its declared zero-width prior exception and joint inverse
objective. All 19 original evaluator source pins remain byte-identical.

The independent [collection audit](collection-audit.json) passes all 14
certificates/media, 12 ordinary final states, both own-clock proofs, actual
energy consumption/bindings, optimizer/penalty histories, streams and costs.
Clock branches each start from their own warmed state; cross-recipe constructor
receipts and stream bytes match, while no pre-warmup tensor copy is claimed.
The audit verifies saved evidence and adds zero constructors, random draws or
optimizer updates. Software prerequisites are separately retained in
[software-validation.json](software-validation.json), including 108 focused and
133 adjacent passing checks, 191 workflow checks across the recorded stages,
and 70 saved-evidence auditor fixture checks. They confer no trained grade.
The final [validation receipt](validation.json) also checks compilation freshness
and preserves all 513 historical publication records and eight qualification
or leaderboard snapshots exactly.

Scientific execution is `67920d9090727bd74adbd331187f0e6f22533e48`, source
`2b04f2853c511c9bcd417ef41e8fd773e97e8c67c9e057da8abdd4fef2de5f12`
(1,216 files), runtime
`640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc`.
An initial report coordinator exposed only 14 keys to a collector that expected
all 58 planned bookkeeping keys. It stopped before any worker claim or charge.
The separately reviewed [administrative driver](administrative_drain.py),
[receipt](administrative-drain-receipt.json) and 12 passing guards restore that
collection view while limiting claims to the original 14 Tier 1 jobs. Coordinator
commit `21e207ef6c72d74ecc2f684fba8188851b060a15` changes no frozen scientific
byte, task, declaration, request ID or child execution source. No paid retry
was needed; higher-tier and foreign jobs remain untouched.

The comparison adds **14 fresh attempts / 384.751732454024 seconds**. The shared
queue totals **231 unique attempts / 6,970.914169566939 seconds**, retaining all
217 prior charges and owners under the unchanged 8,000-second cap. Full declared
allowance remains 5,040 seconds combined / 2,520 per recipe, including each
900-second word reservation. Combined storage remains within the 8 GiB ceiling
and 5 GiB free floor. Bulk logs/checkpoints are archived outside Git in
`/mnt/ml7tb/ParticleGAN-forge/r1r2-energy-distance-repair-20261012`.
Tail `administrative-drain.log` there for the execution summary; per-run logs
remain under the shared queue's campaign directory.

Preserve the stronger control and this negative result. A further global repair
needs a distinct mechanism, saved-evidence justification and separate finite
registration under the original tasks and remaining cap. This contrast permits
no automatic coefficient grid, seed repeats, longer training or Tier 2.
The subsequent [saved-geometry research](research/README.md) records one prior
policy hypothesis and its separate prerequisites. The
[least-squares review](research/least-squares/README.md) remains a deferred
alternative; neither supplies a passing trained result to this closed contrast.

Reproduce from the scientific execution checkout using the declared project
interpreter. Collection is saved-state work:

```sh
python reports/forge/r1r2-energy-distance-repair/workflow.py collect \
  --manifest reports/forge/r1r2-energy-distance-repair/manifest.json \
  --registration reports/forge/r1r2-energy-distance-repair/registration.json \
  --artifacts /mnt/ml7tb/ParticleGAN-forge/r1r2-energy-distance-repair-20261012 \
  --output reports/forge/r1r2-energy-distance-repair/publication --media
python reports/forge/r1r2-energy-distance-repair/audit_collection.py \
  --execution-commit 67920d9090727bd74adbd331187f0e6f22533e48
python reports/forge/r1r2-energy-distance-repair/density_audit.py \
  --execution-commit 67920d9090727bd74adbd331187f0e6f22533e48
```
