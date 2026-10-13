# R1/R2 energy-U prior-freeze readout

**Reject this global repair.** Freezing the prior leaves **5/6 ordinary tasks
passing plus clock PASS**, with Ring16 still failing. Its exact learned-prior
parent has the same gate count. Covariance error barely improves and high-quality
mass worsens. The complete contrast is closed; Tier 1 remains unsolved.
The [technique inventory](../technique-inventory.md) remains the only current
generated leaderboard. This table compares two frozen global recipes.

| Recipe | Gaussian | Pole | Token | AE | Ring16 | Words | Clock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exact cached energy-U parent | [PASS](publication/media/energy_u-gaussian1d_smoke.gif) | [PASS](publication/media/energy_u-two_pole.gif) | [PASS](publication/media/energy_u-unused_token_hold.gif) | [PASS](publication/media/energy_u-ae_gan_hold.gif) | [FAIL](publication/media/energy_u-ring16_acquisition.gif) | [PASS](publication/media/energy_u-five_word_joint_smoke.gif) | [PASS](publication/media/energy_u-clockfree_audit_measurement_v1.gif) |
| Same recipe, frozen prior | [PASS](publication/media/frozen_prior-gaussian1d_smoke.gif) | [PASS](publication/media/frozen_prior-two_pole.gif) | [PASS](publication/media/frozen_prior-unused_token_hold.gif) | [PASS](publication/media/frozen_prior-ae_gan_hold.gif) | [FAIL](publication/media/frozen_prior-ring16_acquisition.gif) | [PASS](publication/media/frozen_prior-five_word_joint_smoke.gif) | [PASS](publication/media/frozen_prior-clockfree_audit_measurement_v1.gif) |

The [publication](publication/results.json) contains seven exact cached parent
certificates and seven fresh treatment results, with actual-training GIFs and
provenance sidecars for all fourteen cells. Parent certificates retain their
original source, initialization, request and charge owner.

| Ring16 endpoint metric | Learned parent | Frozen prior | Original bound |
| --- | ---: | ---: | ---: |
| Component covariance error | 6.371506 | 6.259248 | ≤.85 |
| Minimum eigen ratio | 1.027582 | .748672 | ≥.15 |
| HQ | .747559 | .552246 | ≥.85 |
| Modes | 16 | 16 | ≥16 |
| Mass TV | .030518 | .037598 | ≤.15 |
| Samples | 4096 | 4096 | ≥4096 |

Both recipes have zero full passes among 96 Ring16 observations and terminal
suffix zero against five required checks. At the endpoint both fail covariance
and HQ while meeting spread, modes, mass balance and sample count. The frozen
recipe's core-only covariance error is 2.189327, versus the parent's 1.156282;
these diagnostics replace no gate. Freezing locations does not rescue this
energy recipe; these results do not prove that prior motion was the sole cause.
Global covariance error .049517 and balanced mode occupancy do not establish
the required component density.
The [density audit](density-audit.json) and [readout](density-readout.md) verify
the saved panels and retain the complete terminal comparison.

Words completes all original 20,001 updates. Frozen prior confirms 24/24 checks,
first at update 834, versus the parent's 22/24, first at 2501. Both endpoints
pass generation and inverse checks: five modes, quality 1, exact inverse and
mass TV .018945. Frozen minimum correct-token probability is .999997 and
reconstruction NLL 6.08e-7. These acquisition passes grant no word-hold
qualification. Gaussian confirms acquisition at update 334 (11/24 passing
checks), versus 209 (13/24) for the parent. Both endpoints fail KS: frozen
.051144 and parent .068194 exceed .05. Preserve the original acquisition grades
without claiming retention. Pole/token/AE retain terminal suffixes 12/19/24;
frozen AE reconstruction MSE .003334 and hold .023880 meet ≤.05/≤.35.
Each recipe's own clock passes all four parity comparisons.

The [saved-only research](../r1r2-energy-distance-repair/research/README.md)
motivated the single [planned delta](PLAN.md): `prior_update: learned → frozen`
on the global recipe and all seven effective task bindings. Raw signed energy-U
weight 1, native non-saturating loss, Adam AMSGrad m110, LR .0085, positive
paired R1/R2 gamma .1 every update and all other trainer fields remain fixed.
Remove only the false learned-location capability requirement from candidate
metadata. Gaussian/Ring/AE/word initialized latent tables become buffers with
zero prior updates; G/E/D retain full updates. Direct pole coordinates and token
embeddings remain trainable on their original generator paths because their
prior applicability is `not_sampled`. Initial prior distribution, width and
capacity, architecture, target/data law, ordered batches, sampling, full update
budgets, cadence and gates remain fixed. Seed 0 uses the public deterministic
initializer; actual initial tensors and every consumed named stream match.
All nineteen original evaluator source pins remain byte-identical.

The [registration](registration.json) freezes 34 declarations. The independent
[readiness review](preflight-review.json), [readiness receipt](readiness-validation.json)
and actual [registration review](registration-review.json) precede paid
admission. Software evidence contains 67 workflow checks, 107 pure auditor
checks and eight bounded CPU integration checks. The latter execute 14 actual
software updates across all branches and explicitly synthetic word prefix
counters/producer labels. They earn no trained grade; the partial original word
hold reducer remains INCOMPLETE. Their separate
[fixture receipt](software-validation.json) retains those limits.
The independent [collection audit](collection-audit.json) verifies original
cached identities, fresh owners, optimizer groups and histories, fixed prior
buffers, positive R1/R2 and signed energy consumption, checkpoints, streams,
protocols, media and costs. Clock branches start from their own warmed states;
cross-recipe constructor receipts and semantic streams match, with no shared
pre-warmup tensor copy claimed. Saved audits add zero model constructors,
random draws or optimizer updates. The final [validation receipt](validation.json)
retains compilation and historical evidence preservation.

New execution is `3203c4e419cb7c2a4cd2021d2c48638980579901`; cached parent
certificates keep `67920d9090727bd74adbd331187f0e6f22533e48`. Both bind scientific
source `2b04f2853c511c9bcd417ef41e8fd773e97e8c67c9e057da8abdd4fef2de5f12`
(1,216 unchanged files) and runtime
`640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc`.
The original content-addressed source snapshot retains its original parent
metadata; actual new requests identify the new execution commit. No cached
certificate is relabeled. The reviewed collector exposes all 58 planned keys
and admits only the fourteen Tier 1 cells; higher tiers and foreign jobs remain
unclaimed by this comparison.

The contrast adds **seven fresh attempts / 197.333891441056 seconds**, with
seven zero-charge cache reuses. The shared queue totals **238 unique attempts /
7,168.248061007995 seconds**, preserving all 231 earlier charge identities and
owners under the unchanged 8,000-second cap. Full allowances remain 5,040 seconds
combined / 2,520 per recipe. All six shorter new jobs completed for 61.544201
seconds before the full 900-second word reservation. All ten archives remain
within the 8 GiB ceiling and 5 GiB free-space floor. Bulk logs, JSONL streams and
checkpoints remain outside Git in
`/mnt/ml7tb/ParticleGAN-forge/r1r2-energy-prior-freeze-repair-20261012`.

Retain the stronger non-saturating control and both energy negatives; adopt
neither repair. The [least-squares review](../r1r2-energy-distance-repair/research/least-squares/README.md)
remains a deferred, low-confidence distinct hypothesis. Remaining paid headroom
is 831.751939 seconds, below one unchanged 900-second word reservation, so a
further complete comparison cannot fit this campaign. No extra arm, seed repeat,
longer training, budget increase, qualification or Tier 2 follows automatically.

Collect and audit saved evidence from the execution checkout with the declared
project interpreter and single-thread environment in [PLAN.md](PLAN.md):

```sh
python reports/forge/r1r2-energy-prior-freeze-repair/workflow.py collect \
  --manifest reports/forge/r1r2-energy-prior-freeze-repair/manifest.json \
  --artifacts /mnt/ml7tb/ParticleGAN-forge/r1r2-energy-prior-freeze-repair-20261012 --media
python reports/forge/r1r2-energy-prior-freeze-repair/audit_collection.py \
  --execution-commit 3203c4e419cb7c2a4cd2021d2c48638980579901
python reports/forge/r1r2-energy-prior-freeze-repair/density_audit.py \
  --execution-commit 3203c4e419cb7c2a4cd2021d2c48638980579901
tail -F /mnt/ml7tb/ParticleGAN-forge/r1r2-energy-prior-freeze-repair-20261012/run.log \
  /mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012/queue/r1r2-energy-prior-freeze-repair-v1/*/run.log
```
