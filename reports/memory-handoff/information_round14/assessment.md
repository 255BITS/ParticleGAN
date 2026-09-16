# Round14: diagnose information loss, test future history ranking

Five2k scouts completed on both GPUs, zero failures. All full cold/warm passes
remain0/128 at256/1024 and warm prefixes8/32. Late stopping remains0%.
No scout qualified for extension under the predeclared gates. The previous
round12 match_shuffle25 recipe remains the winner; keep5k as nominal reference
and2k as the stronger late-quality control. No new training is queued.

## Leaderboard

Minimum Q is the worse warm1024 quality across prefixes8/32; higher is better.
It is a continuous diagnostic, not a success probability.

| Model | Minimum warm Q | Q32 | Late Q32 | Radial32 ↓ |
|---|---:|---:|---:|---:|
| saved match_shuffle25_5k | .011008 | .011099 | .009708 | .912 |
| saved match_shuffle25 (2k) | .010901 | .011161 | .010869 | .945 |
| future_full10 | .009970 | .010040 | .008820 | .909 |
| future_mixed25 | .008107 | .008107 | .006609 | 1.063 |
| future_mixed10 | .007403 | .007403 | .006515 | 1.071 |
| future_mixed10_detachwrite | .006626 | .006626 | .005190 | 1.092 |
| future_clean10 | .006507 | .006577 | .005758 | 1.333 |

The best new scout, future_full10, slightly improves radial error but loses
8.5–10.0% Q and13.0–18.9% late Q against the saved2k control. It does not merit
extension. The generated-context writer gradient helps compared with detaching
that branch: mixed10 beats its matched control, while both lose to the baseline.
Unlike round13's broader control, this one retains clean-future and original
immediate-mismatch writer gradients. This is one fixed training comparison,
not a significance claim or proof of a particular internal mechanism.

## Diagnosis that selected the scouts

Frozen baseline checkpoints were probed using2048/512/1024 disjoint diagnostic
train/validation/test histories. Real and generated memory states were collected
after0/1/8/32/128 writes following a32-point prefix. Fixed learned particles were
sampled independently per episode; episodes are held out, particles are not.
Linear ridge and nonlinear probes used train-only normalization and validation
selection, with z-only, shuffled-memory, M+z and oracle controls.

For the5k baseline, generated-memory nonlinear radius/speed R² drops from
.599/.954 at handoff to .126/.436 after32 writes and approximately zero after128.
Matched real-history memory at128 retains .574/.946. M+z does not rescue late
states. Real-trained probes transfer worse than generated-domain probes early,
indicating representation shift as well as loss of readily decodable information.
Finite probe failure is not proof of information-theoretic erasure. Real-state
radius decoding is only moderate, so probe capacity is a material limitation.

The preceding implementation audit reproduced saved trajectories exactly and
found severe errors already within clock support. Existing real-state restoration
also restores next-point accuracy at a late clock. Together these findings
motivated an incentive to preserve predictive information through writes.

## What the new objective did

All five scouts preserve the winning immediate clean mismatch loss. A new D-only
branch ranks original-episode observations4/12 steps ahead against other episodes
using the same point head that G sees at horizon0. A zero-initialized projection
adds explicit horizon conditioning before the first activation. Clean/mixed/full
replacement settings and weights.10/.25 are in plan.md and the configs.

It adds at most one detached G proposal and one write in an independent branch,
averages losses over contexts/horizons, and uses unchanged exact B-cap. It never
generates those future targets, adds no G future loss, and uses no geometry labels
or MSE training objective. Both anchors and donors require available futures;
targets cannot enter the memory or donor selection.

## Better clean information did not produce persistent autonomous information

The same held-out decoding protocol was run on all five completed scouts.
The strongest future weight improves clean-state radius/speed R² to .685/.965,
versus .612/.942 for the2k baseline. After32 generated writes it falls to
.112/.190, below the baseline's .127/.267. By128 it is approximately at chance.
All five scouts show near-chance late decoding; M+z does not rescue it.

Clean-only future ranking has the best generated32 decoding among new scouts
(.170 radius/.371 speed), yet the worst autonomous Q. Thus decoding a process
property alone does not establish that G uses it correctly or that the trajectory
is accurate. Full-write has the best new Q but generated32 decoding only
.122/.292. Preserve both types of measurement.

The precise gradient control improves clean/early decodability when connected:
mixed10 vs detached has clean radius/speed .616/.949 vs .553/.934, and generated8
.375/.851 vs .346/.760. At32 there is no consistent advantage across targets,
and both are at chance by128. See scout_information.md for the complete table.

The future-ranking head learns a modest distinction on the harder nearest-history
panel: strongest-weight h12 clean/full-write ranking reaches59.4%/58.6%, versus
48.4%/49.2% for the horizon-blind2k reference. Its immediate clean ranking falls
to82.8% from88.3%. The task itself remains imperfectly learned, and improved
future ranking does not establish good immediate generator gradients.

Matched-history output interventions agree with the retention failure. Full-write
median speed response falls .749→.122→approximately0 in32-point windows starting
after0/32/128 writes. Stronger-weight falls .863→.171→-.012. Late radius responses
are also near zero. These are output-window responses, not instantaneous memory
measurements, and full-write ranking uses each model's own generated corruption.

## Recommendation

Retain the previous winner. Do not extend these scouts or increase future-ranking
weight on the basis of better clean-state probes. This round successfully tests
the selected hypothesis but does not solve the circle task.

The next mechanistic target is preservation during repeated writes. A possible
next experiment would couple a learned separation of persistent process content
and changing observation state to a local adversarial preservation incentive.
Earlier fixed slow/fast memory and private G-GRU scouts already failed; any new
comparison needs that explicit interaction and matched controls, not another
memory-size/rate sweep. This is a proposal for discussion, not a selected or
queued experiment. Local training has not been proven incapable of solving it.

## Validation and cost

77 distinct focused tests passed:64 existing/information tests plus13 new
future-ranking/diagnostic cases. Coverage includes causality and donor bounds,
gradient ownership, active exact B-cap, four-update bitwise offset0 architecture
equivalence, exact resume and no-MSE training. Two4-step full-batch GPU smokes
with1024-step evaluation passed. Defaults/old checkpoints remain supported.

All five scouts archive identical training source hashes; reference paths and
observed prefixes are bitwise identical to the baseline. No training sources
changed during the queue. Queue wall967.9s (~16.1min), training1609.2 GPU-seconds
(~26.8GPU-min), excluding implementation, smokes and evaluation probes. No
failures, retries, extensions, clipping, EMA, B-cap overrides, seed repeats,
full generated training rollout or runtime expert.

Artifacts: plan.md, leaderboard.md/results.json, extension_decision.json,
diagnosis.md/json, scout_information.md/json, diagnostics.md, process.json,
signal.json, future_signal.json, execution.json, validation.json. Stable log:
`tail -F runs/memory_path/core_round1/train.log`.

Included in the user-requested commit with round13 and the implementation audit.
Push was not requested. Next discussion: whether sequential training needs
special treatment; see next.md. No next experiments selected or queued.
