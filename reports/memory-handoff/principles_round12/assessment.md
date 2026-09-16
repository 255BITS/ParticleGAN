# Round12 scout assessment

Completed12 fresh2k scouts and two exact2k→5k continuations on both GPUs, with
zero failures. The new leading recipe is shuffled mismatched-history ranking
at weight.25. Its5k checkpoint leads the fixed minimum-warm-Q ranking by only
~1% over2k; retain both because2k has better late-window Q. Full-circle success
is still0/128 everywhere. No jobs remain. See [extension assessment](followup/assessment.md).
Selection followed the fixed gates in plan.md; no further extensions are justified.

All scout full-circle passes remain0/128 for cold256/1024 and warm256/1024
at prefixes8/32. None stops late. Q gains describe continuous progress, not
success probability or a solved circle. No images were used for ranking.

## Completed scout leaderboard

| Model | Q prefix8 | Q prefix32 | Radial32 ↓ | Nearest-history rank32 | Train seconds |
|---|---:|---:|---:|---:|---:|
| match_shuffle25 | 0.010901 | 0.011161 | 0.945 | 88.3% | 241.8 |
| match_nearest25 | 0.009982 | 0.009932 | 0.994 | 82.8% | 238.9 |
| future_arch_control | 0.008669 | 0.008853 | 1.101 | 71.1% | 178.1 |
| match25_future10 | 0.008987 | 0.008375 | 1.624 | 86.7% | 332.3 |
| proposal_mixed_pair25 | 0.008180 | 0.008274 | 1.159 | 67.2% | 175.5 |
| match_nearest50 | 0.007530 | 0.007444 | 1.233 | 85.9% | 234.4 |
| match25_recover10_future10 | 0.007361 | 0.007291 | 1.808 | 87.5% | 345.9 |
| future25 | 0.007089 | 0.007938 | 1.477 | 68.0% | 264.9 |
| match_nearest10 | 0.007114 | 0.006758 | 1.854 | 80.5% | 234.4 |
| recover_noise30 | 0.005411 | 0.005798 | 1.347 | 67.2% | 211.3 |
| recover_noise10 | 0.004881 | 0.005318 | 1.386 | 49.2% | 205.0 |
| match25_recover10 | 0.004836 | 0.004741 | 1.285 | 85.2% | 263.1 |
| future10 | 0.004092 | 0.004635 | 2.421 | 74.2% | 265.9 |

## What the scouts establish

- Mismatch training helps in this panel. Shuffled negatives at weight.25 improve
  warm Q33–35%; nearest negatives at.25 improve20–22%. Both improve radial error,
  lateQ and direction agreement enough to meet the predeclared extension gates.
- Weight matters: nearest.1 and.5 regress autonomous quality despite better D
  ranking. Shuffled negatives also improve the harder nearest-history evaluation;
  special nearest-donor mining is not necessary for this round’s strongest result.
- Both prefix-noise recovery scales regress. Adding noise.1 to nearest.25 also
  regresses. This rejects the tested observation-disturbance recipes, not all
  possible state-repair mechanisms.
- Direct future GAN weights.1/.25 lose to both the saved baseline and their
  architecture control. Combining nearest.25 with future.1 does not qualify:
  Q is near baseline and radial error is worse. Three-way combination also loses.
- The future architecture control improves Q6–7%, insufficient for promotion.
  Additional input dimensions change initialization/capacity; use this control
  when interpreting the query objective. Future D adds a separate head.

## Diagnostics and interpretation

The baseline ranks real next points above nearby mismatched-history next points
65–70% across prefixes8/32/48. Shuffled mismatch training reaches86–93%, and
nearly always rejects the earlier/later-point comparisons. D therefore learns
a stronger continuation signal. These fixed-panel comparisons are descriptive,
not statistical significance claims; mismatched points can be plausible under noise.

At real prefix32, positive alignment of ascending point-head score with the
target-minus-generated direction is60% baseline,70% shuffled. At late autonomous
state it is25%/33%; real-state restoration brings it to63%/67%. These probes use
a particular timed target, so phase drift and observation noise confound any
claim that a negative cosine means no useful orbit repair direction exists.
See gradient_diagnostics.md. The probe never enters training.

Baseline late restoration changes next-point MSE1.653→.0114 with clock288 fixed,
but the following128-step Q is still only.0188 and all passes remain zero.
Encoding all288 real observations versus only the latest32 gives similar results.
Resetting the clock further improves the immediate sample, not sustained fidelity.
This implicates autonomous state maintenance as a useful next target without
establishing whether original-process information is erased or unused.

Late process response also remains weak. Shuffled mismatch2k has median radius
response.025 and speed response.0013 (ideal1), with both original/flipped mean
directions correct for9.4% of the panel. Better point discrimination and better
Q do not yet establish retention of the requested process.

## Implementation and validation

Config-controlled default-off mismatch ranking, prefix-noise pair recovery, and
joint future queries. D retains the only persistent state; G uses the existing
proposal adapter and fixed particle. No generated training rollout: two local
outputs/one intervening generated write maximum. Future queries make independent
reads with no writes. Runtime always uses offset0 and requires no expert.

All training objectives here are GAN losses plus the unchanged default prior
regularizer and exact B-cap. No MSE training, clipping, EMA, geometry labels,
moving cursor, B-cap override or seed sweep. Diagnostic MSE is evaluation only.
70 focused tests pass, including causal future queries, clean recovery judging,
writer/G gradient ownership, active exact B-cap, all-objective exact resume,
no-MSE training and analytic diagnostic gradient direction. Two full-batch GPU
four-update smokes include1024 evaluation. Default-off trainer is bitwise equal
to committed9b85923 for four CPU updates including scheduled B-cap.

All twelve scouts archive identical training sources. Scout queue wall1593.5s (~26.6min),
3016 training GPU-seconds (~50.3GPU-min). Source hashes and execution accounting
are in execution.json. Only completed checkpoints entered reporting/diagnostics.

Artifacts: [plan](plan.md), [leaderboard](leaderboard.md), [results](results.json),
[selection](extension_decision.json), [signal probes](signal.json),
[baseline probes](signal_baseline.json), [process probes](process_scouts.json),
[validation](validation.json).

Stable log: `tail -F runs/memory_path/core_round1/train.log`
