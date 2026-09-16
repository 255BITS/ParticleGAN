# Local adversarial recovery scouts

Sixteen2k scouts completed with zero failures. Queue wall time
1018.3s; aggregate training1914.1s.
All scouts have0/128 full cold-circle and warm original-orbit passes at256/1024
and both prefixes8/32. All retain zero late stopping. Toy remains unsolved.

## Continuous progress

Q measures simultaneous radial and signed angular-step fidelity. Perfect clean
motion scores1; the saved noisy expert calibration scores about.515 (shorter
independent224/248-step panel). It is not a probability of success. Absolute
phase is assessed separately by startup/position error. See metric_calibration.json.

| Scout | Q prefix8 /32 | Late Q prefix8 /32 | Radial prefix32 | Longest good arc, prefix32 (turns) |
|---|---:|---:|---:|---:|
| proposal_mixed_pair25 | 0.00818 / 0.00827 | 0.00666 / 0.00677 | 1.159 | 0.059 |
| proposal_pair25 | 0.00709 / 0.00671 | 0.00515 / 0.00528 | 1.636 | 0.061 |
| proposal_mixed_uniform50 | 0.00661 / 0.00721 | 0.00554 / 0.00608 | 1.355 | 0.055 |
| proposal_mixed_pair50 | 0.00647 / 0.00644 | 0.00343 / 0.00393 | 2.158 | 0.062 |
| proposal_clean_s25 | 0.00619 / 0.00643 | 0.00577 / 0.00607 | 1.487 | 0.054 |
| plain_pair50 | 0.00587 / 0.00641 | 0.00438 / 0.00435 | 1.679 | 0.057 |
| proposal_pair50 | 0.00586 / 0.00613 | 0.00358 / 0.00365 | 2.041 | 0.064 |
| proposal_mixed_fixed4375 | 0.00568 / 0.00575 | 0.00373 / 0.00483 | 1.460 | 0.049 |
| proposal_mixed_uniform_pair25 | 0.00543 / 0.00553 | 0.00464 / 0.00461 | 1.424 | 0.039 |
| plain_mixed_s25 | 0.00470 / 0.00480 | 0.00345 / 0.00398 | 2.026 | 0.034 |
| proposal_mixed_shared25 | 0.00439 / 0.00460 | 0.00335 / 0.00395 | 1.777 | 0.039 |
| proposal_mixed_s25 | 0.00369 / 0.00374 | 0.00274 / 0.00330 | 1.871 | 0.029 |
| proposal_clean_uniform50 | 0.00380 / 0.00343 | 0.00216 / 0.00228 | 2.241 | 0.040 |
| proposal_mixed_shared75 | 0.00263 / 0.00274 | 0.00169 / 0.00177 | 1.638 | 0.029 |
| proposal_mixed_mild_full | 0.00223 / 0.00258 | 0.00143 / 0.00197 | 2.152 | 0.031 |
| proposal_shared_s25 | 0.00218 / 0.00256 | 0.00172 / 0.00187 | 1.577 | 0.028 |

## What the scouts support

- proposal_mixed_pair25 has the highest worst-prefix Q, improving about32%/29%
  over the fresh old candidate. Radial error falls24.1%/22.1%. Late Q improves
  at both prefixes, direction is essentially unchanged, and stopping remains0.
  Its first32 position error worsens1.572->1.704 /1.448->1.528. This is progress
  in local orbit motion, not improved phase fidelity or a solved circle.
- The pair loss at25% helps most when combined with mixed judging and mild
  replacement. Pair25 without exploration improves Q less and has worse radial
  and late-quality metrics than the combined model. Mixed judging alone is worse.
- Pair50 is not better: the mixed pair50 model loses late quality and radial
  fidelity. More transition-loss weight is not automatically more effective.
- Uniform replacement improves mixed judging compared with its fixed.25 match,
  but hurts clean judging. It also hurts the winning mixed pair25 combination.
  The mild/full mixture is worse than fixed.4375 with the same expected strength.
  Recovery strength interacts with the judging/task; broader coverage alone fails.
- Comparisons involving the pair branch change D-head capacity and compute as
  well as the observation judged. This is not a parameter-matched causal isolation.

## Memory and repair probes

The previous candidate's history-dependent outputs do not reliably preserve
radius/speed. Matched-prefix interventions have near-zero median process response;
only5.5% get BOTH original and reversed directions right by late mean angular
motion. The new mixed pair25 candidate also has near-zero median radius/speed
response; the corresponding direction fraction is10.9%. Both remain far from
consistent process preservation. See baseline_probes.md and process_*.json.

Bypassing repair changes old-candidate warm Q from.00619/.00643 to.00014/.00017.
For mixed pair25, normal.00818/.00827 becomes.00791/.00786. The new benefit is
much less dependent on the trained adapter at runtime. Bypass changes operating
distribution; a matched trained no-adapter control is selected as an adaptive
follow-up. This does not establish that repair is unnecessary during training.

## Follow-up selection

Only proposal_mixed_pair25 meets the predeclared extension gates at both prefixes.
See extension_decision.json. Its exact2k->5k continuation and a matched fresh2k
plain_mixed_pair25 control completed in recovery_round10_followup. The latter
is an adaptive17th scout selected after these results. The continuation regressed:
prefix32 Q.008274->.006976, radial1.159->1.565, despite improved local prediction.
The matched plain control is worse (Q.005679, radial1.707). Keep the2k candidate;
no further extensions. See [completed follow-up](followup/assessment.md).
All17 scouts and one extension have zero full passes and zero late stopping.
Both queues are sealed/empty, with no failed jobs. Diagnostics have finished.

## Validation and artifacts

86 focused tests passed. Two4-update GPU smokes used full batches and1024-step
evaluation. Fresh proposal_clean_s25 exactly matches the previous2k G/D/particles.
All16 trainer source hash dictionaries match. Legacy configurations keep their
old behavior. New scouts use no MSE objectives, clipping, EMA, private G memory,
seed sweep, full training rollout, or changes to API B-cap defaults. Pair/point
branches have at most one sequential generated write. Reports consume completed
runs only; both GPUs share core_round1/train.log.

After the experiments, a reporting-only fix preserved legacy stability-branch
write counts when combined with a pair head, and included stability-only branches
in the maximum sequential-write metadata. Training computation did not change;
all18 experiments used identical archived sources. No queued config uses legacy
stability auxiliaries. See validation.json.
