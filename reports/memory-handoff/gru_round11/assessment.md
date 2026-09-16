# G recurrence round11: retain the existing winner

Five fresh2k scouts completed on both RTX A6000 GPUs, with zero failures. No candidate meets the predeclared extension gates; no longer runs selected. All cold and warm full-circle passes remain0/128 at256/1024 points and prefixes8/32. Late stopping remains0%. The toy is unsolved.

## Completed leaderboard

Sorted by worst-prefix warm1024 quality Q. Q is not a success probability; absolute values remain small. Radial and position errors below are relative to the reference radius, prefix32.

| Model | Q prefix8 | Q prefix32 | Radial error ↓ | First32 position error ↓ | Training seconds |
|---|---:|---:|---:|---:|---:|
| proposal_mixed_pair25 | 0.008180 | 0.008274 | 1.159 | 1.528 | 175.5 |
| gru8 | 0.006981 | 0.007193 | 1.719 | 1.316 | 241.1 |
| gru16_no_d | 0.006190 | 0.006228 | 1.881 | 1.774 | 238.4 |
| gru16 | 0.005245 | 0.005239 | 2.346 | 1.611 | 234.6 |
| gru16_read_d | 0.004406 | 0.004639 | 2.568 | 1.259 | 243.6 |
| gru16_no_repair | 0.001483 | 0.001660 | 3.081 | 1.883 | 231.6 |

The baseline is the saved round10 proposal_mixed_pair25 at2k. GRU8 is the best new scout, but Q regresses13–15% and radial error worsens48–52% across the two prefixes. GRU16 is worse; feeding D memory into its updater does not rescue it. Removing proposal repair degrades GRU16 further.

The matched GRU16 no-D control has about19% higher prefix32 Q and20% lower radial error than GRU16 with D access, although its startup and first32 position errors are worse. This round does not establish a long-horizon benefit from reading D memory with G recurrence. The no-D model remains below the original baseline and is a control, not a successful memory-GAN solution. There is no matched no-D GRU8 run.

GRU8 and the D-informed GRU do improve warm startup and first32 position error. Prefix32 startup error falls from.0949 to.0847/.0772, and first32 position error from1.528 to1.316/1.259. Yet long-horizon quality and radial fidelity worsen. Their mean longest correct arcs increase slightly (.0588 turns baseline to.0609/.0635), without producing full circles. Short-horizon prediction gains have again failed to stabilize autonomous continuation.

## What the memories do

All diagnostics below regenerate normal and altered paths on CPU from the same saved prefixes and particles. This avoids comparing GPU normal paths with CPU interventions in a sensitive recurrent system. The original GPU metrics remain the ranking panel.

| Model | Radius response median (ideal1) | Speed response median (ideal1) | Both original/flipped late directions correct |
|---|---:|---:|---:|
| proposal_mixed_pair25 | 1.95e-08 | 1.76e-08 | 8.59% |
| gru8 | 1.56e-07 | -6.21e-10 | 7.03% |
| gru16_no_d | 0 | -8.13e-10 | 0.00% |
| gru16 | -1.29e-08 | 2.97e-09 | 0.78% |
| gru16_read_d | -2.75e-08 | 3.53e-10 | 0.78% |
| gru16_no_repair | 2.95e-09 | -9.36e-11 | 0.00% |

Median late responses to the intended radius and speed are near zero throughout. These behavioral probes show weak preservation of the requested process in outputs; they do not prove whether either state has erased the information or whether G fails to use retained information. Direction correctness here is only the late mean angular sign, weaker than the full-circle criterion. Small differences on128 fixed particles are descriptive, not significance claims.

Independent persistent read interventions, warm prefix32,1024 steps:

| Model | Normal Q | D zero | D shuffle | G state zero | G state shuffle |
|---|---:|---:|---:|---:|---:|
| proposal_mixed_pair25 | 0.008129 | 0.009060 | 0.000077 | — | — |
| gru8 | 0.007214 | 0.008936 | 0.000168 | 0.006986 | 0.002958 |
| gru16_no_d | 0.006228 | 0.006228 | 0.006228 | 0.011196 | 0.005545 |
| gru16 | 0.005253 | 0.007832 | 0.000818 | 0.007728 | 0.005288 |
| gru16_read_d | 0.004659 | 0.009473 | 0.000127 | 0.001867 | 0.002123 |
| gru16_no_repair | 0.001660 | 0.007495 | 0.000144 | 0.009008 | 0.001573 |

Shuffling D memory degrades quality sharply in models that read it. G-state interventions also affect outputs, especially with a D-informed updater, but zeroing a state sometimes improves Q. None of these interventions produces a warm full-orbit pass. Sensitivity to a state is not evidence that it helps preserve the intended process. The trained no-D model is bitwise unchanged by D zero/shuffle at both prefix lengths for all1024 outputs, confirming the access control.

## Implementation and validation

Configuration fields: g_state_dim (0 keeps the legacy reader), g_state_reads_d, g_use_d_memory. D retains its own writer M. G owns an observation-updated GRU state S and reads both M/S with the existing clock and proposal adapter. Proposal and final reads never advance S. Real prefixes construct S through real observations with full real-prefix BPTT in the G phase. The local pair generates two points separated by one write to each memory; point feedback writes the same blended observation to both. Both states start at zero for cold evaluation. Fixed particle per sequence.

No MSE/auxiliary training objective, generated full-rollout training, moving cursor, geometry labels, clipping, EMA, seed sweep or B-cap override. The no-D control zeros M before every G reader/adapter call, and its state updater consumes only observations. The proposal adapter can still transform G’s proposal when M is zero; this is an internal G computation with no D-memory information.

90 focused tests passed, including state causality, gradient ownership, active exact B-cap, exact resume, no-MSE training, once-per-observation updates, and cold second-point gradients through G state alone. Two full-batch four-update GPU smokes include1024-point evaluation. The archived round10 trainer and current legacy path produce bitwise identical G/D/prior/RNG tensors after four CPU updates, including active B-cap. All five scouts share identical archived training source hashes. Old experiment configs remain supported.

Queue wall time734.1s (~12.2min); total training1189.3s (~19.8GPU-min). New scouts take231.6–243.6s each versus175.5s for the saved baseline (~1.32–1.39x). Different architectures change capacity and compute: the D-informed updater adds parameters; removing repair reduces reader calls; same-size no-D retains the architecture and changes information access.

## Recommendation

Keep proposal_mixed_pair25 at2k as the winner. Do not extend these GRUs just for more updates or expand the size sweep. The next hypothesis should address recovery under generated observations. A bounded experiment could perturb G state and train the existing one-write adversarial recovery task, testing whether its updater corrects errors while retaining process information. Another useful evaluation-only diagnostic would restore real-history M/S separately at a late time to help locate state drift versus reader/clock failures. Neither is implemented or queued. No full generated trajectory objective or MSE is proposed.

Queue sealed and empty; all diagnostics complete; no jobs running. This round is included in the user-requested commit. Stable tail: `tail -F runs/memory_path/core_round1/train.log`.

Artifacts: [plan](plan.md), [leaderboard](leaderboard.md), [raw results](results.json), [extension gates](extension_decision.json), [process probes](process.json), [state interventions](state.json), [validation](validation.json), [execution](execution.json).

Latest direction: compact, then [return to first principles](first-principles-next.md) before selecting more experiments. No next design is selected.
