# Latest: information diagnosis and future ranking round14 completed

Read ../memory-handoff/information_round14/{assessment,diagnosis,scout_information,next}.md.
Five2k scouts complete on both GPUs, zero failures, no qualifying extensions.
All full cold/warm passes remain0/128. All diagnostics complete, nothing running
or queued. Included in the requested commit with round13 and winner audit;
push not requested. No next experiment selected.

USER'S NEXT DIRECTION: investigate whether sequential training needs special
treatment, starting from the actual loop and first principles. Examine state
carry/reset/detach, temporal credit assignment, representation drift as D's writer
changes, D/G update timescales, and the joint (z,M) distribution during feedback.
Keep existing constraints; do not assume full generated rollouts are now allowed.
Discuss these questions before committing to the earlier persistent/fast-state
proposal. Detailed handoff: ../memory-handoff/information_round14/next.md.

Saved round12 match_shuffle25 remains winner (5k nominal,2k late-Q control).
Best new future_full10 minQ .009970 vs saved2k .010901; radial slightly better,
combined/late quality worse. Detaching only new generated-future writer gradients
worsens matched scout, but neither wins. No5k followups.

Frozen-M probes now support declining accessible process information as well as
early representation shift. Saved5k speed R² .954->.436->approximately0 after
0/32/128 generated writes; real128 remains .946. M+z does not rescue128.
New strongest future loss improves CLEAN radius/speed R² to .685/.965 but loses
it during feedback. All five new scouts near chance at128. Probe limits remain;
this is not proof of information-theoretic erasure or impossibility of local training.

Optional future_rank_* support shares the G-facing point head via explicit
horizon projection, scores real observations4/12 steps ahead after at most one
generated write, retains winning immediate mismatch and unchanged G objectives.
No MSE GAN training, geometry targets or additional generated trajectory depth.
77 tests/two smokes; old h0 training bitwise equivalent, exact resume preserved.
16.1min queue wall/26.8 GPU-min training. Stable tail unchanged.

Next discussion should target preservation through repeated writes. Prior fixed
slow/fast memories and private G-GRU already failed; any new mechanism needs a
new incentive/control. No next experiment selected or queued.

# Previous: generated-write mismatch round13 completed

Read ../memory-handoff/write_match_round13/assessment.md, diagnostics.md and next.md.
Five2k scouts complete on both GPUs; zero failures, no qualifying extensions.
Nothing running/queued. All full cold/warm256/1024 passes remain0/128.
No next experiment selected. Included in the subsequent round14 commit.

Existing match_shuffle25 remains the leading recipe (saved5k nominal,2k late-Q
control). Generated-write mismatch contexts at mild/full strength regress warmQ;
head-only control regresses further. Slight radial gains do not outweigh angular
quality loss. Current explicit mismatch winner still uses real-prefix memory.

New process-response windows show original speed/radius influence fading over
repeated writes. Saved5k median speed response .859/.192/.00066 in32-point windows
starting after0/32/128 generated writes. This does not distinguish information
loss in M from G readout failure. That is a useful next diagnostic question.

75 tests, two GPU smokes, bitwise legacy equality; five shared-source/shared-panel
scouts. ~11.8min queue wall,1183.8 training GPU-seconds. Stable tail unchanged.

# Previous: first-principles local objectives round12 completed

Read ../memory-handoff/principles_round12/assessment.md and followup/assessment.md,
then next.md. Twelve2k scouts plus two exact2k→5k extensions completed on both
GPUs, zero failures. Both queues sealed/empty. All diagnostics complete.
Nothing running. This round is included in the user-requested commit; push was
not requested. After compaction, discuss before selecting the next experiments.

New leading recipe: match_shuffle25. Existing proposal_mixed_pair25 plus D-only
GAN ranking of true next sample against a real next sample from another history,
using the SAME point head G learns from. Shuffled donor order, weight.25;
D loss/default exact B-cap normalized by1+weight. Only real-prefix M currently
receives this explicit mismatch objective. No new persistent state or runtime
expert. Fixed particle, existing clock/proposal repair/mixed point judging/pair25.
No MSE training, full generated rollouts, geometry labels/cursor, clipping, EMA,
B-cap override or seed sweep. Query/recovery alternatives remain config-controlled.

Warm1024 Q prefix8 / prefix32 / radial32:
- old proposal_mixed_pair25 2k: .008180 / .008274 /1.159
- match_shuffle25 2k: .010901 / .011161 /.945
- match_shuffle25_5k: .011008 / .011099 /.912
- match_nearest25 2k: .009982 / .009932 /.994
- match_nearest25_5k: .007605 / .008121 /1.579
All cold/warm full256/1024 passes still0/128, stopping0%. Nominal new reference
is shuffled5k by the fixed minimum-warm-Q ranking, but its Q gain over2k is only
~1%; retain2k because lateQ is better.5k improves radial/early errors. No further
extensions. The finding is the shuffled mismatch recipe, not a solved circle.

Nearest.1/.5 regress; both prefix-noise recovery scales and their combination
with nearest.25 regress. Future query GAN at offsets0/4/12 loses to baseline and
its matched architecture control; combinations do not qualify. Twelve configs,
predeclared gates, exact optimizer/prior/RNG resumes. Main pipeline wall26.6min,
followups6.1min,3728 training GPU-seconds combined. Sources identical across all
runs; evaluation reference/prefix panels bitwise equal to old baseline.

D nearest-history ranking at prefix32 improves67%→88–89% in shuffled models.
Late radius/speed response still near zero; better D recognition does not ensure
process retention. Restoring real-history M at clock288 fixes the next point
much more than sustained continuation. Full-real vs recent32 M similar. Added
local D-gradient alignment probe, evaluation only; late alignment to a timed
reference is confounded by phase drift. Do not claim erased information or a
single proven cause.70 focused tests, two GPU smokes, default-off four-update
legacy equivalence including active B-cap, no-MSE and exact-resume checks pass.

Potential next hypothesis (not selected/queued): apply the winning mismatch
signal to memories AFTER one generated write, retaining real-continuation identity
reference. Compare clean/explored/mixed mismatch contexts; distinguish this from
existing mixed point-GAN judging. Goal is writer/state maintenance, keeping the
same bounded generated-write budget. Discuss before selecting the next round.

Stable tail: tail -F runs/memory_path/core_round1/train.log
Preserve unrelated .claude/, results/motion/, sparse-ucd.log.

---

# Latest: G observation recurrence round11 completed

User selected small G GRU scouts building on proposal_mixed_pair25, explicitly
rejecting a moving cursor in favor of cross-domain transferable mechanisms.
All five fresh2k configs finished on both GPUs through memory_dispatch. Zero
failures; gru_round11 is sealed and empty. All diagnostics complete. Nothing
running. None qualified for an exact5k extension under predeclared gates.
No seed sweeps, MSE training, full generated training rollouts, clipping, EMA,
geometry labels or B-cap overrides. This round is included in the requested commit.
Latest direction: compact, then return to first principles and theorize about
why full-circle success remains0/128. Read
../memory-handoff/gru_round11/first-principles-next.md. No next experiment selected;
do not automatically queue state-recovery scouts. Push was not requested.

Read ../memory-handoff/gru_round11/assessment.md first, then plan.md, leaderboard.md,
results.json, extension_decision.json, process.json, state.json, execution.json,
validation.json and legacy_equivalence.json. Keep the round10 2k winner.
Central tail unchanged: tail -F runs/memory_path/core_round1/train.log

New optional config fields: g_state_dim (0 default), g_state_reads_d,
g_use_d_memory. D owns M; G owns observation-updated S. G reads z,M,S,clock via
existing proposal repair; both proposal/final reads share S and never advance
it. Real prefixes build both memories; S has full real-prefix BPTT in G phase,
no graph in D phase. At runtime S consumes generated points; optional updater
also reads pre-write M. M/S start at zero cold. Local pair has two G outputs
with one intervening M/S write; point feedback writes identical blended sample
to both. No third generated point. Fixed z per sequence. Dedicated module:
experiments/memory_g_recurrent.py; main handoff trainer dispatches optional state.
Cold/warm eval and memory interventions support S. Old local-only diagnostics
that call G without S reject missing state; use diagnose_memory_g_state.py for
new models. Existing legacy experiments remain unchanged with g_state_dim=0.

All five cold/warm full-circle passes0/128 at256/1024, prefixes8/32. Stopping0%.
Warm1024 prefix32 Q / radial / first32 position error:
- saved proposal_mixed_pair25: .008274 /1.159 /1.528
- gru8: .007193 /1.719 /1.316 (best new scout; Q down13-15% both prefixes)
- gru16_no_d: .006228 /1.881 /1.774
- gru16: .005239 /2.346 /1.611
- gru16_read_d: .004639 /2.568 /1.259
- gru16_no_repair: .001660 /3.081 /1.883
GRU8/D-informed GRU improve startup/first32 errors but worsen long fidelity.
No-D GRU16 outperforms with-D GRU16 on long Q/radial but loses short accuracy.
No matched no-D GRU8 scout. Keep repair; adding capacity alone did not help.

Matched radius/speed/direction probes again show near-zero median late process
response. D shuffling strongly disrupts M-reading models. G-state interventions
also alter outputs, but zeroing states sometimes improves Q; dependence does
not prove usefulness. No intervention gives warm full-orbit passes. The no-D
model is bitwise unchanged by D zero/shuffle through1024 steps at both prefixes.
Probe normal/altered paths all regenerated on CPU consistently; process tool
now supports --recompute-original. This changes the baseline's descriptive
both-direction number relative to prior mixed-device probes (8.59% here).
Behavioral results do not prove whether information is lost or unused internally.

90 focused tests pass; two fullbatch4-update GPU smokes with1024 eval; exact
resume and no-MSE/ownership/call-count tests. Archived round10/current legacy
four-update G/D/prior/RNG tensors bitwise equal on CPU including active B-cap.
All five archived training source hash dictionaries match. Queue wall734s;
1189 training GPU-seconds total. Scouts cost~1.32-1.39x saved baseline.

Recommendation: retain existing2k baseline; no more GRU size/length sweep based
on these results. Next discuss adversarial local recovery from G-state
perturbations and/or evaluation-only late restoration of real-history M/S to
separate drift from reader/clock failures. Neither implemented or queued.
Preserve unrelated .claude/, results/motion/, sparse-ucd.log.

---

# Latest: local adversarial recovery round10 completed

User authorized code/config changes and scouts on both GPUs to get autonomous
circles or measurable progress without full-rollout training. Completed17 fresh
2k scouts (16 planned plus one adaptive matched control) and one exact2k->5k
continuation. Zero failures; both recovery_round10 and recovery_round10_followup
queues are sealed and empty. All diagnostics finished. Nothing is running.
Latest user request: commit and push this round, then compact. The user is
considering G-owned memory and other ways to reach full circles. No next
experiment is selected. See ../memory-handoff/recovery_round10/next-brainstorm.md
for candidates and the important distinction from earlier rollout-trained GRUs.

Read ../memory-handoff/recovery_round10/assessment.md and followup/assessment.md,
plus plan.md, leaderboard.md, extension_decision.json, validation.json,
baseline_probes.md, process_*.json, followup/process.json and followup/probes.json.
Central log remains: tail -F runs/memory_path/core_round1/train.log

Best2k candidate: proposal_mixed_pair25. Same stateless proposal-conditioned G
repair as round9, but point GAN averages clean/shared judging50:50 and a separate
local pair GAN gets25% objective weight. Point exploration is still probability.5,
strength.25, min prefix4, ramp500, connected G gradients through frozen writer.
Pair branch independently starts from real-prefix M, generates first, writes it
with D, then generates second. D compares the candidate pair to a real pair using
memory strictly before both. Zero-prefix examples judge real[0:2]. There is no
third generated point and no chain from the point-feedback branch into the pair
branch. D owns/trains the only persistent memory; G adapter never writes back.
Combined recipe costs4 full G calls/8 internal reader calls per D/G phase.
Point/pair GAN losses and default exact B-cap penalties are convexly weighted;
prior regularizer once. No MSE training objectives, clipping, EMA, seed sweep,
private G memory or full generated training rollout. Existing configs still work.

New evaluation-only Q averages radial AND per-step signed-angular fidelity:
1/((1+(radial/.1)^2)*(1+(angular_error/.03)^2)). Q in[0,1], not a success
probability and not an absolute-phase metric. Perfect clean circles Q1; saved
noisy expert panel ~.515. Also good-step fraction, initial good streak and longest
consecutive correct arc in turns. Cold self-fit Q is separate from reference
fidelity. Metrics use saved arrays; no generated evaluation path enters training.

All18 runs: cold and warm complete passes0/128 at256/1024, prefixes8/32;
late stopping0. Circle task remains unsolved. Prefix32 results:
- prior proposal_clean_s25: Q.006426, radial1.487
- new proposal_mixed_pair25 at2k: Q.008274, radial1.159
- matched plain_mixed_pair25 at2k: Q.005679, radial1.707
- new proposal_mixed_pair25 at5k: Q.006976, radial1.565
New2k improves Q~29–32% and radial~22–24% at both prefixes, but first32 position
error is slightly worse. Only it met predeclared extension gates. Exact5k run
regresses on autonomous Q/radial, despite local one-write MSE improving.017637
->.012765 and early position improving. Keep2k checkpoint; no further extensions.
Shared/mixed judging alone fails; pair50 is worse than25; replacement variability
has context-dependent effects and hurts the best combined recipe. Matched plain
control supports keeping proposal repair in training, despite a small post-training
bypass effect on the new checkpoint (Q32 .008274->.007862). Old candidate bypass
Q32 collapses.006426->.000166. Capacity/compute differ in adapter comparisons.

Matched-prefix history probes hold z/clock/center/handoff phase/noise fixed, vary
radius.65/1.35 or signed speed magnitude.14/.36, or flip direction. Near-zero median
late radius/speed response persists in old2k, new2k and new5k. Correct mean direction
for BOTH original and flipped histories:5.5%,10.9%,6.25%. These diagnostics show
weak control of late behavior by process parameters; they do not distinguish
D memory losing information from G failing to use information still present.
Recurrence/history data are also saved; no visual ranking was used.

Recommended next discussion: distinguish writer information loss from reader
failure at late autonomous states. Longer training/local prediction gains do not
solve it. Keep new2k as baseline. Do not automatically launch another sweep.

Validation:86 focused tests passed, two full-batch4-update GPU smokes with1024
step evaluation, exact fresh round9 control reproduction (G/D/prior), exact resume
and no-MSE/gradient-ownership/call-budget tests. All18 experiments used identical
archived source hashes. After completion, a reporting-only fix corrected counts
for optional legacy stability+pair combinations;13 affected focused tests pass.
No training computation changed. Both queues completed with zero failures.

Branch feat/sequential-memory-path; this handoff is included in the user-requested
round10 commit/push. Base before this round:1ef40f6.
Preserve unrelated .claude/, results/motion/, sparse-ucd.log.

---

# Latest: adversarial memory exploration round9 completed

Latest user direction: commit and compact. The next experiment remains open;
the user sees proposal repair as promising. Resume with discussion of what its
benefit means before choosing another sweep. Mixed clean/explored judging below
is one candidate, not an agreed next step. The strongest evidence to build on is
proposal_clean_s25's autonomous adapter benefit (radial1.487 vs4.158 on bypass)
and greater persistence of initial-history effects. Neither demonstrates correct
process preservation yet. No automatic experiments on resume.

User rejected MSE training objectives: repair must learn through particle GAN.
MSE remains allowed as an evaluation metric. Earlier proposed output-consistency
MSE in dynamics_round8/next-experiments.md is explicitly SUPERSEDED.

Completed ten2k scouts plus one exact2k->5k continuation, both GPUs through the
existing pipeline. Queues exploration_round9 and exploration_round9_long are
sealed/empty,10/1done,0failed/pending/running. All diagnostics finished. No new
experiments selected. This handoff is included with the user-requested commit on
feat/sequential-memory-path. Preserve unrelated .claude/, results/motion/,
sparse-ucd.log.

Read ../memory-handoff/exploration_round9/assessment.md, plan.md, leaderboard.md,
mechanisms.md, extension_decision.json, and long/leaderboard.md. Full raw metrics
and saved-panel diagnostics are in the same directory. Stable central log:
tail -F runs/memory_path/core_round1/train.log

New formulation: G reads a real-prefix state after one generated replacement;
D can score both the next real/fake candidates using undisturbed real-history
memory. Both candidate scores and B-cap always share the same causal judging
state. No future/target leak. Shared controls score both using explored memory.
G optionally backpropagates through its earlier proposal and frozen D writer.
Only D trains writer parameters; only G/particles receive G gradients. Clean
judging trains D's writer only via real history; shared judging also trains it
through explored writes. This distinction matters for interpreting the results.

New stateless proposal adapter: proposal=H(z,M,clock), readable=M+R(M,proposal),
final=H(z,readable,clock). Only final is written to D memory. Clock enters R via
proposal, not a separate clock input. Two internal point-reader calls per G
evaluation, up to4 per training phase with the single-write branch. No full
rollouts, private G state, MSE/other auxiliary, clipping, EMA or seed sweep.
Default public API exact B-cap and prior regularization unchanged.

All10 scouts and the5k continuation: full cold circles and original-orbit warm
passes0/128 at256/1024, prefixes8/32; late stopping0. Still unsolved.
Selected prefix32 long radial errors (2k): clock2.597, shared_s25 1.726,
clean_s25 2.056, clean_s25_detach2.467, shared_full2.318, clean_full2.063,
proposal_control1.675, proposal_clean_s25 1.487, proposal_clean_full1.803,
residual_clean_s25 2.547. Shared mild beats clean mild; clean full improves some
errors vs shared full but not original-orbit success. Proposal mild has lowest
radial error at BOTH prefixes. Its adapter bypass worsens radial1.487->4.158,
still zero full passes. Memory-only residual has best local prediction but
bypass improves radial2.547->.700, also no full passes.

One diagnostic extension selected: shared_s25 meets >=25% radial improvement,
better early position/one-write prediction, no worse direction/stopping at BOTH
prefixes versus clock_control. Proposal_control also qualifies, but shared_s25
has lower one-write error at both prefixes and half the runtime reader calls.
Best proposal+exploration does not meet gates versus its proposal-only control.
Exact shared_s25 extension2k->5k REGRESSES: prefix32 radial1.726->2.272,
early position1.365->2.131, direction50.5%->48.3%, localMSE.00579->.00963.
No further extensions selected.

New recurrence evidence: old clock and video slow16_r10 have best late recurrence
near201 steps for ALL128 particles, all prefixes. Clock-rate0.9/1.1 interventions
move median period to223/183, matching clock period scaling. Current clock and
shared mild also largely converge to the same late outputs from cold/warm starts
when aligned at absolute clock time (97.7%/94.5% below normalized error.001).
Proposal clean mild differs: only51.6% recur best within199..203, lag201 median
normalized error.828 vs clock.000057; only7% have similar cold/warm late outputs.
This suggests persistent history effects but could be wrong attractors/chaotic
sensitivity, not correct process memory. Its clock interventions still shift
median recurrence221.5/200/182 with0.9/1/1.1 rates. Toy success remains zero.

Possible next targeted test (not implemented/queued): mix clean and explored
judging contexts for the same generated continuation, ordinary GAN losses in
both views. Combine an anchored reference with D writer exposure to explored
states, without additional generated writes or full rollouts. Retain a proposal
adapter arm to distinguish its autonomous effect from local prediction quality.

Implementation: feedback_judge_memory='shared'|'clean'; adversarial_only guard;
g_memory_adapter='proposal'. Defaults preserve legacy behavior/configs. New
diagnose_memory_recurrence.py measures saved-path recurrence, aligned history
retention and optional clock-rate interventions. Existing dynamics diagnostics
now obtain the actual proposal-conditioned read. Reports distinguish complete
G calls from internal reader calls. 80 focused tests passed; two full-batch GPU
smokes and CPU proposal diagnostic passed. All10 trainer source hashes match;
fresh clock control exactly reproduces historical G/D/particle tensors.

---

# Previous discussion before compact: organize directed memory repair

User watched the saved video and noticed that trajectories appear to retrace a
repeating irregular path. This is a visual observation, not a measured recurrence
or a new basis for ranking experiments. User wants to compact, then work toward
solving the task with more organized memory repair. No new experiments launched.

Hypothesis: the loop may settle into a stable clock-driven response rather than
accumulate noise indefinitely. Clock base frequency .03125 radians/step implies
an underlying continuous period 2*pi/.03125 ~=201.06 steps; all bands are integer
multiples. Integer-step sampling only nearly repeats around201 steps. Measure
recurrence and clock-phase dependence before claiming this explains the paths.
A repeating incorrect orbit does not by itself prove memory corruption.

Repair may need to distinguish progress, persistent process information, and
error. Current repair is memory-only synthetic denoising, with no explicit
before/after transition relationship. Candidate next direction: use previous
memory, G's proposed sample, and time to repair the next G read toward a coherent
continuation. Learn from paired clean and corrupted/generated LOCAL transitions;
retain D-only ownership/writes of stored M and a stateless G-side adapter. This
is a hypothesis, not an implemented architecture or selected loss/config. Exact
runtime timing, training targets, bounded G-call cost, and gradient ownership
still need defining. Previous single-write feedback already failed: isolate what
sample-conditioned repair or predictive supervision adds beyond that exposure.

Do not force every meaningful state difference to contract, or every process to
follow the clock's fixed period. Unseen speeds/radii remain a generalization
objective. First discriminate process preservation from clock-phase replay with
metrics, then select the next experiments. User remains positive about sustained
motion as a milestone. Main work/video committed as e77ca0a. Nothing running.

---

# Latest: memory dynamics and G repair scouts completed

User authorized slow/fast D memory, local feedback stability, and a G-side memory
repair/translator, with config-controlled changes and backward compatibility.
Then suggested that repair could use the generated sample to preserve useful
continuation. We acknowledged that as a next hypothesis, not an implemented
sample-conditioned adapter. The current adapter reads M only.

All work finished: 16 fresh 2k scouts and one exact 2k -> 5k continuation.
Queues dynamics_round8 and dynamics_round8_long have 16/1 done, zero failed,
pending, or running. Both GPUs were used; all diagnostic sessions are closed.
User requested commit and compact; this handoff is included in the commit on
feat/sequential-memory-path. Resume with discussion, not an automatic sweep.
Preserve unrelated .claude/, results/motion/, sparse-ucd.log.

Video: ../memory-handoff/dynamics_round8/slow16_r10_rollout.mp4 (24fps, 1024
saved steps, about43s). It shows cold starts and prefix32 continuations from
slow16_r10, lowest long radial error at BOTH prefixes among new scouts. Examples
are the first CW and first CCW reference episodes, selected without looking at
generated quality. Fixed axes, full faint history, bright last32 steps, and a
moving offline reference point. Script: experiments/render_memory_rollout.py.
Rendering is explanatory only; no experiment selection was based on images.

Read ../memory-handoff/dynamics_round8/assessment.md, leaderboard.md,
mechanism-table.md, long/leaderboard.md, and the local/adapter probe JSON files.

Every new checkpoint: 0/128 cold circles at256/1024 and original-orbit continuation
at prefixes8/32 and256/1024. All retain 0% late stopping; the clock milestone
survives. All late-only cold circles also0. No primary-metric winner.

Useful distinctions:
- translate16 improves prefix32 real-context next-point MSE .00664 -> .00541,
  but long radial error2.545; bypass worsens local prediction yet reduces radial
  error to.856, still0 original-orbit passes.
- repair_raw100 reduces synthetic memory reconstruction error~29% at noise.05;
  local prediction MSE.00474. Bypassing repair barely changes noisy G prediction.
- repair_raw10_n15 at2k has radial1.834 (baseline2.597); bypass worsens to1.981.
  It alone met predeclared continuous-metric extension criteria at BOTH prefixes,
  so received a diagnostic5k continuation. At5k radial worsens to2.456, local MSE
  .00561 -> .00787, all full passes still0. No further extensions selected.
- stable_dg10_cap09 reduces mean worst-direction local feedback gain2.415 ->1.894,
  but radial2.344 and full passes0. Random gain is already below1 on average in
  the baseline; this does not constrain its most sensitive directions.
- Baseline prediction after one generated write MSE.02717 vs.00388 after a clean
  real write atprefix32. This is evaluation-only evidence of short feedback error,
  not proof that memory alone is responsible.

Implementation:
- memory_recent.py: SlowFastWriter damps selected GRU coordinates; same parameters
  as baseline. LocalReader has optional identity-initialized residual adapter.
  No private persistent G state; adapter never writes back into D-owned M.
- memory_handoff_scout.py: slow_dim/rate; g_memory_adapter, adapter_width/bottleneck;
  repair_weight/noise/target; stability_g_weight/d_weight/noise/max_gain;
  dynamics_min_prefix. All default off. Repair trains G adapter only. Stability
  compares two parallel one-step D.write(G(M)) branches, same z/time, detached
  prefix/particle anchors, opposite module frozen. No full generated rollouts.
- Dedicated RNG streams preserve exact resume. Actual old clock checkpoint loads
  strictly; legacy checkpoint resume without new inactive fields/streams tested.
- diagnose_memory_dynamics.py probes completed checkpoints, preserves saved eval
  panel, checks baseline first predictions, tests read/clock/adapter use, noisy
  reads, generated-vs-real write prediction, and optional feedback Jacobians and
  adapter-bypassed autonomous rollouts. Largest singular gain is not a global
  instability certificate; meaningful directions may legitimately be sensitive.
- 74 focused tests passed plus combined full-batch GPU smoke and CPU/GPU probes.
  Metadata-only write counts were clarified after all experiments: archived
  fake_writes_in_training counted feedback replacements only, while stability
  branches are described by local_dynamics and G-call counts. Current code adds
  explicit D/G write counts. No training semantics or results changed.

Next: discuss sample-conditioned repair or preservation of predictive usefulness.
Previous single-write feedback already failed; do not merely rename it repair.
Isolate the new adapter objective/conditioning and bound its local G-call cost.
No new sweep is automatically selected. No extrapolation to unseen radii was
established. Keep fixed particles, D-only writer, default B-cap, no clipping/EMA,
no seed sweeps, metrics-based completed-only decisions, and no unsolicited agents.
Central tail: tail -F runs/memory_path/core_round1/train.log

---

# Compact handoff: clock milestone accepted; first principles next

User explicitly considers sustained motion from the clock a success, even though
full circle preservation remains unsolved. Preserve that distinction: the clock
addresses stopping in these scouts, and memory still supplies predictive
information. This is a milestone to build on, not a reason to discard the clock.

User requested committing the accumulated experiment work before compact. Next
session: hear the user's ideas and work through the formulation from first
principles again, aiming for full-circle behavior without generated trajectory
rollout training. Do not automatically launch another sweep or the previously
suggested writer diagnostic. No experiments or diagnostics are running.

Current runtime concept: fixed particle z, external step counter t, D-owned M;
generate x = G(z, M, Fourier(t)), then M = D.write(x, M), then increment t.
No runtime expert. Keep configuration-controlled changes, default exact B-cap,
no gradient clipping/EMA, metrics-based evaluation, and no seed sweeps.

# Latest completed round: external Fourier clock (round7)

User asked to test an external Fourier metronome for G on both GPUs and determine whether successful models still use memory. Eight 2k scouts completed via runs/memory_path/clock_round7; all jobs done, none failed/pending/running. Four completed models also received clock/memory interventions. Nothing running, no longer runs selected. This work is included in the requested compact-preparation commit.

Read [clock assessment](../memory-handoff/clock_round7/assessment.md), [leaderboard](../memory-handoff/clock_round7/leaderboard.md), and raw results/interventions in the same directory.

**All eight: 0/128 full cold circles at256/1024 and 0/128 original-orbit continuations after prefixes8/32 at both horizons.** All seven advancing clocks prevent late stopping (0%), compared with57.0% in the constant-clock architecture control, but do not preserve circles. Random-origin recent/residual model diverges severely. No winner to promote.

Main clock model: freeze clock after32 generated points ->95.3% late stopping, versus0% normally. On real-prefix32 contexts, matched next-point MSE0.00664; shuffled memory1.45982 (~220x); zero memory0.73404; reset clock0.03601 (~5.4x). Both inputs influence G, but beneficial joint use for autonomous circle fidelity remains unestablished. Clock continues driving activity even with zero/frozen memory. Sustained motion is an accepted milestone; full-circle success remains a separate unmet target.

Implementation: configurable clock_bands/frequency/rate/to_d/origin_max in memory_handoff_scout.py. LocalReader appends sin/cos time features to particle input internally; actual learned particle dimension unchanged. Clock is an external integer counter, no private recurrent state. Explicit target times plus per-episode optional random origins; cold t=0, warm t=prefix length. Shared-clock D sees identical time for both candidate scores. Generic rollout/continuation pass time only for clock-enabled G, preserving other trainers. Old non-clock checkpoints supported. No generated training rollouts or new auxiliary losses, unchanged B-cap, no clipping/EMA.

experiments/diagnose_memory_clock.py validates saved baseline reproduction and startup preservation, then runs frozen/half-speed clocks and zero/shuffled/frozen memory reads after32 generated points, plus paired real-context sensitivity probes. Older handoff/latent diagnostics now also forward explicit time. 68 relevant tests passed; GPU training/evaluation smoke and CPU diagnostic smokes passed.

Earlier recommendation: retain optional clock for controlled comparisons and investigate local write-induced loss of predictive information versus G prediction error. The user's latest direction is first-principles discussion with their ideas before choosing another experiment. Nothing is queued. Preserve first-principles discussion below. Central tail remains `tail -F runs/memory_path/core_round1/train.log`.

---

# Completed: 21 valid bounded-feedback experiments

## Latest user direction: first principles after compact

User is compacting now and says we need to approach this from first principles.
This supersedes automatically launching the local prediction-loss diagnostic
recommended below. Do not start another configuration sweep or add auxiliary
losses before reconsidering what the objective actually identifies.

Start the next discussion with the smallest precise formulation:

```python
# Teacher-context training, with real history strictly before x_t:
M_t = D.write_history(x_real[:t])
x_fake = G(z, M_t)
L_D = paired_GAN_D(D.score(x_real[t], M_t), D.score(x_fake, M_t)) + B_cap
L_G = paired_GAN_G(D.score(x_fake, M_t), D.score(x_real[t], M_t)) + prior_reg
# D updates writer + scorer; G updates reader + particles, with D frozen.

# Autonomous runtime, fixed z throughout:
M = zeros()
repeat:
    x = G(z, M)
    M = D.write(x, M)
```

Questions to derive rather than assume:

- What does matching the conditional next-point distribution under real-history
  memory imply about the autonomous transition M -> D.write(G(z,M),M)? What does
  it leave unconstrained, even at an idealized optimum?
- Training samples z independently of each real episode; runtime memory depends
  on that same fixed z. Are the training and runtime JOINT distributions of (z,M)
  compatible? Distinguish this hypothesis from ordinary state distribution shift.
  A weak particle-swap diagnostic did not establish it as the actual cause.
- Which uncertainty remains after zero, one, or several observed points? How
  should a persistent trajectory particle encode choices during cold startup?
- Is there a coherent real target after a generated replacement? The original
  next real point is a recovery target, not automatically a valid continuation of
  the altered history. Do not assume exposure alone fixes that mismatch.
- What learning signal can constrain local autonomous dynamics without full
  generated training rollouts? Distinguish distribution matching, pointwise
  accuracy, stability, and preservation of the original orbit.

Keep evidence separate from hypotheses: D/G demonstrably use motion information
in M on real contexts. The failures do not prove memory is ignored, OOD is the
sole cause, or pointwise GAN training is impossible. Noiseless controls also
failed at the tested budget. Late oscillations were not reliable cold starts and
collapsed to one direction. Identify a discriminating test only after clarifying
the mathematical objective and the missing constraint.

Preserve the concept: G reads D-owned M, D owns writer parameters, fixed particle
per trajectory, no realtime expert required, and no full training rollouts.
No jobs are running. Work remains uncommitted; no commit requested this turn.

## Completed experimental evidence

Latest session DONE:20 fresh2k scouts and one exact5k extension. No jobs remain.
All full cold256/1024 and1024-step original-orbit pass rates are0. One short warm
exception: recent_bound_control has1/128 atprefix32/256, then0/128 at1024. All other
warm256 rates are0. The toy remains unsolved. Read the comprehensive
[session assessment and recommendation](../memory-handoff/feedback-session.md)
and [verified summary metrics](../memory-handoff/feedback-session.json).

Valid queues: feedback_round3 (8), recent_round4 (6), backprop_round5_corrected (5),
clean_round6 (2). All have zero pending/running/failed; sessions3006,72596,61466,
91344 closed. No further extensions selected. recent4_delta's2k late-circle lead
(60.2%, allCCW) vanished at5k with84.4% stopping. Noiseless controls also failed.

IMPORTANT: backprop_round5 WITHOUT_corrected is INVALIDATED. Missing real-score
context-gradient branch affected G's paired loss:2 completed jobs excluded,
2 active jobs intentionally stopped,1 pending cancelled. Both branches now retain
G gradients when M depends on G; common-offset cancellation has a regression test.
Earlier detached-feedback/recent-memory jobs are unaffected. Report/checkpoint
guards reject invalidated sources/legacy backprop checkpoints.60 focused tests
plus1 reporter-exclusion test passed. Corrected full-batch GPU smoke passed.

Implementation: memory_handoff_scout.py supports configured single-write feedback
(probability,strength,min-prefix,ramp,optional G backprop) and recent-point slots
inside one D-owned M. memory_recent.py implements GRU+raw shift register and
bounded absolute/residual readers. G is feedforward with fixed particles and no
private state. No full generated training rollout, no path loss, default B-cap,
no clipping/EMA. Runtime needs no expert. Per-run source snapshots saved.

Next proposal: configured local G prediction-loss diagnostic on informative real
prefixes, retaining GAN branch, to separate adversarial optimization from
representation/feedback limits. Label its changed objective explicitly. NOT
implemented or queued. Do not keep extending failed models or replace full-path
success with burn-in-only metrics. No seed sweeps or unsolicited agents.

Central tail: `tail -F runs/memory_path/core_round1/train.log`.
Feature branch feat/sequential-memory-path; work uncommitted. No commit requested.
Preserve unrelated .claude/, results/motion/, sparse-ucd.log.

# Completed: pointwise local round 2

## Latest discussion / next round after compact

User is compacting now and wants another round afterward. Do not launch more
jobs during compact preparation. Branch: `feat/sequential-memory-path`.
All current jobs finished; work remains uncommitted. No commit requested here.

User agrees the important mismatch is real-history writes during training versus
generated writes at runtime. D sees fake candidates during training, but those
candidates never affect subsequent memory. Small error is not automatically OOD
(real observations already contain noise); accumulated, correlated errors may
move position and memory away from training support. This is a plausible failure
mechanism, not yet directly established by the completed diagnostics.

Latest proposed next experiment, not implemented or queued:

```python
M = D.encode(real_prefix)  # strictly before x_t
x_fake = G(z, M)
M_next = D.write(M, stop_gradient(x_fake))
# Local GAN prediction of x_(t+1), conditioned on this same M_next for real/fake.
```

This introduces one generated write and two sequential G evaluations per example;
describe that cost honestly, rather than calling it the existing one-point-only
trainer. No long generated training trajectory or trajectory critic is intended.
Use the same particle for both predictions. D alone owns/trains writer parameters;
detach the generated replacement, not all of M_next (D still needs writer grads).
Real x_(t+1) is a recovery/denoising target after replacing x_t; it need not be the
natural continuation of an arbitrarily wrong generated point. Make that assumption
explicit and test a bounded/mixed replacement strength if needed. Rebuild context
after updating D as in the current trainer, keeping causal timing and cached
real/fake contexts aligned. This is the leading discussion direction, not an
already agreed implementation specification. Preserve no-clipping/API B-cap,
fixed particles, expert-free runtime, configs, completed-only metrics, both GPUs,
and completion-driven queue notifications. No seed sweeps or unsolicited agents.

Recent-observation slots inside D memory plus residual G output remain a separate
architecture idea. They do not themselves remove the training/runtime mismatch.
The user deferred further experiments until after compact.

## Completed evidence

Latest user constraint: solve without generated training trajectories. User
correctly challenged the claim that D lacks incentive to use M. Baseline probes
confirm temporal use: matched G MSE .00560 versus shuffled1.405; opposite-direction
histories ending at the same point give D correct-continuation preference~89%.

Eight 2k scouts finished successfully in340.7s on both GPUs. Queue
`runs/memory_path/local_round2` is sealed and empty; drain35723 is closed.
Four ordinary-GAN scouts (G SiLU/tanh, G128, D256) all failed full cold/warm metrics.
Four D-only auxiliary scouts also failed warm fidelity; predictive10 alone had
2/128 cold passes, both CCW at256. No convincing winner; no longer runs queued.
50 tests passed; changes remain uncommitted. No fake training writes or G unrolls.

Read [assessment and next recommendation](../memory-handoff/local_round2/assessment.md),
[leaderboard](../memory-handoff/local_round2/leaderboard.md), and
[plan/formulation](../memory-handoff/local_round2/plan.md).
Next proposal: recent-observation slots inside D memory plus residual G output,
keeping ordinary pointwise GAN objective. Not implemented. Feedback instability
is the working hypothesis; current probes do not establish generated-state OOD.
The toy remains unsolved. Same tail: `tail -F runs/memory_path/core_round1/train.log`.

# Completed: handoff-only scouts and 5k checks

Latest user direction: explore without cold loss because generated trajectories
are expensive. This supersedes extending light/heavy cold-feedback winners now.

Ten 2k-update scouts have finished successfully in
`runs/memory_path/handoff_round1`, using `experiments/memory_handoff_scout.py`.
No cold or warm loss, no trajectory critic, and no generated training unroll.
G makes one batched independent point prediction per D/G update. D still encodes
real history; memories are cached for real/fake/B-cap scoring and rebuilt once
for G after updating D. D alone trains the writer. Four points per episode share z.

Read [the current plan](../memory-handoff/README.md),
[leaderboard](../memory-handoff/round1/leaderboard.md), and
[metrics](../memory-handoff/round1/results.json).

Scouts vary real prefix length (63/8/16), observation corruption (.03/.1), memory
snapshot jitter (.05), memory size (32/8/64), D point/context interaction, and
zero-prefix sampling (.125/.25). Same 2k updates / 10k schedule, API B-cap,
no clipping, no EMA, no seed sweeps. Context/augmentation RNG streams are separate.
The old completed handoff_only baseline is included without retraining.

All 48 focused tests passed, including causal context, active exact B-cap,
cached/recomputed gradient equivalence, ownership, resume, and no generated
training unroll. Batch-128 CUDA smoke passed for memory64 + interaction + noise.

The previous tail command still works because the new queue's train.log points
at the existing central log:

```bash
tail -F runs/memory_path/core_round1/train.log
```

Initial drain session 45793 finished. Completion-only notifications:
`runs/memory_path/handoff_round1/notifications.log`. Reports refresh after jobs
complete, before notifications. Initial queue is sealed and complete. See the follow-up below.
Do not restart this queue blindly. Check completed results only, then compare
original-orbit fidelity, cold stability, both directions, diversity, and training
cost before deciding on longer runs. Long rollouts are final evaluation only.

## Current follow-up

The initial round failed cold256 and real-prefix fidelity in every configuration;
only start25 had a single cold1024 pass (1/128). Input noise .10 reduced late
stopping to 4.7% (default dense 68.8%). Initial queue took 6.59 minutes.
[Full assessment](../memory-handoff/round1/assessment.md).

Both input10 and start25 completed exact continuations to 5k (3k additional updates) in
`runs/memory_path/handoff_round1_long`. These are inexpensive diagnostic extensions,
not convincing winners. No new losses or generated training unrolls.
Drain session 56026 finished without failures; both queues are empty. Same central train.log as before.
[Combined completed-only leaderboard](../memory-handoff/long/leaderboard.md).
Both 5k runs have zero cold256/1024 passes and zero original-orbit passes.
Noise input10 late stopping worsened to 75%; start25 late stopping is 30.5%.
[Final assessment and recommendations](../memory-handoff/long/assessment.md).
Do not promote these failed runs merely for more steps. Next suggested local
scout: temporal mismatch negatives for D, still without any trajectory loss.
No further jobs or implementation changes have been started.

## Previous completed core round

Branch: `feat/sequential-memory-path`. Core formulation trainer, configs, metrics,
and tests are implemented. The shared two-GPU queue in
`runs/memory_path/core_round1` finished all nine jobs successfully in 53.2 minutes.
No jobs are pending or running. Read [the assessment](../memory-core/round1/assessment.md)
before choosing extensions. No extensions have been launched.

## Current work

Seven 2,000-update scouts: handoff_only, handoff_cold, handoff_warm, handoff_both,
warm_only, handoff_cold_light, handoff_cold_heavy. All share GRU32 D memory and
feedforward G, learned writer, fixed particles, default exact B-cap, no clipping,
no EMA, same data/initialization/schedule. No seed experiments.

Training and evaluation rules, comparison caveats, and table of loss weights:
[core round README](../memory-core/README.md).

Two existing autonomous controls (2k and 10k) have completed new evaluation without
retraining. The old 10k model passes cold full256/1024 at 53.1%/51.6%, with 31 CW
and 37 CCW full256 passes, but has zero reference-orbit passes after either prefix.
The 2k control gets 14.1%/15.6% cold and zero reference-orbit passes. Neither proves
that an arbitrary expert trajectory is preserved by its generated feedback loop.
All seven new scouts are now complete. Heavy/light handoff plus cold feedback
reach 33.6%/26.6% cold 1024 success; light preserves balanced passing directions.
Every formulation has zero reference-orbit passes. See the completed assessment.

## Monitoring

```bash
tail -F runs/memory_path/core_round1/train.log
```

- One permanent log for all jobs; labels identify run/GPU.
- Drain emits only completion/failure events. These are saved in `notifications.log`.
- Optional completed-job reporter refreshes the numeric leaderboard before sending
  the completion event. No polling of training logs is needed.
- [Leaderboard](../memory-core/round1/leaderboard.md)
- [Detailed metrics](../memory-core/round1/results.json)
- Queue manifests: pending/, running/, done/, failed/.
- Queue is sealed and finished; its final event is queue_complete with no failures.
- Drain ran in tool exec session 81434. Completion is recorded in notifications.log.

All 41 focused tests passed. Tests cover causal memory timing, read-only candidate
scoring, parameter ownership, full feedback gradients, exact B-cap, exact resume,
old/new cold-only equivalence, and completion-triggered queue reporting. A full
batch-128 CUDA smoke passed with all three losses and full 1024-point evaluation.

## Next decisions after completion

Compare H (real-memory handoff), C (cold feedback), W (mixed real-prefix lengths
0/8/32 then feedback), and their weights. All expose runtime:

```python
M = D.write(real_prefix)  # optional; otherwise zeros
for each generated point:
    x = G(fixed_z, M)
    M = D.write(x, M)
```

H trains next-point prediction from real points strictly before the target. Fake
candidate scoring is read-only. C/W explicitly train generated writes. D alone
updates writer weights; G gradients through generated state remain intact.

Cold and warm results are distinct axes. Warm checks original-circle geometry,
speed, direction and startup discontinuity, not merely any valid resulting circle.
Review continuous errors, particle coverage, passing directions, stopping, state
statistics and interventions before selecting winners. Reported auto-ranked
candidates are provisional; the script does not launch longer runs. If no single
formulation wins both axes, retain a candidate on each axis for further testing.

Then resume promising checkpoints for longer training, with exact optimizer/RNG
state. Add private G recurrence only after interpreting the matched feedforward
comparison. DDGAN, FiLM/Fourier and memory-size sweeps remain deferred.

Previous complete history is committed at dfaa1f1. The toy is still unsolved.
Preserve unrelated `.claude/`, `results/motion/`, `sparse-ucd.log`.
