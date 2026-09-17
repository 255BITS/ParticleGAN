# Round18: separate G state and D clock

Seven2000-update scouts completed on both GPUs, zero failures. All completed-model
D/G memory probes and process diagnostics finished. Queue wall time21.38 minutes;
training37.71 GPU-minutes. All full cold/warm passes remain0/128 at256/1024 steps.
No candidate qualifies under the unchanged extension gates; no longer runs launched.
The original round12 winner remains ahead.

## Matched results

Q is minimum warm1024 continuous quality over prefixes8/32, not a probability.

|G formulation|D clock off|D clock on|
|---|---:|---:|
|Original shared D memory|0.010901|0.005849|
|Separate Mg, encoded emitted point|0.005087|0.007584|
|Separate Mg, internal decoder features|0.000988|0.001050|
|Separate Mg, equal mixture|0.000973|0.001501|

The best new model is embedded8_dclock,30.4% below the saved2k baseline. Prefix32
radial error worsens from.945 to1.429; late Q drops from.010869 to.007363.
D clock helps the separated observation control49.1% and hybrid54.2%, but the
latter starts from a much lower quality. It helps intent only6.2% and worsens the
original shared-memory baseline46.3%. There is no architecture-independent clock
benefit. All comparisons are descriptive at the tested budget, without seed repeats.

The three separated G models share the exact architecture, initial weights,
8-coordinate state,64-coordinate features, particle and clock. Only the generated
GRU input differs. Intent/hybrid are80–86% worse than the matched observation
controls. Pure internal carry therefore fails this implementation test; it is
not an improvement hidden by a strict full-circle threshold.

## State diagnosis

D access is completely absent from the six separated G models. D zero/shuffle
interventions are bitwise identical in the tested runtime paths and exactly equal
in the full reported256-step fidelity metrics. A test also reproduces runtime
without any D calls, using only G state, particle, clock and a dummy zero input.
G state affects outputs: shuffling it worsens first32 position error. Zeroing it
at every read reduces first32 error substantially for intent/hybrid, although it
still does not yield full circles. State dependence is not evidence of benefit.

All Mg models contain decodable clean process information: nonlinear held-out
radius R2 is.414–.441 and signed-speed R2 .862–.879. Continuing real observations
for128 writes retains roughly the same performance. After128 generated transitions,
both Mg and Md probes are near chance; adding the fixed particle to Mg probes does
not rescue it. Counterfactual radius/speed responses in late generated outputs have
median zero for all separated scouts. None preserves both original/flipped late
mean directions in that panel. These are finite diagnostic probes, not proofs of
information-theoretic erasure.

An important earlier failure appears specifically in internal-feature updates:

|Model|Real-trained Mg probe after1 generated transition, radius/speed R2|Probe refitted on generated states after1|
|---|---:|---:|
|embedded8|.394/.855|.402/.857|
|embedded8_dclock|.377/.850|.391/.853|
|intent8|-1.602/.115|.405/.840|
|intent8_dclock|-9.113/-.714|.364/.815|
|hybrid8|-.455/.510|.392/.850|
|hybrid8_dclock|-1.950/-.297|.373/.834|

Thus the first internal transition does not immediately remove all tested process
information. It changes the state representation enough that a readout calibrated
on real-history states transfers poorly, while a newly fitted readout can still
recover information. That is evidence of a real-to-generated representation shift.
It does not prove the GAN decoder fails in exactly the same way as the probe, or
that this shift is the sole cause of long-horizon failure.

The new GRU is shared between observation and generated updates, but its input
comes from a learned observation encoder during real-prefix encoding and decoder
features during intent generation. The existing GAN objectives did not establish
a sufficiently compatible interface in this experiment. Partial feedback trains
feature mixtures; the local pair does include the full runtime transition. We did
not simply omit generated-state training, but its temporal credit remains bounded.

D's clean process probes are stronger in several separated scouts than in the
original model, while G's own small memory is weaker. D/G state sizes and learned
representations differ, so this is not a causal claim that separating memory
improves D or that extra G dimensions would solve the task.

## Decision and next hypothesis

Keep the existing winner and stop this sweep. Do not promote intent/hybrid or
expand their width/clock settings on these results. Separation removed dependence
on D's evolving internal representation but did not solve autonomous state retention.
This weakens the emitted-point bottleneck hypothesis as a sufficient explanation.

If continuing the internal-state idea, the discriminating next question is whether
real-prefix encoding and autonomous generation can use the same internal transition
with observations acting as corrections, instead of switching GRU input sources.
First measure next-read prediction and state response around a single handoff on
these saved models. Any new design should demonstrate better one-step compatibility
and long-horizon retention; another small Q improvement alone is insufficient.
This is a proposed direction, not an implemented or queued experiment. It must still
honor no MSE objective and no full generated training rollout.

D clock remains an optional condition worth retaining in configs, with its effect
measured per formulation. UCD was considered, not added: the repository's UCD API
selects discrete class/time logits and adds D-only classification. The step index
is discrete despite Fourier encoding, but random initial phase makes absolute time
unidentifiable from a point. Memory may reveal prefix length, which is not the same
as learning process dynamics. No clock-classification target or continuous-UCD
analogue was introduced.

## Implementation and validation

Default g_state_update='observation' preserves old configs/checkpoints. New modes
embedded/intent/hybrid require separate G memory with no D access. Reads expose
final decoder features without advancing state. Generated state transitions occur
once per final sample, never during proposal refinement. Real-prefix BPTT remains;
local branches contain at most two generated outputs and one temporal transition.
No new auxiliary loss, MSE training, clipping, EMA, geometry labels, moving cursor,
seed repeats or public API B-cap override.

130 focused tests passed;18 relevant tests passed after the feature-mixing refinement,
and9 after stronger D-free runtime and diagnostic future-leakage assertions.
Four full-panel GPU smokes across both cards completed, including two on final
training sources, plus a saved-checkpoint Mg diagnostic smoke. Exact resume,
gradient ownership, final-feature credit, matched initialization and active exact
B-cap are covered. Common evaluation observations/references are bitwise identical
to the baseline. All seven scouts share identical archived source hashes. Diff
whitespace checks pass. Nothing is running or queued; changes are uncommitted.

See comparison.md, state_comparison.md, extension_decision.json, panel_audit.json,
execution.json, validation.json, plan.md and formulation.md. Stable central log:
`tail -F runs/memory_path/core_round1/train.log`.
