# Round15 completed: raw successor matching

Six2k scouts completed on bothGPUs with no failures. None meets the unchanged
extension gates; all full cold/warm256/1024 passes remain0/128. Saved round12
match_shuffle25 remains the winner. See comparison.md, results.json,
information.json, transition.json and process.json. Queue sealed and empty.

Best new minimum warmQ is uncond_w10 .009286, versus saved2k .010901 (-14.8%).
Its radial32 improves .945->.759, but lateQ32 worsens .010869->.008282. Conditioned
G-only .10 gives minQ .008615, radial32 .770; .25 is weaker at minQ .008096.
Conditioned writer alignment hurts at both weights: joint_w10 .003995,
joint_w25 .004839. State-only writer alignment also loses (.004429). No late
stopping; sustained motion remains intact but correct process continuation fails.

Information probes clarify the failure. joint_w10 improves clean radius/speed
R2 .612/.942 -> .811/.975, but next-read MSE after a generated write worsens
.01496 -> .04916 and Q drops. joint_w25 also has good clean decoding yet poor
read behavior. All models approach chance at128 generated writes, including
M+z controls in the full diagnostic. Better clean information is not better
use or retention. G-only .25 retains somewhat more at32 (.185/.395 vs baseline
.127/.267), yet its autonomous Q is worse. No probe-based promotion is warranted.

K is not uniformly decisive: prefix32 real>fake ranks49-64%. Correct-anchor
paired margins beat shuffled-anchor margins only53-58% for conditioned scouts.
Absolute anchor scores are confounded by arbitrary anchor-only offsets under
RpGAN; diagnostic now reports offset-invariant paired-margin differences.
These metrics do not prove K ignores history or identify its causal features.

The new objectives used independent clean-prefix anchors, one full generated
write, fixed particle/clock, detached observed successor targets. K training
uses public exact B-cap over joint candidate coordinates. G gradients go
through frozen W/K; writer alignment goes only through the generated write,
with point/anchor/real successor detached. No MSE or long generated training.

Validation:91 distinct focused tests, two4step batch128/eval1024 GPU smokes,
pre-change and critic-only bitwise baseline checks, exactresume, gradient and
causality tests. K initialization originally perturbed CUDA RNG; fixed before
all smokes/scouts and regression-tested. Within-round training sources match.
Cost904.52s queuewall (~15.1min),1744.54 trainingGPU-seconds (~29.1GPU-min).

## Decision for adaptive round16

Do not increase raw writer alignment again. Test whether K needs explicit
history compatibility, and whether matching in G's read coordinates makes
successors more useful than raw-state matching. Add state_g10 to disentangle
state-only scoring from harmful writer alignment. Compare joint/state K with
other-episode successor negatives. Compare read-space G-only with writer
weights.01/.10. Six configs declared in round16/plan.md before launch.

This follows GibbsNet's joint-distribution intuition, but does not inherit its
stationarity theorem: fixed particle, clock, conditional process identity,
finite discriminators and local coverage remain distinct challenges.
