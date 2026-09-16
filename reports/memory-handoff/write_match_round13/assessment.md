# Round13: generated-write mismatch training did not beat clean mismatch

Five 2k scouts completed on both GPUs, zero failures. All cold/warm full-circle
passes remain0/128 at256/1024 and prefixes8/32; late stopping remains0%.
No scout meets the predeclared extension gates, so no5k followups were launched.
Queue is sealed/empty. All diagnostics are complete; nothing is running.

## Leaderboard

Minimum Q is the worse of warm prefixes8/32 at1024 steps. Q is not a success
probability; full passes remain primary. References are saved round12 checkpoints.

| Model | Minimum Q | Q32 | Late Q32 | Radial32 (lower better) | Change in minimum Q vs2k reference |
|---|---:|---:|---:|---:|---:|
| match_shuffle25_5k (saved) | .011008 | .011099 | .009708 | .912 | +1.0% |
| match_shuffle25 (saved2k) | .010901 | .011161 | .010869 | .945 | reference |
| explored_mild | .009046 | .009273 | .007249 | .888 | -17.0% |
| mixed_mild | .008805 | .009004 | .008264 | .891 | -19.2% |
| mixed_full | .008591 | .008881 | .007108 | .913 | -21.2% |
| explored_full | .007267 | .007439 | .006022 | 1.105 | -33.3% |
| mixed_mild_headonly | .006121 | .006289 | .004696 | 1.772 | -43.8% |

## Interpretation

Mild generated-write mismatch training slightly improves radial error while
reducing combined radial/angular quality and late quality. Full-strength writes
do not rescue the mechanism; using them exclusively is worse. This rejects
these configurations, not local adversarial recovery in principle.

Training the writer with this mismatch signal helps relative to the matched
head-only control: mixed_mild beats mixed_mild_headonly substantially. However,
the control removes BOTH clean and explored mismatch writer gradients. It does
not isolate a benefit of the generated-write gradient from the known useful
clean-prefix gradient. Neither variant beats the original clean recipe.

The old2k reference already ranks nearest-history continuations correctly85.9%
of the time after one fully generated replacement write at prefix32 (88.3% on
clean histories). New scouts score82.8–85.2% after full replacement; none improves
that diagnostic on this panel. Old5k reaches89.8%. The head-only control has the
highest clean ranking91.4%, despite the worst autonomous Q. Better classification
alone remains insufficient. Each model supplies its own proposal, so these are
closed-loop local probes, not D comparisons under identical corrupted inputs.
Mismatched continuations may be plausible under observation noise.

## New process-response windows

Matched radius/speed/direction histories use the same particle, clock, center,
handoff phase and noise. CPU originals and counterfactuals are recomputed together.
Windows contain32 generated points beginning after0,1,8,32,128,512 writes;
these are local-window responses, not instantaneous state measurements.

The saved5k reference's median normalized speed response falls from .859 in the
first32-point window to .192 in the window starting after32 writes, and .00066
in the window starting after128. Ideal response is1. Radius response similarly
falls .293 -> .131 -> -.039. The original process influences early outputs but
its measurable influence weakens with repeated autonomous writes.

Mixed_mild improves initial radius response relative to the2k reference
(.470 vs .246), but it is near zero later. Explored_mild/mixed_full have fairly
strong initial speed response(.837/.828) that also fades. New scouts' late
median radius/speed responses are essentially zero. No sustained retention win.
See diagnostics.md for all windows and the original last256 measurements.

These output probes do NOT establish whether process information is erased from
M or remains stored but unused by G. They do not prove contraction, a particular
attractor, or impossibility of local training. The external clock still sustains
motion, but motion alone does not retain the requested process.

## Recommendation

Keep match_shuffle25 as the leading recipe: saved5k nominal reference, saved2k
late-quality control. Do not extend or tune this failed sweep merely because
radial error improves. No additional experiment is selected or queued.

Before another large training sweep, distinguish loss of process information
inside M from failure to use it: frozen-checkpoint, evaluation-only probes across
autonomous depth, with held-out histories and real-history baselines, would help.
Failure of a simple probe would not prove information absence; compare probe
capacity and data support. Geometry labels would be diagnostic only, never an
added circle-specific training objective.

A possible later training comparison is to align G point-loss context exposure
with explored mismatch contexts. This round intentionally leaves G point feedback
at strength.25, while the independent existing pair branch already reads after
a full write. Broader point feedback was tried before in round10 and did not win,
so it should only be revisited as an explicit interaction with the new mismatch
signal, with a matched control, not advertised as a new general idea.

## Implementation and checks

Config fields: mismatch_context(clean/explored/mixed), mismatch_write_strength,
mismatch_ramp_steps, mismatch_writer_grad. Default clean behavior is preserved.
One detached proposal at target clock minus1, one D write before target; real
continuation/donor targets never enter memory. Prefix and replacement write get
D gradients unless this objective's writer control is disabled. G/prior get no
gradients from this D-only objective. Mixed averages losses equally. Public exact
B-cap and existing loss normalization remain in place.

No MSE objectives, full generated training rollout, geometry cursor/labels,
clipping, EMA, seed sweep, new persistent state or runtime expert. Maximum
sequential generated depth remains one write; independent branches do not chain.
D/G phase complete G calls are5/4 (10/8 internal proposal-reader calls).

75 focused tests passed, including active B-cap, causal timing/targets, gradient
ownership, all-objective exact resume, no-MSE training, process-window calibration,
and diagnostic zero-strength equivalence. Two full-batch4-step GPU smokes with
1024-step evaluation passed. Legacy clean mismatch training is bitwise equal to
commit13d8476 through4CPU updates across39 G/D/prior/RNG tensors.

All five scouts archive identical training source hashes. Saved reference paths
and observed prefixes are bitwise equal to round12. Queue wall709.8s (~11.8min),
1183.8 training GPU-seconds (~19.7GPU-min); implementation, smokes and CPU
diagnostics excluded. No retries, failed runs or extensions.

Artifacts: [leaderboard](leaderboard.md), [results](results.json),
[diagnostics](diagnostics.md), [signal](signal.json), [process](process.json),
[selection](extension_decision.json), [execution](execution.json),
[validation](validation.json), [legacy equivalence](legacy_equivalence.json).

Stable log: `tail -F runs/memory_path/core_round1/train.log`
