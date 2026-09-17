# Round16 and two-round outcome

Both requested adaptive rounds completed:12 fresh2k scouts,6 perround, bothGPUs,
zero failures. Round16 was chosen after completed round15 training and all
diagnostics. No scout meets unchanged extension gates, so none was extended.
All full cold/warm256/1024 passes remain0/128. No jobs remain queued/running.
Keep saved round12 match_shuffle25:5k nominal reference,2k stronger lateQ control.

See two_rounds.md for all14 rows (12scouts+2savedreferences), comparison.md for
round16 metrics/probes, ../gibbs_round15/comparison.md for round15, and each
round's results/information/transition/process.json for full evidence.

## What the adaptive round established

Best round16 minwarmQ is read_w01 .009212 vs saved2k .010901 (-15.5%). It beats
read_g10 .006940 by32.7%, but increasing writer weight from.01 to.10 regresses
to.004944. Weak writer alignment can help this branch; writer learning is not
uniformly bad. The best new score across bothrounds is uncond_w10 .009286,
still14.8% belowbaseline and with poorer lateQ. Its almostzero early signed-speed
response also cautions against interpreting its radial gain as retained process
identity. No model qualifies for extra compute.

History negatives improve their matched variants modestly. Joint minQ changes
.008615->.009033 (+4.9%), state-only .007432->.008340 (+12.2%). Joint lateQ32
improves .007180->.008753, but baseline .010869 remains higher. Both history-
negative critics reach100% real-vs-other-episode successor ranking atprefix32.
Their real-vs-generated ranks remain48.9%/47.3%, and correct-anchor paired
margins beat shuffled-anchor margins49.0%/49.6%. These are distinct tasks.
Easy donor discrimination does not establish correct fine-grained continuation
gradients. No claim that conditioning is unused follows from these probes.

The clean-information/behavior split persists. Strong read writer alignment
produces the highest clean probe scores (.891radius/.986speed R2), versus
baseline .612/.942. Yet its next-read generated-write MSE is .04694 versus
.01496baseline, minQ .004944, and generated32 R2 .032/.207. At128 generated
writes all six round16 M-only probes approachchance; M+z results in the full
report likewise do not rescue useful long-term process decoding. Finite probes
do not prove information-theoretic erasure. By128writes median normalized
speed response is nearzero across all12scouts, matching the output-level loss
of process fidelity.

## Formulation and validation

Round15 added a training-only K over one observed versus generated transition,
using detached real-prefix anchors and default exact B-cap. K owns a separate
optimizer. G trains through frozen K/W. Optional writer alignment trains only
the fake write through frozen K, preserving original D losses.

Round16 added config-controlled K history negatives and read-space candidates.
The latter uses current G's next-clock read after the real/fake write; the
real read is detached, fake reader parameters and successor-particle input are
frozen/detached while the memory derivative remains. Thus G alignment reaches
the initial proposal, writer alignment reaches only its fake write. One generated
write maximum; two sequential generated outputs maximum, matching the existing
pair branch's bound. Three G calls pernewreadbranch (proposal, real read, fake
read), with no successor read written. These are adversarial consistency
targets, not new observed future supervision. Existing pairGAN stays active.

No MSE training, analytic cursor, clipping, EMA, B-cap overrides, seed sweeps or
long generated training rollout. All experiments start from the same original
winning config. Fields defaultoff and old checkpoints remain supported. G/D
runtime architecture unchanged; K is training-only.

99 distinct focused tests passed, including exactresume, previous memory-space
and default-off bitwise training equivalence, phase ownership, activeBcap,
fixedparticle/clock, one-write causality, donor exclusion, and diagnostic offset
invariance. Four4step productionbatch GPU smokes across bothrounds passed.
Updated diagnostic test file also passed after adding the exact history-negative
task metric. All archived sources match provenance; sources identical within
each round; realdata,reference and observedprefix panels bitwise matchbaseline.

Combined queue wall time1806.94s (~30.1min), trainingGPU3489.33s (~58.2GPU-min).
This excludes implementation, smokes and completed-model diagnostics. Round16
alone902.42swall,1744.79GPUtrainingseconds. No extensionqueue was created.

## Recommendation

Retain the baseline. Do not claim these results refute local training or the
GibbsNet motivation. The tested local joint/read objectives did not produce
long-term conditional process preservation in this model.

Before another loss/architecture sweep, measure representation drift across
one actual D optimizer update. On cloned savedmodel/optimizer states and the
same trainingbatch, compare baselineD versus writer-alignmentD updates. Hold G,
held-out histories,z andclock fixed to measure changed reads; then measure how
much the ordinary G update repairs the change. This uses atmostone generated
write and geometry labels only for evaluation. Current evidence does not yet
identify optimizer-time drift as the cause: it measures runtime degradation and
differences between trainedmodels.

If large unrecovered drift appears, investigate writer/G update timescales with
matchedcontrols, accounting for earlier frozen/slow-writer experiments. If it
does not, study the frozen composite recurrent map and selective preservation
of process directions. No next experiment or thirdround is queued.
