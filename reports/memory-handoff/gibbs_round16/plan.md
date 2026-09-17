# Adaptive round16: history compatibility and useful successor reads

Selected only after all six round15 scouts and information/transition/process
diagnostics completed. See ../gibbs_round15/assessment.md. All new minwarmQ
trailed savedbaseline; writer alignment improved clean decoding but harmed
next-read error and Q. State-only alignment also failed. K's offset-invariant
conditioning effect was modest. No round15 extensions selected.

Preserve original winner and fixed training/evaluation protocol. No MSE GAN
training, long generated trajectories, geometry labels, clipping/EMA or B-cap
overrides. BothGPUs, same pipeline/central tail log. Six fresh2k scouts with
10k schedule and128evalparticles, no seed repeats.

|Config|K candidate|G weight|Writer weight|K history-negative weight|
|---|---|---:|---:|---:|
|state_g10|successor M|.10|0|0|
|joint_match_g10|point + successor M|.10|0|.25|
|state_match_g10|successor M|.10|0|.25|
|read_g10|point + successor read|.10|0|0|
|read_w01|point + successor read|.10|.01|0|
|read_w10|point + successor read|.10|.10|0|

All K heads conditioned on detached starting M; same500step alignment ramp.
state_g10 compares with completed state_w10; joint_match_g10 with joint_g10;
state_match_g10 with state_g10. The read variants isolate the effect of writer
alignment in a different candidate space. All Kwidth128 and default optimizers.

## History negatives

Additional K-only paired classification: actual real successor from anchor's
episode versus a deterministic other-episode real successor. Exclude same
episode; choose donors independently of targets. Normalize K's adversarial
and exact B-cap terms by1+negativeweight. G/writer objectives unchanged. No
additional generation/write, and no analytic labels. This imposes history
compatibility explicitly, though easy coarse differences remain a shortcut.

## Read coordinates

For current real-prefix M and fixed z/time t:
real branch = [observed x_t, G(z,W(M,observed x_t),t+1)];
fake branch = [G(z,M,t), G(z,W(M,G(z,M,t)),t+1)].

The real branch's successor read is detached. G's parameters are frozen for
the fake successor read, but derivatives through its memory input remain.
Particle values are identical; successor-read z is detached to avoid a direct
latent shortcut. G-phase alignment therefore trains the first proposal through
frozen W and the frozen successor reader. Writer-phase alignment trains only
the fake write, through frozen G/K, with the first proposal detached.

This is local adversarial read-consistency/self-distillation: the positive
successor read is current G's interpretation of an observed write, not an
additional ground-truth future point. Existing pairGAN still supplies real
two-observation supervision. The new branch can reinforce a poor reader, so
require downstream metrics. Exactly one generated write; at most two sequential
generated outputs, same bound as existing pairGAN. Each transition phase uses
three G calls total (first proposal + real and fake successor reads), none of
the successor reads is written or continued. Default exact B-cap domain becomes
four candidate coordinates for point+read.

Validation beforelaunch: default memory-space equivalence, exact read-space
resume, phase ownership including frozen-reader parameter/input distinction,
causal times/particles/write counts, donor exclusion, no MSE and GPU smokes.
After completion run same long autonomous metrics and held-out information,
process and offset-invariant transition diagnostics. Sources frozen perround.

Keep extension gates from round15 unchanged. At mosttwo qualifying2k checkpoints
across both rounds may be resumed exactlyto5k. No qualifiers means noextensions.
No third adaptive scout round is planned.
