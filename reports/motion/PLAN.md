# HumanAct12 motion completion: first transfer scout

Branch: experiment/motion-completion, from master e0db288.

## Question and fixed comparison

Can the existing four-step Particle DDGAN generate plausible, varied human
motion futures from an observed prefix? Compare hybrid D with MLP D on both
GPUs. First 1k viability, then both 10k regardless of early quality rank unless
there is a correctness or numerical failure. No seed-only repeats, LR decay,
extra losses, skeleton projection, or pretrained networks.

- 8 observed frames -> 16 generated frames, 24 joints x xyz (72 channels).
- About 15 fps in PHSPD v1; use native frame order without resampling. Durations
  are approximate. Display 15 fps; metrics use per-frame units.
- One translation anchors the last observed root. Preserve subsequent global
  displacement and original orientation. Divide by a single training-only RMS
  of root-relative pose coordinates. No per-joint scaling or future-derived context.
- Custom subject split: train P01-P09, validation P10, test P11-P12. This is
  not an official benchmark split. Split before windowing, exclude clips with
  fewer than 24 frames, and reject exact duplicate arrays. Keep a file/hash manifest.
- Draw action uniformly, clip uniformly within action, then a valid window
  uniformly within clip. Independent real batch in D and G phases.
- Evaluate up to 16 deterministically spaced clips/action/split, one midpoint
  window each, eight generated futures per prefix. No cherry-picked windows.
- G: shared temporal U-Net width32 with skips; D: global MLP width256 or hybrid
  global width232 + temporal width32 + fusion116. Approximately equal D parameter
  counts (881,456 vs883,412). Both share the identical G.
- Learned 20k x32 latent particles; Gaussian terminal, forward and reverse-step
  noise; T4 alpha_bar [1,.9,.5,.05,.0001]. Fresh latent draw each reverse step.
- Joint UCD uses 48 heads: (t-1)*12+action. Prefix is explicit D context, while
  action and diffusion timestep select the head and supply D-only CE targets.
- Rp logistic, exact lazy4 bcap with kappa1 and coefficient1, unique-row latent
  VICReg coefficient1, UCD .02, EMA .995. Constant Adam LR .0006, D x1.5,
  prior x10, beta1=0. Same optimizer loop as train_trajectory.py.
- Batch128. At10k:1.28M examples/phase,2.56M total real draws. Equal exposure
  across architectures; report training wall time as well.

## Evaluation

Single-draw ADE/FDE and best-of-eight ADE/FDE compare to one recorded future;
other plausible futures may differ. Report generated pairwise joint diversity,
bone-length error relative to the observed skeleton, boundary displacement and
acceleration, mean speed and acceleration. Compare held-pose, constant-velocity
and recorded references. A pose can be anatomically implausible despite low ADE.

Class-marginal SW1 compares one generated draw/prefix with recorded futures in
that action. This is a small-sample distribution diagnostic, not image FID or
proof of conditional mode coverage. Sort the table by validation SW1 and assess
all metrics together. Test data are for reporting; don't tune against their rank.

Within-action prefix shuffling with all randomness fixed measures whether G
responds to observations; it is not itself evidence of correct conditioning.
Visuals show the first four draws for the first evaluated clip of each action,
with a recorded reference overlay and shared camera/scale. A GIF shows walk and
boxing, first three draws each. No quality-based example selection.

## Reproduce

Install normal project dependencies plus `.[motion]` for the download helper.

```sh
.venv/bin/python -u experiments/prepare_motion.py
.venv/bin/python experiments/train_motion.py  # no-argument 10k hybrid starter
```

Two GPUs, one process each:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -u experiments/follow_grid.py \
  --root results/motion/scout_1k --log results/motion/live.log -- \
  --configs 'configs/motion/scout_1k/*.yaml' --gpus 0,1 --workers_per_gpu 1 \
  --trainer experiments/train_motion.py
# Repeat with confirm_10k in both paths after the scout pair completes.
tail -F results/motion/live.log
.venv/bin/python experiments/analyze_motion.py \
  --root results/motion/confirm_10k --out reports/motion/round1/confirm_10k
```

Read results only after the whole launched pair finishes. Save inference EMA
checkpoints, not optimizer resume state. Source/config/data fingerprints and
runner completion certificates accompany the outputs. Raw data and generated
arrays remain ignored locally.

## Data sources and scope

- [Action2Motion project](https://ericguo5513.github.io/action-to-motion/): official
  HumanAct12 release, 1,191 clips, action taxonomy and data download.
- [Official code](https://github.com/EricGuo5513/action-to-motion): MIT code license;
  loader and skeleton conventions. Its code license is not a separate dataset
  license grant. The released AboutHumanAct12.txt gives metadata and registration
  conventions without separate license terms; keep raw files local and cite it.
- [PHSPD source](https://github.com/JimmyZou/PolarHumanPoseShape): subject-based
  recordings, approximately15fps, dataset use/citation instructions. No SMPL body
  model, images, or private dataset access is needed for these released joint arrays.
- Download URL and per-file IDs/hashes are recorded under data/humanact12;
  actual used/excluded files and custom split are exported with each run.

This is a research transfer experiment on recorded skeletons. No interaction,
actions sent to a robot, rewards, collision model, or claim of RL performance.

## Setup audit and scout transition

All1,191 joint arrays downloaded and verified;1,103 eligible clips and88 short
exclusions. Eligible train/validation/test clips:786/81/236. Evaluation uses192
train,77 validation and174 test prefixes. All12 actions present in all splits.
Raw source SHA256: e8c166350f2704ebdd7e615ff432ac0b1e805c76b2730acacd35633ff2f81903.

34 tests passed across motion, trajectory, and runner. The extracted optimizer
loop was compared verbatim to master, and legacy MLP/hybrid G/D state dictionaries
and initial values matched bitwise. The motion code adds configurable model
channels/classes/context dimensions with original toy defaults preserved.

The1k pair completed without numerical failures; both proceed to10k. A browser
smoke test found that released coordinates have positive Y downward. Corrected
visual orientation before10k without changing training coordinates, losses,
architectures, sampling or metrics. Scout report animations are regenerated from
saved samples with a separate renderer fingerprint. Archived training sources
and certificates remain intact; reporting validates against each run's original
source archive. Run reuse still requires current-source/config certificates.
