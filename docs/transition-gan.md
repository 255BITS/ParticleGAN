# A world-model toy without trajectories

Running `python -u examples/transition_gan.py` uses the winning **shared-state
encoder** configuration from [the leaderboard](../reports/transition/leaderboard/README.md).
The default output directory is `results/transition/default`; use `--out-dir` to
choose a fresh directory for subsequent runs.

```text
G1 -> st
G2 -> at
G3 -> st+1

E(st, at) -> z -> G3 -> predicted st+1
G1/G2 -> (st, at) -> E -> z_hat -> G3 -> synthetic st+1

D_joint(st, at, st+1)
D_action(at)
D_state(st at t) and D_state(st+1 at t+dt)   # same state critic
```

Every generator receives the same noisy MoG draw plus observed geometry, time
and preference class. Each is an independent two-hidden-layer MLP. The encoder
adds an observation-to-latent path, with real reconstruction/prediction supervision
and synthetic composition. E never receives real st+1. The shared state critic
uses common physical-state normalization and the actual time of each state.
[Exact objectives, gradient paths and comparisons](transition-gan-encoder.md).

The original direct `G1/G2/G3` path still determines the generation leaderboard;
conditional prediction and synthetic composition have separate metrics. This is
inspired by MisGAN's coordinated networks, and uses complete triples. There is
no missingness generator or mask critic.

The [Lunar Lander follow-up](gym-gan-control.md) now tests finite data, missing
action labels, and actual simulator control. The toy settings and results below
describe the original experiment; [lessons from that follow-up](#what-lunar-lander-taught-us)
explain which conclusions carried over.

Earlier adversarial-only experiments remain available explicitly:
`ucd_joint.yaml` has only a joint critic; `marginals.yaml` adds separate marginal
critics; `monolithic.yaml` generates all six coordinates with one network.
These historical configs retain UCD, unit class gain and no encoder.

## What the toy generates

The reference data comes from the existing analytic two-route obstacle family.
We evaluate only p(t) and p(t+dt), then create:

```text
st   = p(t)                   # position, two coordinates
at   = p(t+dt) - p(t)         # displacement, two coordinates
st+1 = p(t+dt)                # next position, two coordinates
```

Here action means displacement, not force. The known relation is `st+1 = st + at`.
G3 predicts its own next state. We measure this relation after generation; there
is no physics loss and no replacement of G3's output with `st + at`.

Each training item contains one transition. No whole path is constructed or fed
to a model. The two positions use the same hidden route and coefficient draw;
these hidden variables are never supplied as context. Time is a position along
the analytic route, not a diffusion step. Each update uses fresh analytic records,
so this first example does not address limited dataset size yet.

Frozen per-coordinate statistics from 32,768 training transitions normalize all
six outputs. This makes small displacements visible to the critic. Checkpoints
include these statistics; consistency metrics convert back to physical units.

## Recipe and comparison

The example calls `get_recipe("mog")`, changing the component count to **1,024**,
latent dimension to 32, conditioning to explicit two-class inputs, and any explicit step/batch
overrides. Each component is a learned center with Gaussian noise. All branches
share the same selected center **and the same Gaussian noise draw**.

`sigma_rel=0.025` multiplies the initial median nearest-neighbor center distance.
The resulting absolute sigma stays fixed. Centers are standardized when sampled.
The prior spread regularizer sees all 1,024 **raw centers**, once per generator
update, rather than noisy or standardized draws. `prior.json` records calibration.

Defaults are 28,000 updates, batch 256, Rp logistic loss, Adam with G LR 0.0006,
D multiplier 1.5 and MoG prior multiplier **100** (prior LR 0.06). G/D betas are
(0, 0.999), prior betas are (0.5, 0.999). Spread weight is 1, bcap threshold and
coefficient are 1 for every critic on every step, and EMA is 0.995. LR stays constant for the
first 60% of updates, then follows the default cosine decay to 5%.

The default `d_conditioning: concat` feeds two class indicators into each
scalar real/fake critic and has no UCD classification loss. Geometry and physical
time are continuous inputs. Older `d_conditioning: ucd` configs instead select
one of two class scores and add UCD supervision in the D update. The joint critic
has 135,169 parameters for concat versus 134,914 for UCD.

`g_class_scale` controls the scale of the two class indicators supplied to G;
the example default is 8. Historically, `concat_class4.yaml` raised it from 1 to 4 while keeping the latent,
geometry/time inputs, networks and parameter counts unchanged. This tests the
strength of generator conditioning after explicit critic conditioning alone
failed to separate the classes. Restore this constructor setting from the saved
config when loading a checkpoint; it is not a learned weight. `concat_class8.yaml`
raises the same scale to 8; `concat_class8_marginals.yaml` adds three marginal
critics at that scale. These configs retain the same three independent generators.

`g_context_scale` scales G's four observed geometry/time inputs (default 1).
`concat_class8_context4.yaml` tests context scale 4 with class scale 8 and one
joint concat critic. It preserves the shared noisy latent draw, independent Gs,
parameter counts and training budget. Restore both scales from the checkpoint
config using defaults of 1 for older checkpoints; neither is a learned weight.
`concat_class8_context2.yaml` tests the intermediate geometry/time scale of 2.
`concat_class6_context2.yaml` lowers the class scale to 6 at context scale 2.

Each critic role minimizes Rp + bcap, plus UCD when selected. D updates sum these
losses, with bcap computed in the critic's own input coordinates. The historical adversarial-only generator
minimizes `L_joint + marginal_weight * mean(L_state, L_action, L_next)` plus the
single prior regularizer. `marginal_weight=1` by default; joint-only omits the
marginal term. The default encoder averages this adversarial objective across
original/composed paths and adds real triple MSE plus synthetic state/action
reconstruction, each with weight 1. It retains one raw-prior regularizer.
Individual losses are logged in `metrics.jsonl`.

The default has three width-128 MLPs (65,286 G parameters), an encoder with
42,688 parameters, and 203,779 D parameters. The separate-state encoder has
238,084 D parameters. Both use 14,336,000 real training draws plus 32,768
normalization samples. Encoder supervision reuses the real G batch. The
adversarial-only baselines omit E and its losses, so they are a different
supervision/capacity cohort. The board reports those differences explicitly.

## Run

From the repository root, with the experiments dependencies installed:

```bash
.venv/bin/python -u examples/transition_gan.py
```

To reproduce the historical three-way adversarial-only comparison:

```bash
.venv/bin/python -u examples/transition_gan.py --config configs/transition/ucd_joint.yaml
.venv/bin/python -u examples/transition_gan.py --config configs/transition/marginals.yaml
.venv/bin/python -u examples/transition_gan.py --config configs/transition/monolithic.yaml
.venv/bin/python experiments/analyze_transition.py \
  results/transition/mog_1024/branches_joint \
  results/transition/mog_1024/branches_joint_marginals \
  results/transition/mog_1024/monolithic_joint
```

Tail either run through the same file:

```bash
tail -F results/transition/live.log
```

A small CPU check (evaluation still visits every context):

```bash
.venv/bin/python -u examples/transition_gan.py --device cpu --steps 20 --out-dir results/transition/smoke
.venv/bin/python -m pytest tests/test_transition.py -q
```

Use a fresh output directory for each run. Each run saves `log.txt`, flushed
`metrics.jsonl`, resolved config/recipe, source hashes and archive, normalization,
EMA checkpoint, reference/generated samples, a PNG and standalone `viewer.html`.
Encoder runs additionally save `inference.json` and train/test inference arrays.
The historical analysis command writes its report to `reports/transition/mog_1024`.
Checked-in reports/viewers work without local run data. Rebuilding or verifying
pinned leaderboard entries requires their original ignored `results/transition/`
artifacts; those checkpoints/sample archives are not included in Git.

## Persistent leaderboard

The [current leaderboard](../reports/transition/leaderboard/README.md) carries
completed runs across model revisions. Primary ranking is conditional joint SW1;
consistency, marginal distances and preference-class separation are reported
alongside it. Repeated model selection makes these geometries a development
benchmark, rather than an untouched test set.

```bash
.venv/bin/python -u examples/transition_gan.py --config configs/transition/concat.yaml
.venv/bin/python experiments/transition_leaderboard.py \
  --add concat_joint results/transition/conditioning/branches_concat \
  --note "Explicit class input to a scalar joint critic; G and MoG unchanged."
```

The registry is `configs/transition/leaderboard.json`. It pins completed summaries
and source archives, verifies identical data/evaluation functions and reference
draws, and requires the same MoG recipe and update budget except for critic
conditioning. Generator parameter counts must stay within 5% of baseline.
Extra encoder/critic capacity and compute are disclosed. Encoder entries add
paired supervision and form a separate cohort with a fixed E parameter count.
Changed model/trainer sources are permitted;
the original strict analyzer remains useful for comparisons within one revision.
Refresh the report by running `experiments/transition_leaderboard.py` without
`--add`. Registered runs are immutable; use a new configuration and output
directory for each experiment, not a seed repeat.

The [experiment readout](../reports/transition/leaderboard/READOUT.md) explains
current results and recommendations. Earlier conditioning experiments established
`concat_class8_marginals` as the adversarial-only leader at 0.10027 joint SW1.
Increasing class gain helped class separation, but left substantial coverage and
geometry-generalization errors. Context-gain changes traded distribution fit
against transition consistency. These findings motivated the
[encoder and shared-state-critic experiment](transition-gan-encoder.md).


## Previous results

The [completed MoG comparison](../reports/transition/mog_1024/README.md) reports all
three configurations. Added marginal critics improved joint SW1 only slightly
(0.2550 to 0.2500), at about 2.5 times the training time. All three configurations
largely ignored the preference class; the monolithic generator had the best
transition consistency but not the best conditional distribution fit. Subsequent
conditioning experiments are recorded in the persistent leaderboard above.

The [original point-particle comparison](../reports/transition/README.md) remains
as historical context. It used the `gan` preset, 20,000 atoms and 7,000 updates,
so it is not a matched ablation of MoG alone: the prior optimizer and training
horizon also changed. The current default implements the requested MoG recipe.

## Read the results

The main distance is conditional joint sliced Wasserstein-1 (SW1) on normalized
triples, with state/action/next-state SW1 beside it. Report interpolation and the
extrapolation geometry separately. Reference-vs-reference draws give a sampling
floor. Joint support coverage, precision and conditional variance ratio help
identify collapse and excessive spread; their definitions are in the report.

Consistency is `||st+1 - st - at||` in physical units, including median and p95.
Shuffling each block independently **within each fixed context** preserves its
empirical marginal distribution but breaks its relationship to the other blocks.
We evaluate this control on both real and generated triples. A useful joint
diagnostic should detect shuffled real triples even though their marginals match.

The viewer selects a scene, class and time. Blue arrows end at `st + at`; orange
points show independently predicted `st+1`. Red lines show their disagreement.

This comparison establishes whether the architecture learns useful joint
transitions. Testing whether learning all three helps any one marginal needs a
marginal-only baseline. The joint-plus-marginal arm tests added marginal feedback;
it does not yet provide that marginal-only comparison. The Lunar Lander follow-up
now covers sparse complete triples alongside partial records and physical control
rollouts, with the results below.

## What Lunar Lander taught us

The three-generator structure carried over to continuous Lunar Lander with the
same 1,024-particle MoG family and bcap recipe. For choosing actions, we changed
the encoder to consume exactly what is available before acting:

```text
prior -> z -> G1 -> st
           -> G2 -> at
           -> G3 -> st+1

observed st -> E -> z -> G2 -> at
actual simulator.step(at) -> observed st+1 -> repeat
```

Terrain context also enters E and every G. G1/G3 share the encoded latent during
training, but playback needs only E/G2/prior. The simulator advances the world;
G3 predicts a successor for diagnostics. With this state-only encoder, G3 cannot
answer what would happen under an independently chosen alternative action.

The latest matched experiment trains all Gs, E, and the prior from scratch with
GAN losses active throughout, alongside paired reconstruction/action losses.
It retains 9,297 state/successor pairs from 47 heuristic-expert episodes but only
1,010 action labels from five fixed episodes. This is sparse action supervision,
not learning from only five episodes or from no data. Training uses individual
transitions with no trajectory unrolling, reward optimization, or simulator calls.

The [GAN-only leaderboard](../reports/gym/lunar_lander_gan_control/README.md)
selects checkpoints on 20 validation worlds and evaluates them on the same 50
fresh test worlds:

| Controller | Action-labeled episodes | Test landings | Mean return |
| --- | ---: | ---: | ---: |
| New joint GAN | 5 | **34/50** | **157.33** |
| Legacy joint GAN | 47 | 23/50 | 137.35 |
| New joint + marginals | 5 | 16/50 | -0.72 |

The legacy model uses pretrained weights and a different encoder, so only the
two new models form a matched comparison. The joint GAN is the current playable
default, chosen by validation. These counts describe one training run and label
subset per arm; they do not establish reliability across training runs.

The main lesson is that **better measured distribution fit does not necessarily
produce better control**. Adding marginal critics improved prior-sample SW1
from 0.19794 to 0.17363 and expert successor MSE from 0.051738 to 0.047770, while
landings fell from 34 to 16. These Lander SW1 scores pool normalized records;
they are not comparable numerically with this toy's conditional SW1. Some sample
metrics also worsened, including prior precision. See the
[full readout](../reports/gym/lunar_lander_gan_control/READOUT.md) for definitions,
paired outcomes, and uncertainty.

Keep separate measures for joint generation, conditional prediction, and control.
The toy's generation winner remains useful evidence about sample fitting, while
simulator rollouts determine whether a controller works. The follow-up does not
yet establish that GAN training beats a matched non-GAN controller, or that
learning G1/G3 helps G2. A proposed next GAN comparison reduces marginal generator
loss weight from 1 to 0.1 while retaining every discriminator and adversarial
updates throughout; it has not been run.
