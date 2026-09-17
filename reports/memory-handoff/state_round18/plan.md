# Separate generator dynamics, round18

Hypothesis: G's recurrent state loses the process partly because it must infer
it again from a two-coordinate emitted sample at every update. Compare a shared
G-owned GRU driven by (a) learned observation features, (b) final decoder features,
or (c) their equal mixture. All six recurrent scouts have the identical architecture,
initial weights, eight state coordinates, and no access to D memory. Each pair
changes only D Fourier clock conditioning. A seventh scout adds D clock to the
saved round12 winner. This clock setting already exists in the public experiment
config; no UCD/classification objective is added. Random initial circle phase
makes absolute time classification questionable; ordinary conditioning need not
infer time from the point. Both real and fake receive identical conditioning.

Real-prefix encoding uses a learned 2->64 observation projection followed by the
same G GRU used at runtime. Final decoder features before the 2D output layer
provide its generated input in intent mode. Proposal reads do not advance state.
The decoder still uses the fixed particle and G's Fourier clock. Hybrid mixes
internal features and encoded final output equally. Embedded observation control
has exactly the same trainable parameter layout and initialization.

At runtime intent state does not depend on the emitted coordinates; hybrid and
embedded state do. D owns/trains its observation memory and scoring heads. G owns
its encoder, decoder and state cell, trained only through existing GAN losses
and default particle regularization. No D state reaches G, including through
proposal adapter. D writes remain in evaluation for instrumentation; G paths
can be produced without them. D weights may influence G only via training loss.

Point feedback starts from a real prefix, mixes one true/generated observation
for D as before, and mixes corresponding real/internal features for G intent or
hybrid. Strength zero is teacher forcing, strength one is the runtime transition.
The embedded control uses the same feature-space mixing, with encoded generated
output in place of internal features. Thus all three share the mixing rule.
All modes share fully generated transitions in the independent local pair branch.
No generated history is chained between branches. At most two generated temporal
outputs and one intervening generated transition per branch. Real-prefix BPTT
is retained. No full generated rollout training, MSE training objective, geometric
cursor, seed repeat, clipping, EMA, or B-cap override.

Seven 2000-update scouts, common 10000-update schedule, current best
match_shuffle25 base, both GPUs through memory_dispatch. Initial four-update
full-panel smokes precede launch. Stable tail:
`tail -F runs/memory_path/core_round1/train.log`.
Training sources freeze before launch. Inspect completed jobs only.

Primary metrics remain 256/1024 cold circles and warm original-orbit passes at
prefix8/32. Secondary: minimum warm1024 continuous quality Q, late Q, radial
error, direction, stopping, and longest good arc. Q is not a success probability.
Keep both existing2k/5k baselines. Existing fixed extension gates unchanged:
up to two exact2k->5k continuations if both warm pass fractions improve, OR Q
improves >=20% at both prefixes, late Q no worse, radial <=5% worse and direction
<=2 percentage points worse; either route requires cold late stopping <=1 point
worse. Rank qualifiers by min warm passes then min Q. No extension if none qualifies.

Completed diagnostics: identical held-out radius/speed probe panels for D memory
(all scouts) and G memory (six recurrent scouts), depths0/1/8/32/128, with
real-write controls and particle-conditioned probes. Probe regression is evaluation
only. Process interventions and standard independent D/G memory read interventions
measure behavior; shuffling sensitivity alone does not establish benefit. Finite
probe failures are not proof of information erasure. D clock effect is judged by
matched trained controls. No cross-model comparison of raw state coordinates.
