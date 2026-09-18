# Next rungs: 8192 and 16384 learned particles

User steering: continue scaling ParticleGAN, building on the smaller experiments that already motivated learned mixture particles. Do not divert this round into a Gaussian-prior comparison. Target remains FID50k below13, the user's desired BigGAN comparison level; benchmark protocol parity is not established here.

The 4096 arm reached17.8876 at25k and17.4796 at30k, versus1024 control21.8973/21.0193. The current best observed score is17.4796,4.4796 points above13. The40k continuation is still running; these are intermediate measurements, not a completed leaderboard.

## Queued experiments

After both40k continuations finish and certify, run CUDA preflight tests, then two8-update smokes, then the two full scouts using the experiment pipeline:

| GPU | Particles | Original parent | Joint updates | FID50k evaluations |
|---|---:|---|---|---|
|0|8192|CNN E-only10k|10k→20k|initial,15k,20k|
|1|16384|same checkpoint|10k→20k|initial,15k,20k|

Reuse completed1024/4096 scouts as benchmarks; no extra seed or repeated baseline experiments. Using the same original10k parent makes count comparisons directly aligned with that scout. A later extension can compare a promising larger-count arm with the existing4096 duration trajectory. No automatic promotion beyond20k for these new scouts.

Fixed G/D/E architecture and states, rates, sigma, d0, EMA, Adam steps, E-only reconstruction, bcap coefficient1 every8×8, one D update. Reference count1024; expansion factors8/16. Clone per-row moments without LR compensation and preserve standardization/regularizer reference corrections. Original parent/data/noise RNG pairing retained, child selection has a separate persisted stream.

The new standalone trainer is the certified expansion trainer with only allowed count values and diagnostic panel dimensions generalized. Historical/shared files stay unchanged. CPU actual-parent mapping tests passed2/2 before queuing; CUDA identity, optimizer mapping, full-state resume and E-only gradient tests for both new counts are mandatory before smokes/training. A failed stage prevents later stages from launching. Full initialFID50k is audited against19.4482.

Report FID trajectories, wall/training time, per-center exposure and sibling separation. Compare1024/4096/8192/16384 in a single scaling leaderboard. Greater particle count reduces direct sampling exposure per center; a flat short scout may reflect slower adaptation. Do not infer a scaling law or guaranteed path to13 from one count increase.

Persistent queue: `experiments/queue_cifar_ae_scaling.py`; PID/log in LAUNCH.json. Current queue stage is written to QUEUE_STATUS.json. Follow waiting, preflight-stage transitions and full scout training with:

`tail -F runs/cifar_particle_ae/particle_scaling_scout/PIPELINE.log`

Smokes have a separate detailed log at `runs/cifar_particle_ae/particle_scaling_smoke/PIPELINE.log`. On completion the full pipeline writes results.json, scaling_curve.json, LEADERBOARD.md and FINDINGS.md. GPU tests write TESTS.txt; current CPU-only results are CPU_TESTS.txt. No subagents used.
