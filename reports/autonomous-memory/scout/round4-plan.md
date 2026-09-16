# Round 4: memory size, conditioning, and four-step DDGAN

**Historical plan: stopped and superseded.** The user prioritized testing the
core real-data -> D memory -> G handoff and requested a commit. One scout (M=8)
completed; two running jobs were cancelled and six pending jobs never launched.
No process remains running. See [round outcome](round4/README.md) and
[next experiment](../../memory-path/NEXT.md). Retain the plan below as provenance;
do not restart these jobs automatically.

All scouts train for 2,000 updates on the existing 10,000-update LR schedule,
with 64-point trajectories, batch 128, 512 particles, fixed initialization seeds,
and evaluation on the first 128 particles at 256 and 1,024 points. No seed sweeps.
Keep default exact B-cap, no gradient clipping, no EMA, zero initial memory,
one fixed particle per trajectory, and no runtime expert. Each run saves a
resumable checkpoint and source provenance. Existing 32-value baselines are reused.

| Config | Persistent memory | G read | Comparison / question |
|---|---|---|---|
| `gru_m8` | D-owned GRU, 8 values | concatenation | Can a smaller state learn more stable dynamics? |
| `gru_m64` | D-owned GRU, 64 values | concatenation | Does additional memory capacity help? |
| `gru_memory_film` | D-owned GRU, 32 values | concatenation plus memory FiLM | Does direct memory modulation help? |
| `gru_memory_film_only` | D-owned GRU, 32 values | memory FiLM only | Can modulation replace concatenation? |
| `gru_private_film` | D-owned GRU32 plus private G GRU64 | concatenated M, particle FiLM | Can z modulation recover direction coverage? |
| `gru_memory_w_film` | D-owned GRU, 32 values | concatenation plus MLP(M) FiLM | Does a learned M-to-w translation help? |
| `gru_memory_fourier_film` | D-owned GRU, 32 values | concatenation plus Fourier(M) FiLM | Does periodic encoding help without a translator? |
| `gru_memory_w_fourier_film` | D-owned GRU, 32 values | concatenation plus Fourier(MLP(M)) FiLM | Does translating before periodic encoding help? |
| `gru_ddgan4` | D-owned GRU, 32 values | M conditions four denoising calls | Can within-point computation improve autonomous dynamics? |

Basic memory FiLM directly maps M to per-layer feature scale and shift. It has no
Fourier encoding and no extra persistent state. The three additional scouts
separately introduce a learned `M -> Linear32 -> LeakyReLU -> Linear32 -> w`
mapping, two Fourier frequencies (pi and 2*pi), and their combination. Fourier
features retain raw M/w alongside sin and cos, avoiding forced periodic aliasing.
These mappings belong to G; they cannot train writer parameters. Zero-initialized modulation
preserves the ordinary G's initial function in the concatenation-plus-FiLM case.
The FiLM-only case changes the base input network and initially emits a static
point until modulation learns. Particle FiLM is the existing option driven by z.

Changing M size also changes the writer and G/D input parameter counts; this is
a practical size scout, not a parameter-count-matched causal isolation. Record
parameter counts and wall time. Shared writer/critic/prior initialization is
isolated where shapes permit; the original default configurations reproduce
saved pre-change initialization and 64-step outputs bit for bit.

DDGAN uses the public four-step schedule `[1, .9, .5, .05, .0001]`. M stays fixed
within denoising; only the final clean point is written. z is fixed through both
time axes; Gaussian initial and posterior noise are separate. Training averages
an autonomous path adversarial objective and a paired denoising-transition
objective (configurable weights, initially 1:1), each with default B-cap.
The transition penalty differentiates the candidate with conditions fixed.
Real-prefix transition training is supplemented by full generated-feedback path
training. Only D trains writer parameters. Extra objectives, randomness, and
compute make this a formulation scout rather than a pure depth comparison.

## Queue and logs

The shared dispatcher lets either free GPU claim the next config, including jobs
with different trainers. Appends and claims are locked; pending work is durable.
An interrupted running job is not silently retried. Workers block on process
completion or FIFO notifications; stdout announces results/failures only.

```bash
# One permanent live stream across both GPUs and all jobs:
tail -F runs/memory_path/scout_round4/train.log

# Queue lifecycle (launches, completions, failures):
tail -F runs/memory_path/scout_round4/queue.log

# Append before sealing (the DDGAN job uses --trainer experiments/memory_ddgan_scout.py):
.venv/bin/python experiments/memory_dispatch.py add \
  --queue runs/memory_path/scout_round4 \
  --configs experiments/configs/memory_scout/gru_m8.json

# Run once, then seal once all intended jobs have been submitted:
.venv/bin/python -u experiments/memory_dispatch.py drain \
  --queue runs/memory_path/scout_round4 --devices cuda:0 cuda:1
.venv/bin/python experiments/memory_dispatch.py seal --queue runs/memory_path/scout_round4
```

Do not run the example add command again if its job is already queued; duplicate
names are rejected. Use a fresh queue directory for a subsequent round.

Rank on full cold-start circle success, 1,024-point persistence, both-direction
coverage, and memory interventions. Retain the 2k GRU-flat and 2k recurrent-G
checkpoints as matched-update references. Keep the prior 5k/10k results labelled
as longer-trained context. Promote based on completed metrics, not intermediate
losses or images. The existing toy-solving target has not yet been met.

The user explicitly favors retaining both the single-GRU and double-GRU
directions; neither is superseded merely by proposing diffusion or Fourier
conditioning. Require improvements on the completed leaderboard to justify
longer continuations. Report direction collapse alongside circle success.
