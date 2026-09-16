# Next: test the original D-memory-to-G handoff

Branch: `feat/sequential-memory-path`. All experiment jobs are stopped. The user
requested a commit and explicitly superseded the queued DDGAN/FiLM exploration
with a core-formulation experiment. Do not restart that queue automatically.
Preserve unrelated `.claude/`, `results/motion/`, and `sparse-ucd.log`.

## User intent and current implementation

The core idea is **real data updates D's M, and G reads that updated M**. Fake
candidate scoring should read that context without overwriting it. D alone trains
the memory writer; G trains its reader and particles. One fixed particle belongs
to each trajectory. Expert-free runtime remains a requirement:

```python
M = zeros()
for each runtime step:
    x = G(z, M)
    emit(x)
    M = D.write(x, M)
```

The observed-prefix prototype in `experiments/memory_path.py` implements the
real-context handoff. The newer autonomous trainers instead construct M from
fully generated prefixes; D separately replays the writer to score full paths.
Replaying the same deterministic writer on the same prefix produces equal state
values, so buffer identity alone is not a scientific difference. The substantive
change was removing the real-context memory handoff from G's training.

The user explicitly asked to compare different core formulations using the
established metrics, without assuming the original or autonomous version wins.
**OOD continuation with no external X is a primary evaluation axis.** No better
alternative has been established. The comparison is **not yet implemented or run**;
this commit preserves a reviewable stopping point before changing training.

## Prioritized next experiment

Use config files and retain simple single-GRU and double-GRU directions. Start
with a simple GRU32 writer and feedforward G to isolate the handoff, then use
metrics to decide whether private G recurrence is useful.

1. **Real-memory handoff only:** D builds M from a real prefix; G consumes that
   actual context. Real next-point and fake candidate are scored against the
   same M without candidate writes. D trains the writer; G cannot change it.
2. **Handoff plus generated feedback:** retain that interaction and add an
   explicitly configured autonomous rollout objective, teaching the runtime
   feedback loop. Keep this separate from the handoff-only result.
3. **Controls:** reuse completed autonomous single-/double-GRU baselines where
   training settings match. If adding a conditional head or changing objective
   scale, state that confound and isolate it where feasible.
4. **Optional curriculum:** config-driven switching from real writes to generated
   writes during a training trajectory. Compare this separately from merely
   adding independent real-handoff and autonomous losses; do not conflate them.

First compare the core write/read/training rules with a common GRU32/feedforward-G
architecture. Then use completed results to choose whether to try private G
recurrence. DDGAN, FiLM, Fourier mappings, and size sweeps remain deferred until
the core-formulation comparison is understood. Do not assume any formulation
wins because it matches the original wording or had the highest old circle rate.

## Primary evaluation: continuing without external X

Every tested formulation must expose an expert-free D/G loop. Keep state/particle
initialization and the point-to-memory write ordering explicit. Evaluate two
different deployment conditions and report them separately:

- **Cold start:** M=0 (and private h=0 if present), fixed z, then generated points
  only. Retain full 256- and 1,024-point metrics; no burn-in exclusion.
- **Real-prefix handoff:** D consumes a declared real prefix, then disconnect X
  completely. G reads the resulting M; only generated points enter subsequent
  D updates. Use fixed prefix lengths (for example 8 and 32) across formulations,
  then measure 256 and 1,024 generated steps after the cutoff. Clearly specify how
  private G state is initialized during the prefix if that variant is used.

For the prefix condition, report continuation fidelity to the original circle
(center/radius, speed and direction), error growth with time, and stopping or
collapse, as well as the existing generated-circle diagnostics. Otherwise a
model could jump to a different valid circle and appear successful. Reference
future points/known toy geometry may be used offline for evaluation only; they
must never enter the rollout after cutoff. Count startup discontinuities.

Track memory norms/saturation and zero/shuffle interventions as diagnostics for
state distribution shift. Label the test as continuation after removal of real
input; do not claim actual state-distribution OOD merely from the label. Report
direction coverage and particle diversity alongside success, and avoid promoting
a one-direction oscillator solely on its circle rate. Match evaluation conditions,
updates, and architecture where possible; disclose compute/parameter differences.
Longer horizons can stress finalists after the common 256/1,024 evaluation.

The user requested compact preparation before working through these experiments.
Do not launch new training during this handoff/commit turn.

Important design details before launching:

- Define the temporal target precisely. Build M from real points strictly before
  the candidate being predicted; do not give G a memory already containing its
  target and call reconstruction next-point prediction.
- Real and fake candidate scores must see the same context. D's real write and
  G's read should be explicit in code and ownership tests.
- In the handoff-only branch, scoring a fake must not mutate the real memory.
  Generated-feedback writes in the autonomous branch are a deliberate separate
  training condition, not an implicit replacement for real writes.
- Include zero-prefix starts in training or explicitly report their absence.
  Evaluate from zero M, with no real prefix and no runtime expert. Conditional
  observed-prefix quality is a diagnostic, not autonomous success.
- Preserve full generated-state BPTT where feedback is trained, while freezing
  writer parameters in G updates. D should score detached fake candidates.
- Keep default API B-cap, no clipping, no EMA, fixed z per trajectory, no analytic
  circle projection/loss, and no seed experiments. Use numerical metrics only.

## Completed evidence

The earlier optimization round completed 12 scouts + 6 continuations (46,000
updates, no training failures). Full-table evaluations enumerate 512 learned
particles; this is not a held-out training split.

| Model | Updates | Full 256 | Full 1,024 | Passing CW / CCW |
|---|---:|---:|---:|---:|
| GRU32 writer + private G GRU64 | 5,000 | 85.2% | 84.0% | 0 / 436 |
| GRU32 writer + feedforward G | 10,000 | 52.5% | 50.4% | 134 / 135 |

Both lose all passing circles under zero/shuffled memory. However, matched
recurrent G without memory reading reaches 52.3% at 2k versus 55.5% with memory;
shared-memory superiority is not established. Keep the writer learned: freezing
it at 5k worsened the 10k result. The toy remains unsolved.

Round 4 completed only `gru_m8`: 22.7% full256 and full1024 on 128 particles,
all 29 passes CCW, 10.2% late stopping. This improves raw success over the prior
GRU32 2k control (14.1%) but worsens direction coverage and stopping. No promotion.
DDGAN and GRU64 were cancelled mid-training; six FiLM configs never launched.

Read:
- [Earlier report](../autonomous-memory/scout/README.md)
- [All 18 earlier checkpoints](../autonomous-memory/scout/completed/leaderboard.md)
- [All-particle validation](../autonomous-memory/scout/validation/leaderboard.md)
- [Stopped round 4 outcome](../autonomous-memory/scout/round4/README.md)
- [Historical DDGAN implementation](../autonomous-memory/ddgan/README.md)

## Infrastructure and checkpoints

`experiments/memory_dispatch.py` provides an appendable shared two-GPU queue.
Jobs specify a config and trainer; both GPUs claim from the same queue. Training
output from all jobs is labelled in one `train.log`; `queue.log` tracks lifecycle.
Drain stdout emits only completion/failure events. Workers block on process/FIFO
notifications rather than polling logs. The completed queue is sealed and its
cancelled jobs are recorded; use a fresh directory next time.

```bash
# Preserved log, no longer live:
tail -F runs/memory_path/scout_round4/train.log

# Future usage, once actual handoff configs exist:
# .venv/bin/python experiments/memory_dispatch.py add --queue FRESH \
#   --trainer TRAINER --configs CONFIG...
# .venv/bin/python -u experiments/memory_dispatch.py drain --queue FRESH
# .venv/bin/python experiments/memory_dispatch.py seal --queue FRESH
```

- Balanced checkpoint: `runs/memory_path/scout_long/gru_flat_10k/model.pt`
- Geometry checkpoint: `runs/memory_path/scout_recurrent_long/gru_private_5k/model.pt`
- Single-GRU 2k: `runs/memory_path/scout_round1/gru_flat/model.pt`
- Double-GRU 2k: `runs/memory_path/scout_recurrent/gru_private/model.pt`
- New GRU8: `runs/memory_path/scout_round4/runs/gru_m8/model.pt`

Raw runs are gitignored; source, configs, tests, and numerical reports are durable.
Use `.venv/bin/python`. No production API source was changed in this scout round.
