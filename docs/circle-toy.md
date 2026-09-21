# Circle tracing toy: start here

This is the entry point for continuing the circle experiments on branch
`feat/sequential-memory-path`. It is an adversarial particle/memory toy, not an
RL reward environment. The code, configs, metrics and research history are
preserved here; the experiment has not been merged into `master`.

**Current goal:** learn autonomous circle continuation using local training.
**Current result:** unsolved. The latest round scored **0/128** complete cold
circles and warm original-orbit continuations at both 256 and 1,024 steps.
No new experiments were launched for this handoff.

## Training contract (clarified September 21, 2026)

- Shuffled local examples and randomly sampled target positions are allowed.
  There is no requirement to train examples in chronological order.
- Do not play full generated trajectories during training, including detached
  full rollouts. Full autonomous trajectories belong in evaluation with frozen
  weights, after training.
- The current local trainer supports independent next-point predictions and
  bounded local feedback/pairs: at most one generated temporal state transition
  in a local branch. This is the supported starting point.
- Real observations within a context retain their temporal order. The existing
  writer encodes real prefixes sequentially and uses real-prefix backpropagation;
  this implementation is not entirely recurrence-free. Shuffle examples, not
  the observations inside each example. A shuffled-memory mismatch is a deliberate
  negative example, not permission to scramble positive context/target pairs.
- Keep cold generation and warm continuation distinct. Warm evaluation gets an
  initial real prefix, then only generated observations.
- Compare mechanisms, not repeated seeds. Use metrics and leaderboards for
  decisions; summarize results, explanations and recommendations after each run.

Historical configs also preserve adversarial-only training, no geometry/MSE
training objectives, no clipping/EMA, and the public API B-cap defaults. Keep
those fixed for a matched baseline comparison. This handoff changes no losses
or training implementation.

## Get the code and run

Use this branch's source and dependencies together; it predates the current
package on `master`. No external dataset is required: circles are procedural.

```bash
git clone --branch feat/sequential-memory-path https://github.com/255BITS/ParticleGAN.git ParticleGAN-circle
cd ParticleGAN-circle
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[experiments,dev]'

# Inspect the local trainer without starting a run.
.venv/bin/python experiments/memory_handoff_scout.py --help

# Optional baseline reproduction: 2,000 updates, then frozen evaluation.
.venv/bin/python -u experiments/memory_handoff_scout.py \
  --config experiments/configs/memory_handoff/principles_round12/match_shuffle25.json \
  --out runs/memory_path/circle_next/runs/match_shuffle25 \
  --device cuda:0
```

Run commands from the repository root. The output directory must be fresh;
use a new name for each mechanism. `--device cpu` is supported but slower.
In another terminal:

```bash
tail -F runs/memory_path/circle_next/runs/match_shuffle25/experiment.log
```

The line-buffered JSON log includes start, training and completion events.
The run saves `model.pt` (including optimizer and RNG state), `summary.json`,
`trajectories.npz` from evaluation, input/resolved configs, source snapshots and
provenance. Generate a report after all runs in this comparison finish:

```bash
.venv/bin/python experiments/analyze_memory_core.py \
  --source runs/memory_path/circle_next/runs \
  --out runs/memory_path/circle_next/report
```

Read `report/leaderboard.md` and `report/results.json`, then write the mechanism
comparison and next recommendation alongside them. Compare equal update budgets
and also report wall time, parameters and examples per update.

## Baselines and what we learned

The established reference is round12 `match_shuffle25`: a proposal memory adapter,
partial local feedback, a local-pair GAN, and shuffled-history negatives.
The 5k continuation narrowly leads aggregate warm quality; retain the 2k model
as the matched scout baseline and stronger late-window control.

| Reference | Updates | Minimum warm Q ↑ | Full-circle success |
|---|---:|---:|---:|
| Round12 shuffled mismatch, nominal reference | 5,000 | 0.011008 | 0/128 |
| Round12 shuffled mismatch, scout control | 2,000 | 0.010901 | 0/128 |
| Round18 best new separate-state model (`embedded8_dclock`) | 2,000 | 0.007584 | 0/128 |

Q is the worst of the prefix8/prefix32 mean radial-and-signed-motion quality
scores over 1,024 generated steps. It lies in [0, 1] and is **not** a success
probability. The 5k row has more training, so it is not a matched-budget win.
See the [round12 comparison](../reports/memory-handoff/principles_round12/followup/assessment.md)
and [round18 assessment](../reports/memory-handoff/state_round18/assessment.md).

Predicting a next point from real observations worked in the original toy, but
autonomous memory evolution loses the circle's radius/speed information. In
round18, some internal-state variants changed their representation after just
one generated write: a probe trained on real-history states stopped transferring,
although refitting the probe still recovered information. By 128 generated
writes, tested process probes were near chance. This motivates checking the
local real-to-generated handoff before another architecture sweep.

Primary evaluation: cold-circle success and warm original-orbit success at
256/1,024 steps, warm prefixes8/32, both directions. Also report radial error,
signed-speed error, direction agreement, startup discontinuity, late stopping,
early/late Q, good-arc length, particle coverage and zero/shuffle interventions.
Low next-point error or sustained motion alone does not solve the task.

## Recommended next experiment

First measure next-read compatibility before and after **one** generated state
transition, with a real-observation write as control. If that diagnosis supports
the hypothesis, compare a shared internal state-transition rule for real-context
encoding and generated updates, using observations as corrections. Train on
sampled local examples; reserve full trajectories for the frozen evaluation.
Use the round12 2k control at the same budget. Require improvement in both local
compatibility and long-horizon retention before extending a run.

This is a proposed experiment, not an implemented feature or queued sweep.
Do not restart historical sealed queues or simply increase the number of Gibbs
decodes: those experiments did not solve continuation.

## Code and artifact map

| Purpose | Location |
|---|---|
| Local training entry point | [memory_handoff_scout.py](../experiments/memory_handoff_scout.py) |
| Starting config | [match_shuffle25.json](../experiments/configs/memory_handoff/principles_round12/match_shuffle25.json) |
| Circle sampler and original observed-context toy | [memory_path.py](../experiments/memory_path.py) |
| Frozen rollout evaluation | [memory_core_scout.py](../experiments/memory_core_scout.py) |
| Continuous orbit metrics | [memory_orbit_metrics.py](../experiments/memory_orbit_metrics.py) |
| Local trainer tests | [test_memory_handoff_scout.py](../tests/test_memory_handoff_scout.py) |
| Latest completed round | [round18 leaderboard](../reports/memory-handoff/state_round18/leaderboard.md) |
| Reference comparison | [round12 leaderboard](../reports/memory-handoff/principles_round12/followup/leaderboard.md) |
| Detailed historical handoff | [NEXT.md](../reports/memory-path/NEXT.md) |

Start with `memory_handoff_scout.py`; older `autonomous_memory.py` and
`memory_core_scout.py` training modes include trajectory objectives and are
historical comparisons, not the recommended training entry point.

Reports and configs are committed. Raw checkpoints and run directories under
`runs/memory_path/` are ignored by Git and are **not included in a fresh clone**.
Historical continuation configs contain machine-specific `resume` paths. For an
exact continuation, obtain the matching checkpoint, copy its config, set `resume`
to its local path, increase total `steps` within `schedule_steps`, and use a fresh
output directory. Otherwise use the fresh 2k config above; do not claim an exact
resume. On Martyn's existing machine, the saved controls are:

```text
/home/martyn/dev/ParticleGAN/runs/memory_path/principles_round12/runs/match_shuffle25/model.pt
/home/martyn/dev/ParticleGAN/runs/memory_path/principles_round12_followup/runs/match_shuffle25_5k/model.pt
```
