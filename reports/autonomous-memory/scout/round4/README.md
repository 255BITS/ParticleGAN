# Round 4 stopped: test the core memory handoff first

The user superseded the DDGAN/FiLM exploration with an explicit test of the
real-data -> D's memory -> G training interaction, then requested a commit.
The queue was stopped, completed artifacts preserved, and pending/running jobs
marked cancelled. No jobs are running or queued; no promotion was made.

## Completed comparison

Only `gru_m8` completed in this round. The other two rows are previously completed
2,000-update controls, evaluated on the same first 128 particles.

| Model | Full 256 | Full 1,024 | Passing CW / CCW | Late stopped |
|---|---:|---:|---:|---:|
| GRU32 + private G GRU64 (prior control) | 55.5% | 54.7% | 0 / 71 | 0.0% |
| GRU8 + feedforward G (new) | 22.7% | 22.7% | 0 / 29 | 10.2% |
| GRU32 + feedforward G (prior control) | 14.1% | 15.6% | 5 / 13 | 1.6% |

Smaller memory improved raw full-circle success and radial RMSE (.152 vs .237)
but lost clockwise passes and increased stopping. This is not an overall win,
and changing memory size also changes parameter counts. No images were used.
Detailed coverage, interventions and provenance: [leaderboard](leaderboard.md),
[results](results.json).

## Cancelled work

- Running: `gru_ddgan4`, `gru_m64`. Neither reached final evaluation; no result is
  inferred from partial training losses. The trainers save final checkpoints at
  completion, so these cancelled jobs have no final resumable checkpoint.
- Pending, never launched: `gru_memory_film`, `gru_memory_film_only`,
  `gru_private_film`, `gru_memory_w_film`, `gru_memory_fourier_film`,
  `gru_memory_w_fourier_film`.

The code and config files remain available but are not endorsed next steps.
[Queue outcome](queue-outcome.json) records the cancellation. Raw logs remain at
`runs/memory_path/scout_round4/train.log`; they end with `queue_cancelled`.

## Recommendation

Compare core formulations using simple GRU memory before revisiting diffusion or
FiLM. Separate handoff-only training from handoff plus generated-feedback
training; compare both with fully autonomous training. Evaluate zero-memory starts
and real-prefix initialization followed by complete removal of external X.
Continuation without an expert is a primary selection axis. Retain the existing
single- and double-GRU autonomous results as explicitly labelled controls.
Do not treat deterministic replay versus the same mutable buffer as a meaningful
mathematical difference by itself: the substantive difference is what history
populates the memory G reads and which paths may write during training.

The handoff comparison is **planned, not implemented or run**. See the exact
contract and open training choices in [NEXT.md](../../../memory-path/NEXT.md).
