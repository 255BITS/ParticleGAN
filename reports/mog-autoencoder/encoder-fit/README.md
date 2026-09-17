# Frozen encoder fitting

**Result:** oracle query regression learns most of the available center-selection
improvement. Final MSE falls 25.42% from the bounded checkpoint's zero-offset
baseline, closing 76.94% of its oracle gap. The reconstruction-only control
regresses to 9.81× the starting error. Both use the same architecture, initial
encoder, fresh data, optimizer settings, and 6,000-update budget.

| Path | Hard-choice MSE ↓ | Oracle gap closed | Oracle ID agreement | Correct grid mode |
|---|---:|---:|---:|---:|
| Initial bounded encoder | 0.00280830 | 0% | 48.21% | 100% |
| Oracle query supervision | **0.00209430** | **76.94%** | **72.13%** | **100%** |
| Reconstruction control | 0.02755613 | −2666.70% | 63.40% | 93.753% |
| Exhaustive center oracle | 0.00188026 | 100% | 100% | — |

[Final leaderboard](LEADERBOARD.md) · [configurations and learning histories](results.json) ·
[checkpoint error audit](tail_audit.json)

Oracle-supervised encoder MSE decreases from 0.00211594 at 2k to 0.00210447 at 4k and 0.00209430
at 6k. These are all 100k-example evaluations. The encoder can learn most of the
selection opportunity with direct targets; simply extending the original
reconstruction surrogate under frozen G and particles does not achieve it in
this tested setup. Oracle ID agreement remains below 100%, and 23.06% of the
original selection gap remains. Query regression may average neighboring
targets and does not directly minimize hard-choice reconstruction error.

The control's ID agreement also rises, but its error distribution develops a
large tail. A replay of the final checkpoint finds 6,247/100,000 reconstructions
in the wrong grid mode; those cases contribute 94.10% of total squared error.
Its 99th-percentile per-example coordinate MSE is 0.43635, versus 0.01392 for
oracle supervision. The control's same-grid subset has conditional MSE 0.0017355,
but it excludes the failures and must not be treated as overall performance.
The initial and oracle-supervised models both have zero wrong-grid cases on
this set. Exact agreement with the saved final MSE verifies this replay.

Oracle supervision also raises entropy-effective particle usage from 227.7 to
285.5; the control lowers it to 133.7. Uniform usage is not an objective here,
and the exhaustive oracle itself has effective usage only 217.7. Thus more
uniform use is not proof of more accurate oracle assignments.

**Next recommendation:** add detached oracle-query supervision to joint
bounded-offset training, with a matched continuation control and a fixed loss
weight chosen before running. Retain the actual hard forward choice and fixed
sigma. Test whether the inference benefit survives moving particle centers and
whether generation improves. Generation is unchanged in this frozen experiment:
G and the prior are exactly the same, so its 92 modes / 82.23% HQ are inherited,
not a newly measured gain. The two losses differ in target, gradient, and scale;
this result does not isolate the routing-gradient estimator as the sole cause
of the control's failure. No seed sweep or best-checkpoint selection was used.

## Prespecified protocol

Protocol fixed before launching either arm. Start both from the final bounded
scout checkpoint. Freeze G, particle means, and sigma; set offsets to zero.
Keep the existing encoder architecture and nearest-query hard selection.

```text
oracle_query:
  X -> try all G(p[k]) -> k_best
  E(X) -> query
  loss = mean((query - stop_gradient(p[k_best]))^2)

recon_st:
  E(X) -> nearest k -> G(p[k]) -> X_hat
  loss = mean((X_hat - X)^2)
  backward: original global soft routing surrogate, temperature 0.25
```

Oracle supervision is regression toward the selected particle's latent position,
not classification with a new head. The decoder chooses the targets in data
space. Hitting a distinct target center guarantees selecting it, but imperfect
regression can average targets or cross nearby Voronoi boundaries. No extra
reconstruction or balancing term is added to that arm. The different objectives
also have different scales; no scale or learning-rate sweep is performed.

Both arms use 6,000 encoder-only updates, batch 256, fresh Adam at LR 0.0006,
betas (0, 0.999). Both reset optimizer moments because the original optimizer
jointly trained G, particles, and E under a different objective. Resume the same
training-data RNG state saved after the original 6,000 updates, so both arms see
the same fresh continuation of training data. Seed remains 24002; no seed sweep.
Evaluate at 0/2k/4k/6k on the original 100k held-out examples, with separate RNG.
Rank final models by actual hard-choice coordinate MSE. Report oracle gap closed,
ID agreement, particle usage, same-grid-mode accuracy, and training runtime.
No checkpoint selection from intermediate results.

Check model-state hashes for G and prior (including sigma) at every evaluation;
check the original checkpoint file hash after training. Save each trained encoder,
optimizer, data RNG, full source, configurations, flushed logs, and learning curves.
Unconditional generation is unchanged because only E learns.

```bash
python -u experiments/train_mog_encoder_fit.py \
  > runs/mog_autoencoder/encoder_fit.console.log 2>&1
tail -F runs/mog_autoencoder/encoder_fit.console.log
python -u experiments/analyze_mog_encoder_fit.py runs/mog_autoencoder/encoder_fit \
  > runs/mog_autoencoder/encoder_fit.audit.log
python -m pytest -q tests/test_mog_encoder_fit.py tests/test_mog_oracle.py \
  tests/test_mog_autoencoder.py tests/test_mog.py tests/test_mog_api.py
```

Run directories must be new; the trainer refuses to overwrite them.
All 38 tests pass. Encoder-fit tests check decoded-output targets rather than
latent nearest neighbors, exact hard-query selection, encoder-only gradients,
and hashes that cover particle buffers and extra state. An initial startup
hashing error was fixed before any updates; its log is retained as
`runs/mog_autoencoder/encoder_fit.startup.log`.
