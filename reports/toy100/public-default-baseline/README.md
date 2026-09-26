# Public-default baseline handoff

The fresh package 0.8.0 ring baseline has been run at the declared seed.
See [results and recommendations](RESULTS.md) and the lossless logs, declarations
and source snapshots under [evidence](evidence/). The hold and target-shift
protocols are separate measurements; full 22-task qualification remains NOT_RUN.

Worktree: `/home/martyn/dev/ParticleGAN-continuous-search`.
Branch: `codex/k3p-continuous-search` (PR #155).
Merged master: `0ff9a7af` (package 0.8.0).

## What changed

The experiment now has a dedicated [public API runner](../public_default_baseline.py)
using `get_recipe()` and `GANTrainer`. K3P's package optimizers perform the critic
guard, EMA anchor and sparse latent damping; no legacy hooks are installed.
The formula remains the K3P described in [the package guide](../../../docs/k3p.md),
not the exploratory R2 release controller. The master merge also retains the
legacy benchmark compatibility code needed to read/replay older work.

[parameters.json](parameters.json) records the resolved default recipe. The only
recipe override is the declared protocol budget: hold 7500, shift 3600. Both
use batch 2048, 20000 particles, z dimension 2, LR .00425, prior multiplier 2,
Adam (0,.999), penalty coefficient/cap 1, anchor decay .999, guard 5 after 200,
latent damping threshold .5, network horizon cap 1600/floor .01, prior floor .05,
anneal start .6, input noise .5 decaying over .1 of the budget, and output noise
.029 warming over .2 of the budget. Generator/prior EMA decay is .995.

The ring target and host model family remain eight modes, radius 3, sigma .07,
MLPs of width 96 with three hidden layers and critic Fourier depth 3. Model input
shape follows the current recipe's z dimension. Use the declared host seed 0;
there is no seed-sweep option. FP32, deterministic operations, TF32 disabled,
non-fused/non-foreach Adam, one CPU thread. Evaluation uses its own fixed stream.

This is **a fresh baseline**, not an exact replay of the old host-adapted settings:
the current default particle count/batch/latent shape and trainer-owned RNG/noise
schedules apply. In particular, noise follows the declared budget rather than
the historical driver's fixed 1200-step noise horizon. Old initialization fixtures
are not loaded into incompatible shapes. No old 22/22, hold or recovery score is
inherited. Full transfer/native qualification is still separate and NOT_RUN.

## Reproduction

From the worktree root, prepare declarations without training:

```sh
python -m reports.toy100.public_default_baseline --protocol hold --prepare-only --output /tmp/k3p-public-hold-plan
python -m reports.toy100.public_default_baseline --protocol shift --prepare-only --output /tmp/k3p-public-shift-plan
```

The measured protocols used these commands (use fresh output directories for any authorized rerun):

```sh
python -u -m reports.toy100.public_default_baseline --protocol hold --device cuda:0 --output runs/public-default-baseline/hold
python -u -m reports.toy100.public_default_baseline --protocol shift --device cuda:1 --output runs/public-default-baseline/shift
```

Each run writes `declaration.json` with complete parameters and source hashes,
`source.zip`, flushed `metrics.jsonl`, `result.json` and `final-state.pt`.
Watch from another terminal:

```sh
tail -F runs/public-default-baseline/hold/metrics.jsonl
tail -F runs/public-default-baseline/shift/metrics.jsonl
```

The hold reuses the existing observation-only convergence gate: after step 1200,
first confirm 200 consecutive passing updates within 4800 settling updates, then
require the next 1200 updates and all 300 extension observations to pass. Training
runs to its declared budget; a later recovery never erases a failed hold.

The shift translates the target by (1,0) after update 2400. Require all five
stationary checks (1000:50:1200), 120 pre-shift checks (1210:10:2400), and 81
deadline checks (2800:10:3600), each with eight modes and HQ >= .90. A frozen
control is restored from the exact live checkpoint at update 2400, including
optimizer/anchor/history/stream state; it takes no further updates and must pass
zero deadline checks. `shift-state.pt` also saves the real-data stream and target.
The new control is forked in-process, not borrowed from an older run.

Live metrics determine verdicts; EMA is diagnostic.
[The leaderboard](../continuous-practical-leaderboard.md) records both outcomes,
including failures, minimum HQ and runtime.
Delayed/repeated shifts and the full 22-task suite are not implemented as automatic
follow-ups by this entrypoint and must not be claimed from these two ring runs.

## Validation, not qualification

```sh
python -m pytest -q tests/test_public_default_baseline.py tests/test_k3p.py tests/test_k3p_trainer.py tests/test_k3p_selection.py tests/test_continuous_candidates.py tests/test_convergence_gate.py
```

These tests check current/frozen formula parity, package checkpoint behavior,
unchanged historical evidence, zero-training declaration generation, complete
window accounting and full-state frozen-control construction. They are small
CPU correctness tests, not baseline scores. All 54 passed during migration.
The 14 legacy/default/device compatibility checks also passed with
`CUDA_VISIBLE_DEVICES=''`; two device tests explicitly assume CUDA is absent.
