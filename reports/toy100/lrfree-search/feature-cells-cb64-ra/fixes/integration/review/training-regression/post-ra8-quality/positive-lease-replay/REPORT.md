# Actual RA8 positive serving lease: prepared CUDA contract

**Frozen; CUDA execution pending root.** CPU metadata preflight passed on the
actual final2000 checkpoint. Its saved stamp is977 coherent rows against973
required, snapshot250, step2000, valid chart and live lease. This is the
original saved stamp, with no synthetic eligibility or policy changes.

The helper/protocol and55 source/input files were sealed at08:15:02.590UTC
before this contract's numerical checkpoint read. Metadata preflight finished
at08:15:34.068UTC with CUDA uninitialized: zero model constructions, forwards,
sample emissions, optimizer/training operations or RNG restores/draws. All
saved private/global RNG buffers have their original CPU uint8 placement.
There were no failed helper attempts. The CPU process exited before this seal.

The root-only command uses two independent restores and two256-row chunks per
branch from the original scorer's existing314259 stream. Branch1 also saves
and reloads the actual trainer/scorer state between chunks. Trace wrappers
delegate the original ordinary sample, latent perturbation and noisy
_generate calls. No scorer, oracle, quality metric, real batch, training or
learned-feature pass is invoked. Every serialized state leaf must be exact;
no timing/state exclusions are allowed. The contract separately checks live
paired serving, retained FAST weights, RNG continuation, row/latent/noisy
output identity and cache rebuilding while the ephemeral chart stays absent.

Budget: four toy generator forwards,1024 repeated mechanical output rows
total, two initial restores, one reload, no training. Each branch repeats only
the first512 rows from the already specified8192-row scorer sequence. Bounded
latent work uses256 query rows and at most72 candidates per row. Numerical
GPU time is expected to be a few seconds; setup, hashing, state validation and
host transfers may take tens of seconds. These are unmeasured estimates.

Root's frozen serial wrapper owns the CUDA launch after the current grid.
Its source and COMPOSITION are pinned. The original1000-step CUDA replay is
unchanged and remains required. This supplemental contract tests a positive
lease's sample/reload mechanics; it supplies no quality acceptance or
positive-lease training-continuation claim.

Evidence: `SOURCE-FROZEN.json`, `cpu-preflight-attempt1/result.json`, retained
CPU log, `READY.json`, `FROZEN.json`; full execution plan in `PROTOCOL.md`.
