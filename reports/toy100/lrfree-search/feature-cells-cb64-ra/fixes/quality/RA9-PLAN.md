# Prospective finite-fit cell resolution

Status: proposed private source/config, pending CPU source, state and reaction
contracts. No numerical quality run has started.

## Fixed hypothesis

RA8 passes the final emitted toy gate but fails the full canonical grid gate.
Saved CUDA clouds put the grid's residual center and maximum covariance errors
mainly in its anchors. A fixed current-D/current-FIFO CPU refit shows that 64
cells merge grid supports even though the raw learned features distinguish
them. Requesting 128 cells separates those observed supports. On the smaller
toy reference, 128 cells instead produces sparse fitted regions and vetoes
the otherwise successful averaged view.

Request 128 cells and use

`actual_cells = min(requested_cells, max(1, even_fit_rows // max(1, metric_rank)))`.

The denominator is the effective metric rank fitted from observed real
features. The cap reserves at least that many even fit rows per cell on
average. It does not guarantee occupancy or sufficient estimation accuracy in
every cell. It uses neither oracle labels nor benchmark identity.

For the frozen fixtures, toy512/rank8 retains64 cells and grid10000/rank8 uses
128. The only proposed config change from RA8 is `birth_death_cells: 128`.
Optimizer bases, learned noise formula/floor, serving geometry/expiry,
population stationarity law, birth/copy bounds and original quality gates
remain unchanged. The existing common count family remains `3K+2`, evaluated
at actual fitted K: its size rises194 to386 on grid. This reduces local count
power as well as increasing geometric resolution.

## Required source/state contracts

- Explicit backend schema8 and a serialized finite-fit resolution policy;
  trainer schema5 retained. Old backend7 states are rejected before model
  mutation. There is no production checkpoint migration.
- Stamp rank/cells, real split/calibration rows, category count, multiplicity
  and cutoff must agree with the same reaction and the requested config.
- Invalid rank/row counts and degenerate charts must conservatively veto
  averaged serving. Derived caches remain absent after load.
- Matched actual saved toy reactions1250/2000 must preserve all original
  numerical plans, updates and RNG draws at effective64 cells. Fixture
  metadata rebinding for this CPU comparison must be labeled explicitly.
- Independent source and actual serving/reaction reviews precede root freeze.

## Quality and cost

Run the original2000-update CUDA toy first. Only a final PASS proceeds to the
complete7000-update grid, all34 observations, five terminal20k clouds and the
independent100k holdout. Required replay/state/portability checks follow a
candidate passing both quality targets. Every failed result remains negative
evidence, with no earlier checkpoint substituted for final acceptance.

Finer resolution increases O(NK) chart work, bounded K-by-K topology work,
storage and count multiplicity. The fit-sample cap and rank8 do not establish
universal support resolution, isolated GPU scaling laws or readiness for
arbitrary larger architectures. Main package/default promotion requires its
own completed evidence and is outside this experimental configuration.
