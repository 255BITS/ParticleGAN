# Finite fitted-reference cell resolution

This private proposal requests128 cells and caps actual fitted K by the
number of even reference rows per effective metric rank:

`K = min(requested_cells, max(1, fitted_rows // max(1, effective_rank)))`.

This regularizes **average** resolution; it does not guarantee rank many
rows in each nonempty cell. Effective rank is the existing even-only rank,
already limited by requested rank, varying feature coordinates and fitted
rows minus one. Rank zero uses denominator1 and keeps the original disabled
metric/discoveries/serving behavior. No odd calibration row, fake query,
support p-value, gate outcome or benchmark label chooses K.

For the existing saved toy reference512/rank8, request128 yields actual64.
For grid even10000/rank8 it yields actual128. The prospective config changes
only `birth_death_cells:64->128`. All model, optimizer, rates, noise,
population, averaging, serving geometry, turnover expiry, action quotas,
parent reservations, birth work bounds and evaluator/fixture gates retain
RA8 source/config values.

## Source and requested/actual audit

Only feature_cells.py changes: one policy constant/helper, the fit's cell
count assignment, backend schema7->8, one versioned settings string, and
strict persisted chart/count metadata checks. Every other package module,
including trainer5, remains byte identical to RA8.

`snapshot.requested_cells` remains request metadata. Settings.cells is the
requested configuration passed into fit. Every center, bin, pool, topology,
target/ledger array and loop, all category assignments and the actual common
count correction use `snapshot.cells`. The family remains actualK+2K+2,
with cutoffQ/(3K+2), including empty categories. Parent reservoir64 and the
5% joint ordinary action budget are unchanged.

Backend8/settings explicitly reject backend7 checkpoints. There is no
production conversion or inherited old-law evidence. For any noninitial
stamp, the loader recomputes actual K from ceil(N/2) even rows and the saved
actual rank, requires stamp/last.cells equal K, validates rank feasibility,
and requires category/multiplicity/cutoff and the even-only count-partition
metadata to agree. Existing strict stamp, scalar, last-chart, duplicate,
topology, future-step and TTL validation is retained. Trainer's existing
backend validation occurs before loading model/optimizer state; trainer
schema5 can remain. Loading still discards the ephemeral chart/head/axis
cache while preserving its typed semantic serving lease.

## Fixed CPU qualification

Freeze source/design/helpers/input hashes before numerical tests. Use tiny
fixed cap/odd/rank-zero/degenerate/initial/malformed metadata controls plus
one actual maybe_apply reaction on each saved RA8 toy1250/2000 input, for
baselineRA8 request64 and proposal request128. Reuse the frozen no-init
saved-weight fixture equations, saved FIFO/moments/history/graph and cloned
saved CPU RNG. Readiness is forced solely to exercise the same reaction;
this is not historical CUDA replay or a training/quality run.

The proposal fixture's recipe request is128. This is explicitly private
fixture metadata, not loading/migrating an old checkpoint. Actual64 must
give bit-identical fitted geometry, count evidence, chosen rows/new latents,
paired EMA actions, optimizer row state/history, supported/group ledger,
birth/copy/counters, modes/gradients/buffers and RNG continuation. Backend
state comparison permits only declared schema, resolution-policy/settings
request differences and the original observational eval_seconds.

Independent state/source and trainer API reviewers own separate strict load,
old-schema atomic rejection and serving/cache/API checks. No GPU, quality
run, new seed, output emission, fixture edit, gate change or production
promotion is authorized by these contracts.
