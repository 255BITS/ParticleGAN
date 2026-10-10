# Paired live/EMA copy jitter

## Reproduced defect and repair

RA4 computes copy jitter from the current fast table, then applies that same
displacement to the EMA parent's coordinates. The corresponding EMA table can
have much smaller radii or different coordinate widths. Its newborn can
therefore violate the local kernel bound that EMA sampling itself applies.

The private `pkg-PAIR-EMA` draws the same single noise tensor and evaluates its
displacement separately against the current live and EMA tables before either
table is overwritten. Live rows, all Adam row moments, latent history, graph
updates and RNG consumption remain exactly the RA4 copy law. The EMA row uses
its own parent and its own current radius/coordinate widths. Serving, output
noise, generation API, sample/fake-pool jitter, count tests and action budgets
are unchanged.

Backend schema 4 becomes 5 and checkpoint settings add
`copy_noise_policy=shared_noise_separate_live_ema_current_geometry_v1`.
Old-schema states and states with the new schema but old settings are rejected
before mutation. The trainer schema remains 4. This changes the copy law;
derived sorted-axis caches still remain outside semantic checkpoint state.

## Fixed saved-input evidence

The sampler diagnosis uses eight unchanged CUDA toy checkpoints: RA4 and
matched CUDA E22 at updates 1250, 1500, 1750 and 2000. All probes use the existing
saved `fold128` latent noise and `fold2` output noise. CPU scores are labeled
mechanical reconstructions with one fixed draw per row; the saved canonical
CUDA 8192-sample metrics remain authoritative quality evidence.

| RA4 update | Saved emitted precision | Saved clean precision | Saved emitted modes | EMA radius violations /1024 possible parents | Maximum old EMA radius ratio |
|---:|---:|---:|---:|---:|---:|
| 1250 | 0.876465 | 0.947266 | 22 | 875 | 4.694 |
| 1500 | 0.614014 | 0.663086 | 19 | 889 | 8.884 |
| 1750 | 0.616821 | 0.673828 | 19 | 951 | 5.130 |
| 2000 | 0.758057 | 0.820312 | 21 | 892 | 9.188 |

These are possible-parent kernel probes, not reconstructed historical GPU
actions. Paired EMA geometry has zero bound violations on every probe.
RA4 bounded radii match the exact all-table oracle on about 98–99.4% of rows
in these saved tables. Removing lineage changes fixed-noise precision by at
most roughly 0.2 percentage points. Replacing the running global bandwidth
with current coordinate spread leaves fixed-noise precision unchanged; the
global cap binds at most about 0.12% of RA4 coordinates and the final refreshed
displacement is bit-identical. RA4 generator changes from latent jitter have
RMS about 0.0013–0.0115, versus approximately 0.092–0.103 for E22. These checks
do not explain most of the clean learned/served support decline through noise
or an inaccurate bounded radius.

The nine actual mechanical copy contracts cover unique parents, repeated
manual parents and a simultaneously overwritten parent in each of saved RA4
toy1250, RA4 toy2000 and native RA2 grid100 final state. For 128 fixed native
RA2 possible parents, 81 old displacements exceed the EMA radius, with a
maximum ratio 25.123; paired geometry gives zero. This is a saved z=2 native
table contract and a saved z=128 learned table contract. RA4's running native
screen has output-cloud snapshots but no saved latent trainer state yet, so
its historical copy bounds are not inferred from those clouds.

The contracts verify exact fast-row/moment/history/graph/RNG parity, a single
private noise draw, own-EMA placement, both versioned cache rebuilds, symmetric
graph validity and the 72-candidate bound. Copy continuation after save/load
is bit-identical and derived caches are discarded. A tenth control sets EMA
equal to the saved live table and clears lineage in private memory: all old
EMA output bits and RNG are preserved. No optimizer update, new seed or CUDA
context is used. The earlier equal-control receipt and source are retained;
the final control only moves the helper import before Torch so CPU environment
settings precede numerical imports.

## Integration and limits

`PAIR-EMA.patch` applies to the frozen RA4 package. `READY.json` lists the
decorated `_move` AST splice, backend schema change, added policy setting,
source maps, proof receipts and root-only CUDA contract commands. Integrating
with other new laws must retain this policy tag and an incompatible backend
schema, without replacing their count or controller methods.

This fixes an independently reproduced geometric inconsistency. It does not
establish that the strict learned toy or full native grid stability gates
will pass. RA4's recurrent clean-mode loss, unavailable eligible parents and
stationarity/serving lag are separate measured risks. Root owns any CUDA
verification and subsequent frozen quality run. All prior packages, sources,
metadata adapter, gates, seeds, budgets and old receipts remain untouched.

## Receipts

- `sampler-diagnosis.json`: fixed saved-noise live/EMA geometry and generator probes.
- `cpu-copy-contract-01.json`: nine saved-table copy and resume contracts.
- `cpu-equal-control-02.json`: equal-prior/empty-lineage bit parity control.
- `check_paired_copy.py`: CPU default; `--device cuda` is root-only execution.
- `check_equal_control.py`: the corresponding equality contract.
