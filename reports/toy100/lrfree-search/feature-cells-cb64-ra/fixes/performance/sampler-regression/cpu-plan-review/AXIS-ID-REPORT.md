# AXIS-ID validation

This private RA3 correction removes cached scalar-axis reads and restores the
canonical native harness's sampled-row-ID call. It changes two files only.

Seven focused CPU tests pass, and the existing eleven lineage tests pass.
Saved cold/warm radius, local width and displacement are bit-identical across
six saved cases and rank1/8/64/128 edges. Ties, duplicates, lineage endpoints,
constant tables and four cache versions preserve the original behavior.

| Warm CPU profile | Frozen RA3 scalar reads | Optimized scalar reads |
| --- | ---: | ---: |
| z2, 2048 query rows, chunk256 | 16 | 0 |
| z128, 1024 query rows, chunk256 | 32 | 0 |

These counts match the coordinator's identified per-axis per-chunk barrier.
They do not establish CUDA speedup or explain the full grid100 timing change.

The unchanged canonical screen.py hash is
ee8193adbdf09e93511befae7b6491143c26de88612eddf065cbb92eb2153c3c.
Its exact option resolver now detects indexed generation. Its exact native
argument construction forwards the same row IDs through the fifth positional
argument. Four argument output/RNG behavior, rows keyword behavior, live/EMA
sampling and two trainer loss/state updates match RA3 bits. Supplying both
nonnull ID arguments fails before advancing RNG or graph state.

`AXIS-ID.patch` applies cleanly to frozen RA3. `AXIS-ID-READY.json` records the
base and candidate SHA maps, checks and runner. Backend/schema/kernel settings
are inherited unchanged because the sampler math and semantic state law match
RA3. The root-only paired GPU runner compares cached and cold geometry on fixed
tensors, preserves original evidence and requires zero optimized warm scalar
reads. It has been prepared and remains unexecuted here.
