# Paired GROUP-COUNT CUDA artifact review

PASS. All 88 phase guards and 61 recorded numerical inputs match. Launch command, phase READY, result hash, GPU0 UUID and serial-slot provenance agree. The frozen runner asserts exact full plans and saves matching complete byte hashes; both cases retain RNG, callbacks, work and untouched parameter gradients. This review read existing artifacts only.

| Saved step | Nonzero calls, old → new | Warm median, old → new | Scalar reads | SVD calls |
| --- | --- | --- | --- | --- |
| 1250 | 355 → 55 | 145.37 → 134.14 ms | 399 | 8 |
| 2000 | 347 → 47 | 160.20 → 108.60 ms | 368 | 6 |

Timing is descriptive: three warm repeats per side in reference-then-candidate order, with one profiler run per side. The fixtures use saved RA4 clean tables under the same RA6 birth law. They cover planning, not historical RA6 actuation or the early worst case of 32 linearizations. CPU owner proof separately covers complete copy/birth state effects. Nested profiler totals overlap and cannot be added. This is no quality claim; RA6 learned toy remains FAIL.

## Remaining measured cost

Candidate scalar reads consume profiler CPU totals of 27.26/19.30 ms; SVD consumes 21.68/22.76 ms, with unchanged 8/6 calls. A narrow prospective optimization is one small detached progress/acceptance packet per evaluated iterate, retaining original GPU comparisons, algebra, early exits and trace fields. Reusing final trace scalars avoids duplicate return conversions. It requires separately frozen fixed-input parity before use. SVD offload or masked matrix multiplication changes arithmetic and is outside this exact proposal.
