# Independent RA4 learned, graph and count artifact review

PASS for evidence: both original learned jobs COMPLETE/VALID, both exact replay audits PASS/VALID. All 20 training checkpoints and four replay endpoints pass independent sparse-graph, backend schema/settings, and count metadata checks. Initial model/prior bytes, original data cursors, saved GPU-typed fingerprints, per-update losses and full semantic replay are validated by the unchanged original checker. Source/config hashes match RA4 readiness; every tensor load used CPU storage and CUDA stayed uninitialized.

| Saved final artifact | Toy | MNIST |
|---|---:|---:|
| Graph edges | 618 | 447 |
| Maximum graph degree (cap 8) | 6 | 4 |
| Cumulative ordinary copies | 10701 | 824 |
| Cumulative isolation repairs | 0 | 0 |
| Final reaction mass/local/global | 1/6/44 | 0/0/0 |
| Final reaction ordinary total | 51 | 0 |

All observed phase splits sum exactly to ordinary moves; ordinary plus isolation agrees with total moves and distinct parent count. Stored family is original K + support 2K + global 2, with 128 categories, 194 overlapping hypotheses and cutoff 0.05/194. The support boundary metadata has 512 even fit rows, ordinal 487, ties inside. Calibration and query identity metadata agree across saved state and checkpoint diagnostics. Graphs are bounded, symmetric, unique, have no self links, and omit transient caches. Both replay branches retain identical graph tensors (toy 692 edges, MNIST 447).

Phase fields describe the latest reaction at each saved checkpoint. Full-lifetime mass/local/global counters are not saved, so their cumulative breakdown cannot be reconstructed from these sparse checkpoints. The total ordinary and isolation counters above are cumulative.

Quality remains separate: final toy precision 0.758056640625, 21/25 modes, mass TV 0.2635400390625 fails all three unchanged toy gates. MNIST class mass TV 0.045684375, confident coverage 10, confident fraction 0.785400391 and mean confidence 0.928826988 are descriptive; no new numerical gate. Toy training took 103.610 s, MNIST 106.499 s. Aggregate count coverage does not qualify the known high-dimensional support-law limitation.

The first supplemental attempt retained the RA3 readiness path and stopped before checkpoint inspection. The private auditor path was corrected; the failed attempt is retained alongside the passing receipt. No production source, initialization, gate, fixture, seed or trajectory changed.
