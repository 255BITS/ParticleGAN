# Saved RA9 versus RA8 training neutrality

Status: **COMPLETE_EXACT_TRAINING_PARITY** across all ten original saved toy endpoints.
Unexpected serialized training differences: **0**.

| Update | Actual K | Rank | Differences | Coherent rows | Serving lease |
|---:|---:|---:|---:|---:|:---:|
| 0 | 0 | 0 | 0 | 0 | False |
| 100 | 64 | 8 | 0 | 95 | False |
| 250 | 64 | 8 | 0 | 259 | False |
| 500 | 64 | 8 | 0 | 521 | False |
| 750 | 64 | 8 | 0 | 852 | False |
| 1000 | 64 | 8 | 0 | 638 | False |
| 1250 | 64 | 8 | 0 | 943 | False |
| 1500 | 64 | 8 | 0 | 881 | False |
| 1750 | 64 | 8 | 0 | 963 | False |
| 2000 | 64 | 8 | 0 | 977 | True |

Every serialized training leaf is compared exactly, including FAST/EMA model and
buffer values, both optimizers and row history, controller/settlers/evidence,
dedicated streams and CPU/CUDA RNG bytes, FIFO/lineage/actions/count and birth/copy
state, serving stamps, work and action counters. Only backend8→7, requested cells
128→64 in recipe/settings, the verified new resolution-policy setting, and the
original last.eval_seconds diagnostic are normalized.

The helper was frozen before any RA9 numerical checkpoint was opened. Both
variants' original post-save log events seal inputs; checkpoint and guarded source
hashes are verified before and after each CPU parse and again after watcher exit.
Actual fitted chart/count metadata is recognized by the frozen production scalar
validator. Existing typed bit-comparison functions are AST-identical to the
previous frozen audit. Original checkpoint objects and global CPU RNG are unchanged.

This is descriptive CPU parsing of saved endpoints, **not numerical replay,
intermediate-step equivalence, emissions, training or quality acceptance**.
No model construction/forward, RNG restore/consumption, CUDA context, optimizer
step or new seed is performed. The original quality gates remain separate.
Outer quality/log/provenance records are outside serialized training parity.
Closed post-exit seal: 2026-09-30T09:22:25.272204+00:00.
