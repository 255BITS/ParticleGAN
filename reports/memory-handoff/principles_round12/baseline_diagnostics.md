# Baseline local-signal diagnostics

Evaluation only,128 fixed particles and saved reference processes; all variants on CPU.

| Prefix | Nearest wrong-history rank | Shuffled rank | Earlier point rank | Later point rank |
|---|---:|---:|---:|---:|
| prefix8 | 64.8% | 77.3% | 68.0% | 67.2% |
| prefix32 | 67.2% | 74.2% | 87.5% | 55.5% |
| prefix48 | 69.5% | 77.3% | 85.2% | 60.2% |

Nearest endpoint distance is about.14–.16 versus1.54–1.55 for shuffled donors.
These negatives are mismatched rather than guaranteed impossible; the diagnostic is
a ranking test, not an estimate of perfect-discriminator classification accuracy.

| State at clock288 | Next point MSE | Relative startup error | Following128 Q |
|---|---:|---:|---:|
| autonomous | 1.652511 | 1.7097 | 0.008640 |
| real_full | 0.011406 | 0.1302 | 0.018763 |
| real_recent32 | 0.011404 | 0.1302 | 0.018897 |
| real_recent32_clock32 | 0.004514 | 0.0883 | 0.017518 |

Real-state restoration sharply improves the immediate output. Full288-observation
and recent32-observation encodings behave similarly. Resetting the clock to32
further improves the next point, but none of the128-step diagnostic continuations
passes. This supports investigating autonomous state maintenance; it does not
establish information erasure, rule out reader error, or identify a sole cause.
Restoring state also resets implied position/phase. Long-term promotion uses the
unchanged1024-point panel, not these short intervention scores.
