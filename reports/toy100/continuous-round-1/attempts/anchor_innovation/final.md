**innov3 keeps K3P's ring hold and the 300-update extension, and it loses target-shift recovery completely.** It does not outrank the parent. The pinned K3P sources were not edited and were not rerun.

| Candidate | Hold | Extension | Recovery |
|---|---:|---:|---|
| K3P parent (prior evidence) | 1200/1200, min HQ 0.907 | 300/300, min HQ 0.988 | 28/81, delay 1130 |
| **innov3** | **1200/1200, min HQ 0.993** | **300/300, min HQ 0.991** | **0/81, never stable** |
| innov1 | NOT_CONVERGED, max 6 modes | not reached | not run |
| innov2 | stopped at step 3650, stuck at 5 modes, HQ 0.997 | not reached | not run |

innov3 leaves the learning-rate mix unchanged and always includes the anchor penalty. The reference speeds up only when that residual is large relative to the real R1 term: `follow = P / (P + R1)`, `alpha = 0.001 ** (1 - follow)`. During acquisition that follow was high enough for the reference to catch the critic, and the ring matched the parent (8 modes by step 600, convergence at 1400). After the shift at update 2400, follow rose from 0.13 to 0.50 while G and D stayed at the 1% floor (4.25e-5) and both optimizers kept stepping (2400 updates at the shift, 3600 at the end, no reset). Modes climbed only to 6 and the run ended at 4 modes, HQ 0.488. Zero of the 81 deadline checks passed.

innov1's grad-versus-Adam follow stayed near 0, so the only real change was a full anchor from the second penalty call, and the ring never reached 8 modes. innov2 scaled the penalty by `R1 / (R1 + P)`; that weight averaged 0.67 and produced the same 5-mode plateau.

The anchor formula does not read the training horizon. On this ring driver the binding schedule horizon is still the hardcoded 1200-step noise horizon, and the A/B mix still follows the critic's learning-rate ratio. No toy screen or repeated-change stress test was run, because recovery failed.

Details, replay commands, and artifact paths are in `result.md`. Gate rows are in `tests.jsonl`.
