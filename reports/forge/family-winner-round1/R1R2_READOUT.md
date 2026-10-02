# R1/R2 frozen round-one readout

All eight complete configurations retain the **3 smoke / 19 quality / 2 endurance** denominators. This is a display of the certified study report, with no training or independent qualification supplied by this projection.

| Configuration | Critic multiplier / cosine start / floor | Smoke | Quality | Endurance | Status | First observed non-pass | Paid seconds |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: |
| `083deac11d39` | 0.5 / 0.6 / 0.25 | 0/3 | 0/19 | 0/2 | FAIL | two_pole: FAIL — terminal suffix 2/5 | 6.253 |
| `14eeafe7d865` | 0.5 / 0.6 / 0.1 | 0/3 | 0/19 | 0/2 | FAIL | two_pole: FAIL — mean_abs=0.263801 (>=0.3); terminal suffix 0/5 | 6.302 |
| `151d3b970d66` | 0.5 / 0.2 / 0.25 | 0/3 | 0/19 | 0/2 | FAIL | two_pole: FAIL — mean_abs=0.191756 (>=0.3); terminal suffix 0/5 | 5.799 |
| `302b6baa44f6` | 1.0 / 0.2 / 0.1 | 3/3 | 1/19 | 0/2 | FAIL | residual_student: FAIL — identity_mse=0.148376 (<=0.02); success_rate=0 (>=1.0); wrong_pad_rate=1 (<=0.0); terminal suffix 0/5 | 33.145 |
| `4363c95b3f6f` | 1.0 / 0.2 / 0.25 | 3/3 | 0/19 | 0/2 | FAIL | trajectory: FAIL — identity_mse=0.196978 (<=0.02); terminal suffix 0/5 | 25.764 |
| `948182625770` | 1.0 / 0.6 / 0.25 | 3/3 | 0/19 | 0/2 | FAIL | trajectory: FAIL — identity_mse=0.350531 (<=0.02); terminal suffix 0/5 | 26.026 |
| `dad6fdf50034` | 0.5 / 0.2 / 0.1 | 0/3 | 0/19 | 0/2 | FAIL | two_pole: FAIL — mean_abs=0.189384 (>=0.3); terminal suffix 0/5 | 5.944 |
| `e033ba04e5b9` | 1.0 / 0.6 / 0.1 | 3/3 | 0/19 | 0/2 | FAIL | trajectory: FAIL — identity_mse=0.351152 (<=0.02); terminal suffix 0/5 | 26.009 |

Selection: **best_observed**; all trials terminal: **True**; qualified: **False**.

UNKNOWN remains unmeasured. A failed or stopped prerequisite does not turn later tasks into passes. Complete-suite Q1 remains unresolved; the separate supported smoke and trajectory parameter constructions give zero ordinary qualification credit.

No fastest-convergence ranking, robustness claim or default promotion follows from this provisional screen. The paid cost records execution; it is not acquisition time.

Frozen scientific digest `9f89fa7cc7af552ef7e41405fab415abbd00acf3d3c6e4af8e4435fb2423142e`; executed commit `05a2b3c021155fa72471b2293c3fd6d41d1e58a0`. Later source changes do not qualify this cohort as current.

[Every observed scalar metric, verdict, original receipt binding and unknown task](r1r2-execution-readout.json) · [Original configuration-search report](../configuration-search/r1r2-modern-family-round1-v1.json)

Original report SHA256 `22d28269f739705be9fe1bbdaa4640d434886fd006ad92051df920c9a02db4f0`.

The best observed configuration `302b6baa44f6` passes trajectory at identity MSE **1.31488252e-06**. The same shared configuration then fails the residual correspondence question: the raw endpoint has no correct pads and every selected pad is wrong. The numerical gate detects this identity failure; it does not identify a critic, optimizer or conditioning cause by itself.

Within this declared eight-point screen, earlier strong decay with the full-rate critic is the only combination to pass trajectory. This finite observation supports that configuration as a research candidate; it does not establish robustness or justify mixing it with another configuration's later task passes.

Actual completed task devices: cpu 21. The study's declared later CUDA resources do not turn these behavioral executions into GPU measurements.
