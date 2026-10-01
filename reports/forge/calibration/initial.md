# Forge calibration: initial

Adoption: **BLOCKED**. Training spent: 0 seconds.

Historical totals are descriptive; prior/runtime cohorts remain separate. Unknowns never become scientific failures.

| Lineage | Smoke | Independent quality | Classification | Smoke seconds | Reference seconds |
|---|---|---|---|---:|---:|
| gapfill-k3p | PASS | PASS | true_accept | 19.33 | 1248.84 |
| gapfill-k3g | PASS | PASS | true_accept | 19.27 | 1357.47 |
| gapfill-k3 | PASS | PASS | true_accept | 20.71 | 1435.18 |
| gapfill-rg5-bcap | PASS | FAIL | false_accept | 19.11 | 1100.90 |
| dv12-ams-rc3-c22 | FAIL | FAIL | true_reject | unknown (0/3 timed) | unknown (0/16 timed) |
| st-10-c22 | FAIL | FAIL | true_reject | unknown (0/3 timed) | unknown (0/16 timed) |
| row-em-renew-all22 | UNKNOWN | FAIL | unknown | unknown (0/3 timed) | unknown (0/16 timed) |
| pr215_exact_host_init | UNKNOWN | FAIL | unknown | unknown (0/3 timed) | unknown (0/16 timed) |
| pr215_qr_adapter | UNKNOWN | FAIL | unknown | unknown (0/3 timed) | unknown (0/16 timed) |
| pr217_qr_native_adapter | UNKNOWN | FAIL | unknown | unknown (0/3 timed) | unknown (0/16 timed) |

```json
{
  "lineages": 10,
  "paired": 6,
  "paired_fraction": 0.6,
  "reference_positives": 3,
  "reference_negatives": 3,
  "all_reference_positives": 3,
  "all_reference_negatives": 7,
  "true_accept": 3,
  "false_accept": 1,
  "true_reject": 2,
  "false_reject": 0,
  "unknown": 4,
  "false_accept_fraction": 0.3333333333333333,
  "false_reject_fraction": 0.0,
  "blocked_lineages": 1,
  "incomplete_lineages": 0,
  "reference_unknown_lineages": 0,
  "smoke_unknown_lineages": 4
}
```

Keep initial gates provisional. Neither profile satisfies frozen adoption criteria. Preserve independent native quality gates; repair evidence/parity before the small proposed campaign, then calibrate new MoG separately.

Exact cards, fixture/package links, missing-cost lists and per-cohort acceptance checks are in the adjacent JSON.
