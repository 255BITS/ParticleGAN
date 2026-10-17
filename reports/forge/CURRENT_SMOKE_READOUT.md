# First current-cohort smoke calibration batch

The frozen `current-k3p-mog-v1` profile measured all nine smoke cells across
three substantive controls at seed 0. Five cells passed and four failed. Every
control fails the overall initial screen. Independent reference outcomes remain
UNKNOWN; false accepts/rejects and the smoke/reference cost ratio cannot yet be
measured. Adoption is BLOCKED under the unchanged criteria.

| Control | two_pole | unused_token_hold | ae_gan_hold | Complete smoke wall seconds | Independent reference |
| --- | --- | --- | --- | ---: | --- |
| k3p | FAIL | PASS | PASS | 16.495 | UNKNOWN |
| forge-onboarding-anchor-ablation | FAIL | PASS | PASS | 16.383 | UNKNOWN |
| forge-no-critic-penalty | FAIL | FAIL | PASS | 18.799 | UNKNOWN |

All three AE cells use the learned-MoG default and pass; the first two tasks
carry explicit particle-cloud/nonsampled exceptions. The ordinary fresh-checkout
penalty ablation supplied one two_pole cell in 7.371 seconds. The registered
nonqualifying diagnostic lane supplied the eight missing cells in 44.306 seconds,
with zero final reservations and no GPU execution. Total observed smoke cost was
51.678 seconds, against separate 900-second walkthrough and 2,400-second diagnostic
ceilings. These wall times include worker startup and independent grading.

All attempted revisions have a concluded readout in compiled memory. Diagnostic
outcomes cannot unlock ordinary qualification. Compare the exact
[profile](../../configs/forge/calibration/current-k3p-mog-v1.json),
[registration](calibration-lanes/current-k3p-mog-smoke-v1/registration.json),
[matrix](calibration/current-k3p-mog-v1.md), and
[parity audit](CHEAP_SCREEN_PARITY_AUDIT.md).

The audit found a redundant metadata defect: inherited noise receipts printed
legacy seed labels while their bound named-stream manifests recorded the actual
Forge seeds. Preserve all original bytes and named manifests. Correct this
reporting defect under a new source/cohort before further execution; do not edit
old certificates. Historical/current learning-rate and L2 policies also differ,
so old K3P passes do not establish an independent positive here. No threshold or
learning update was changed in response to these failures.

Next: validate the metadata correction, freeze the replacement source profile,
and collect only compatible cheap cells. Deeper reference work needs a declared
selection and the requested GPU ownership window. Do not launch an unbounded
matrix or count missing reference measurements as negative results.

## Corrected current source (v2)

The reporting-only correction is frozen at source `c673226c` (software commit
`416d04ca`). The [v2 matrix](calibration/current-k3p-mog-v2.md) completes all nine
cheap cells, with no certificate conflicts and every readout concluded. All nine
[saved verdict/metric pairs](calibration/source-correction-parity.json) exactly
match v1; these source cohorts must not be counted as independent repetitions.

| Control | Smoke passes / required | Complete smoke wall seconds | Whole smoke | Reference |
| --- | ---: | ---: | --- | --- |
| k3p | 2/3 | 16.898 | FAIL | UNKNOWN |
| forge-onboarding-anchor-ablation | 2/3 | 17.008 | FAIL | UNKNOWN |
| forge-no-critic-penalty | 1/3 | 19.305 | FAIL | UNKNOWN |

Total v2 smoke cost: **53.211 CPU wall seconds**. The ordinary fresh-checkout
cell cost 4.800 seconds and the eight registered diagnostics cost 48.411 seconds.
All final reservations are zero. Only the declared diagnostic lane continued
after failures; it cannot confer qualification. Noise receipts now report the
actual named-stream seeds and explicitly null the obsolete legacy offset.

The [two-cell learned-MoG GPU pilot](MULTI_GPU_PILOT.md) is registered against v2,
with a 5,400-second campaign ceiling including at most one deliberate cancellation
repair. It is not enqueued. Both GPU owners must release a coordinated window.
The remaining independent reference matrix, accepted calibration and migration
cutover remain outstanding. No criterion or threshold was relaxed.

Follow-up: the [bounded CPU reference batch](CPU_REFERENCE_READOUT.md) records
three K3P task passes for 20.865 seconds, with full reference still unknown.
It also establishes that an all-smoke-FAIL profile cannot meet the unchanged
positive-reference and zero-false-reject criteria. Preserve v2; further screen
calibration needs a separately declared study.
