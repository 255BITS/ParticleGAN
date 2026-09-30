# Merged-develop K3P quick-screen readout

The registered GPU baseline screen completed **0/3 PASS in 38.915 seconds**.
All three outcomes are scientific failures, with complete observations and no
execution errors. This is a measured rejection of this exact candidate/task
combination; its independent-reference quality is still unknown.

The subsequent adapter audit found that Forge omitted the reference image host's
`[0,1]` clamp after adding noise to real templates. The two image measurements
therefore describe the recorded Forge implementation, and cannot establish
canonical reference-host parity or a formulation-level rejection. Preserve their
negative results and costs while fixing future execution; do not relabel them as
measurements under the corrected data law.

## Frozen execution

- Profile: [develop-20260929-quick-v2](../../configs/forge/calibration/develop-20260929-quick-v2.json).
- [Registration](calibration-lanes/develop-20260929-quick-baseline-v2/registration.json):
  three selected tasks; 1,800 seconds each, 5,400 candidate/campaign ceiling.
- Scientific source: `b4fae98a`, digest
  `44bbbe0db5639cc855874930bbc9f8a98f921d0d702a22a753b27c547bea4cfa`.
- Candidate revision:
  `989a12b2dd014078e0e5dcf7c674be243b5a73bee19289c9500964a941369c69`.
- Request: `cad9a940626a944b3a063210`; fixed screening seed 0; live weights.
- Physical GPU 0: RTX A6000, UUID
  `GPU-ed080e41-3193-3755-6756-f3d46c433331`. The unrelated NPC process had exited
  and the device was idle before enqueue/drain. GPU 1 was not used.

`mode_hold` samples the learned MoG prior. The two image hosts explicitly use
sigma-zero clouds and finite-table enumeration. Scoring excludes training output
noise. Each task recorded all 24 observations and zero unintended RNG deviations.

## Results and cost

| Task | Updates | Verdict | Final modes / required | HQ / minimum | Paid wall seconds |
| --- | ---: | --- | ---: | ---: | ---: |
| mode_hold | 1,200 | FAIL | 5 / 8 | 0.998779 / 0.90 | 17.294100396 |
| img_bars4 | 600 | FAIL | 2 / 4 | 0.937500 / 0.90 | 11.235113958 |
| img_intensity2 | 600 | FAIL | 0 / 2 | 0.000000 / 0.90 | 10.385996893 |

Every task had zero passing observations and zero passing terminal suffixes.
Mode coverage, not sample quality alone, rejected `mode_hold` and `img_bars4`.
The bar host concentrated 78.125% of its particles in one mode; its mass TV was
0.53125. Intensity quality never passed and ended at RMSE 0.175. These curves
show failure to acquire the required quality/coverage, not late overtraining.

A2 was enabled but applied zero times in these hosts; only separate synthetic
activation probes passed. These runs do not establish an A2 training benefit.
Full metrics, mechanism counts and immutable receipts are linked from the
[exact-revision readout](records/readout-9c29229a2baba56de3b40845.json).

The three supervised attempts cost **38.915211247 seconds**, including their
recorded execution overhead. Optimizer-update phase time totaled about 22.798
seconds. Timing under different hardware or contention is a separate comparison;
FLOPs remain unavailable. The CUDA memory probe failed before allocator
initialization, so peak allocated/reserved memory is **unavailable**, not zero.

## Operational checks and provenance

The feature checkout and a fresh clone resolved identical requests and verified
the same immutable source snapshot. After completion, submission from the fresh
clone returned the same request ID. All three selected tasks were reusable,
there remained exactly three attempts, no additional work launched, and paid
cost was unchanged. The concluded request has **zero reservations**.

This was serial execution on one device. It does not satisfy the separate
[physical multi-GPU/cancellation/recovery pilot](MULTI_GPU_PILOT.md).

The candidate's old `claim_contract.sampling_law` string still says
`public_noisy`. The campaign explicitly preregistered clean scoring, and source
plus task receipts confirm clean sampling. Record this declaration discrepancy;
do not reinterpret the old label as MoG noise or alter these receipts. Future
declarations and validation need to make the intended law unambiguous. Diagnostic
results confer no ordinary qualification credit.

The software that ran these tasks passed
[full CI](https://github.com/255BITS/ParticleGAN/actions/runs/36660755141):
1,582 tests and 18 subtests passed, with 16 explicit skips, plus Python 3.10/3.11
smokes and release packaging. Nine skips concern unavailable historical objects
in the shallow CI checkout; those checks passed locally with the pinned history.

Logs remain tail-friendly in the shared queue:

```sh
tail -F /home/martyn/dev/ParticleGAN/runs/forge/calibration-develop-20260929-quick-baseline-v2/progress.jsonl
```

## Calibration and next action

The [reduction](calibration/develop-20260929-quick-v2.md) retains all 16 independent
reference tasks. None is measured in this cohort. The two ablation screens are
also unmeasured, so there are no paired reference classifications or justified
false-accept/reject fractions. Adoption remains **BLOCKED**.

Stop automatic expansion to the control/reference matrix. First diagnose the
host/formulation's acquisition failure and compare initialization, task budgets,
public API settings and intended mechanisms against a bound solvable reference.
Select any further work through a new bounded registration. Preserve thresholds,
the full reference denominator and these negative results; do not rerun failed
science merely to repair reporting. Fix future telemetry and sampling declarations
separately, and retain the physical pilot as an outstanding requirement.
