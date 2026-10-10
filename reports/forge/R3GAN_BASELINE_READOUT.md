# Configurable Modern GAN training baseline on Forge toys

The corrected baseline is configurable and has been run through Forge's ordinary
gates. It **fails the first two-pole movement gate** after all 80 declared updates.
The [current technique leaderboard](technique-inventory.md) includes this
separate training-recipe adaptation with complete 3/19/2 required denominators.
Its [numerical source snapshot](technique-evidence/8bfb0b738e288fbe9ca2e055539469e52c34b27adeda9901b317d44faf07ea2f.json)
retains the frozen evidence and recipe bindings as provenance.

| Technique / recorded source | Tier 1 passes/total | Tier 2 passes/total | Tier 3 passes/total | Measured FAIL | Remaining UNKNOWN |
| --- | ---: | ---: | ---: | ---: | ---: |
| R3GAN Stacked-MNIST training recipe, toy-host adaptation / `77a37364` | 0/3 | 0/19 | 0/2 | 1 | 23 |

One substantive recipe change produced **one attempt, 80 updates, 24 observations,
5.347 paid seconds, zero execution errors and zero remaining reservations**.
The campaign's 44,100-second ceiling was not spent: the first required failure
stopped all later work. There were no seed repeats or post-failure diagnostics.

## Executed recipe and result

The [idea card](../../configs/forge/ideas/r3gan-stacked-training-toy-v1.json)
selects native Adam and the [Modern GAN paper's Stacked MNIST training profile](https://arxiv.org/html/2501.05441v1#A4).
The [configuration guide](../../docs/r3gan-toy-baseline.md) documents every knob
and the public scheduling API. Actual observations in the
[compact run receipt](r3gan-baseline-run.json) confirm:

| Executed quantity | Observation |
| --- | --- |
| Generator-side and critic optimizer types | `torch.optim.Adam` |
| LR, both active roles | 0.0002 for all 80 updates |
| Beta1 / beta2 | 0 / first 0.9, last 0.99 |
| R1/R2 coefficient | First 1, last 0.1; 80 applications |
| Burn-in | Cosine over first 20% of frozen host horizon, then hold |
| Input/output noise, A2, guard, anchor, direct-particle gain | Disabled; zero mechanism applications |
| Finite parameters/state, unintended RNG deviations | All finite, zero deviations |
| Terminal mean absolute particle movement | **0.0114494**, required ≥0.30: FAIL |
| Terminal median critic input gradient | **0.133215**, required ≤1: PASS |
| Passing observations / stable suffix | 0/24 observations; 0/5 suffix |

This is a training-recipe adaptation to existing toy architectures, initialization,
priors, auxiliary objectives and budgets. It does not reproduce the paper's image
architecture, fixed Gaussian latent prior, image count or EMA evaluation.
The two-pole host keeps its direct-particle control, stored critic initialization,
and unsampled particle-prior exception. The fixed screening seed is 0 with named
streams. The requested CUDA cohort executes this CPU-only host on CPU as declared.
Paid time is not a cross-hardware speed ranking.

The earlier matched R1/R2 penalty swap reached movement 0.178637 and gradient
0.064047 under its separate K3P recipe/source. Both fail the unchanged movement
gate; their whole training recipes differ, so no single-knob causal attribution
is supported. The [original readout](TECHNIQUE_INVENTORY_READOUT.md) and generated
report bytes remain intact. BCap retains 3/3, 5/19, 0/2 and recorded tier 1 in
that original cohort; the new row transfers no passes between cohorts.

## Interpretation and recommendation

The requested recipe now executes correctly. Its failure establishes insufficient
transport within this fixed 80-update host. Native quality, coverage and endurance
remain unknown. The smaller LR and the paper's longer budget make this short
screen insufficient to judge the paper's convergence claims.

Stop this exact failed revision. To compare converged toy quality, the next useful
experiment is a separately declared, bounded reference protocol with an independently
justified horizon and unchanged quality metrics. Calibrate the provisional screen
before treating early rejections as a scientific ranking. Do not vary seeds, rerun
unchanged configurations, relax this gate, or fill downstream tasks after failure.

## Reproduction and preservation

Measured source commit: `77a373648e3a7a5f1923015fa2097ec9e0ec877f`.
Source digest: `8bfb0b738e288fbe9ca2e055539469e52c34b27adeda9901b317d44faf07ea2f`.
Candidate revision: `724a672468491e40b3f214c3b44b8913b9317266346527139da7a1de750d8d44`.
Attempt: `ea66e44192c048eaa97f8595c5e55508`.
The [summary receipt](technique-receipts/ea66e44192c048eaa97f8595c5e55508.json)
and [archive manifest](r3gan-baseline-archive.json) bind original receipt hashes.
Compact publications are display artifacts, never qualification inputs.

Local archive: `runs/forge/r3gan-stacked-training-toy-v1/receipts-and-source.tar.gz`.
SHA256: `a0446d48bfb7102995e2ab9d3ecddd18022a30e4db3d6359c6c87712d3ac51a8`.
It preserves the byte-exact envelopes, source snapshot, worker artifacts, full
readout, driver and logs. Obtain the exact archive, verify its hash, and extract
from the repository root to hydrate ignored evidence. Regrading also requires
the recorded runtime/hardware. No bulk logs or state dumps are committed.

```sh
# Regenerate the single current leaderboard; no training/hydration.
.venv/bin/python reports/forge/regenerate_technique_inventory.py
# Independently regrade this source after hydrating its original evidence.
.venv/bin/python reports/forge/regenerate_technique_inventory.py --source-commit 77a37364
# Tail the local campaign logs.
.venv/bin/python -m experiments.forge logs --follow --campaign r3gan-stacked-training-toy-v1
```

A subsequent serialization fix preserves nondefault burn-in fractions when a
schedule endpoint is inactive. Both endpoints were active in this experiment,
so its settings are unchanged. Forge's broad source identity still changes:
frozen replay verifies the actual implementation and does not qualify a newer
checkout. This caused no unchanged experiment rerun.

Subagents verified Adam update parity, penalty units, endpoint/role schedules,
checkpoint resume/atomicity, original-horizon extension, all eight behavioral
hosts, and scalar/image/vector/native adapter dispatch. The final public regression
set passed 141 tests; Forge validation found 47 tasks across six views. Forge
checks passed 769 tests in the broad run plus the corrected zero-start audit
assertion in its targeted rerun (770 distinct checks). Shortened software fixtures
earn no scientific qualification.
