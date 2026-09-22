# Transfer leaderboard with test importance

**Cosine remains the baseline.** The broader learned-controller search did not
improve practical coverage or transfer. Deliberately poor architectures and
data stresses remain visible without vetoing candidate selection.

| Test importance | Cases | Effect on selection |
| --- | ---: | --- |
| **Required** | 9 | Every established live behavioral regression must sustain success |
| **Ranking** | 16 | Realistic data, dynamics and architecture variations affect rank; failure is nonblocking |
| **Diagnostic** | 8 | Deliberately limited or pathological setups have zero effect on selection |

Every test declares its reason, limitations, family and importance before
fitting. Reference solvability is a separate field: **demonstrated**, **not
demonstrated**, or **unmeasured**. A failed reference does not silently demote a
test. These tiers describe relevance; each test's statistical ability to predict
real-world transfer is **not yet measured**.

[Every test and its rationale](study/README.md#test-importance-and-reference-evidence)
· [Selection rules and reproduction](../../benchmarks/transfer_suite/README.md)
· [Reference calibration](calibration/README.md).

## Development leaderboard

All counts below require sustained success with live weights, a complete
24-observation curve and at least five final passing observations. EMA is
separate. Configuration/posture checks are excluded.

| Controller | Required | Ranking | Diagnostic, no selection weight | Balanced ranking score |
| --- | ---: | ---: | ---: | ---: |
| **Cosine** | **9/9** | **4/16** | 2/8 | **25.0%** |
| Frozen feedback A + cosine | 9/9 | 3/16 | 2/8 | 16.7% |
| Feedback B + cosine | 9/9 | 3/16 | 2/8 | 16.7% |
| Feedback A, gradient inputs disabled | 8/9 | 3/16 | 2/8 | 16.7%; ineligible |

Feedback A is `transfer_g00_p02`; B is `transfer_g00_p04`. The score gives data,
dynamics and image domains equal weight, then weights ranking families equally
within each domain. Extra similar 2D cases cannot outvote images. Diagnostic
results never affect the score or break ties. Eligibility is not an all-tests
PASS: the unresolved ranking failures remain visible.

Eleven distinct policies were compared: cosine, the prior learned equation, and
nine new coefficient proposals across two generations. Six failed required
regressions and were screened before the new fitting cases. The two best eligible
nonzero fitting candidates and cosine advanced to the eight image validation
cases. The remaining candidates are explicitly incomplete for full-development
selection; they are not credited with unrun tests.

Cosine sustains three of six data tests and one of four healthy image tests.
Neither learned finalist sustains a healthy image test. Feedback A solves the
unequal-width mixture that the fixed references failed, but loses the anisotropic
case and the four-blob image case. These are visible tradeoffs, not a uniform win.
No finalist sustains the six ranking dynamics cases.

## Frozen transfer

The annulus, update-cadence and residual-image-architecture families were
declared before fitting and evaluated only after the challenger was frozen.
They did not select or refit it. The feedback-disabled arm retains the learned
constant offsets and cosine; only its four gradient inputs are zeroed.

| Reserved family | Cosine | Frozen feedback A | Feedback inputs disabled |
| --- | --- | --- | --- |
| Continuous annulus | **PASS**, confirmed 467 | **PASS**, confirmed 467 | **PASS**, confirmed 1,467 |
| D updates every second step | FAIL | FAIL | FAIL |
| Residual upsampling image GAN | **PASS**, 4/4 modes, HQ 93.75% | FAIL, 2/4 modes, HQ 71.88% | **PASS**, 4/4 modes, HQ 93.75% |
| **Sustained families** | **2/3** | **1/3** | **2/3** |

The changed-cadence case fails component spread despite high sample quality;
its mean component covariance errors are 15.38, 10.82 and 10.93 against ≤0.85.
The feedback-disabled arm is still ineligible on the required development suite.
Feedback helps retain the old eight-mode behavior but hurts the reserved image
architecture in this comparison. This does not support a transferable default.

## Convergence and timing

| Controller | Required-suite mean confirmation / budget | Eight-mode ring confirmation | Required-suite wall seconds |
| --- | ---: | ---: | ---: |
| Cosine | 0.5850 | 1,050 | 21.92 |
| Frozen feedback A | 0.6123 | 1,150 | 21.18 |
| Feedback B | 0.6035 | 1,100 | 24.70 |

The learned finalists take more normalized updates on the same nine required
tasks. Single observed wall times include setup and measurement and do not
establish a dependable speedup. Confirmation is retrospectively checked against
the complete curve; no early-stop rule is implemented.

## Evidence and limits

The three calibration agents retained **52 reference training attempts**. Four
of the 16 ranking tasks have demonstrated sustained fixed-reference solutions.
The other twelve were unresolved in that calibration; this does not establish
that their data or architectures are invalid. Feedback A subsequently sustains
the unequal-width case. The fixed-reference column and candidate outcomes remain
separate, so that new evidence is visible without changing the test's tier.

Review found that averaging covariance error could accept partial component
collapse. Before controller fitting, versioned vector/dynamics scoring added a
per-component minimum normalized covariance eigenvalue of 0.15. Original runs
and sources are intact; corrected scores are explicit rescoring, not new
training examples. [Correction and original evidence](calibration/README.md#measurement-correction-before-fitting).

The central study retains **245 complete training episodes**, with no numerical
errors, including every screened candidate, post-freeze ablation and reserved
comparison. All use seed 0; there are no seed sweeps. Every episode includes
live/EMA curves, actions, exact configuration, source hashes and runtime.
These are observations and outcomes, not optimal-action labels or resumable
model checkpoints. The search adjusts one shared LR equation, retaining Adam,
cosine and the declared host base rates; it does not learn arbitrary optimizer
updates or tune every model parameter independently.

**54 focused tests pass.** All 33 cosine control live results and full measurement
curves exactly reproduce the earlier host/calibration results. Archive checks
verify episode bytes, fitting-source hashes and frozen selection records.
[Validation](validation.json) · [Complete leaderboard](study/README.md)
· [Raw results](study/results.json.gz) · [Frozen policy](study/policy.json)
· [Source bundle](study/source.tar.gz) · [Run log](study/run.log).

The image cases are small procedural 8×8 GANs with finite particle support;
they do not establish natural-image fidelity or large-network transfer.
Production defaults and the supported training API are unchanged. The existing
unrelated [particle-native CI failure](../ci_status.md) remains documented.
