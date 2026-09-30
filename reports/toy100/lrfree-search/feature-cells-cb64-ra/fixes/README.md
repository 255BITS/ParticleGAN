# CB64-RA diagnostic corrections

This archive contains the experimental corrected packages, their matching
configurations, focused CPU/CUDA diagnostics and matched learned-model results.

**Current target: one package passing both the learned toy and canonical grid100.**
No completed candidate qualifies. E22 remains the broad reference; RA4 has the
best measured MNIST feature distance among these candidates, but fails the toy
and the strict grid gate. Descriptive MNIST gains do not qualify a package.

RA6 completed the unchanged 2000-update toy run and failed: final emitted
precision 0.516968, 23/25 modes and mass TV 0.483032. Its grid was not run.
It combines separately
reviewed repairs for live/EMA copy geometry, bounded novel latent births from
observed real features, and population stationarity after row replacement.
RA5 exercised the same numerical mechanisms but stopped at checkpoint logging
because three diagnostics were tensors; its error is retained separately.
RA6 fixes diagnostic serialization. These repairs have CPU contract evidence;
they do not establish the required quality result. The next repair will be
selected from the frozen final diagnostics and tested as a fresh candidate.

RA7 is the fresh rate test selected from those diagnostics. It lowers constant
generator and learned-sigma base rates by four while preserving prior and
critic base rates, and adds the separately reviewed exact integer group-count
reduction. Its source/config, original fixtures and full schedules are frozen.
Its final toy result fails: precision 0.681641, all 25 modes and TV 0.318359.
Its grid was not run. Adaptive scaling offsets the lower base rate: its final
applied generator rate equals RA6's. Saved clean EMA results are diagnostic;
they do not qualify the emitted distribution.

RA8 completed the original CUDA toy and full grid schedules. It
keeps RA7's config, optimizer laws, copy/birth actions and population
stationarity rule. After each reaction it checks the paired averaged generator
and prior against the current learned support chart: at least 95% of rows must
be supported and agree with their corresponding live row's real-only support
group. This empirical check can permit averaged serving for less than one
real FIFO turnover. Noise and all emitted quality gates remain unchanged.
CPU source and serving API reviews pass. The final GPU toy gate passes:
precision 0.965332, all 25 modes, TV 0.052114 and minimum supported mode mass
0.032715. All ten saved training states match RA7 exactly apart from the declared
new serving metadata and performance counters. All ten toy checkpoints are
independently VALID. The full grid is VALID/FAIL: all five terminal checks fail;
final precision is 0.969950, center RMS 0.203570 sigma, radial KS 0.041203 and
maximum covariance eigenvalue ratio 1.819186. The holdout passes its accuracy
subtest but fails the original frozen gate. RA8 is not recommended for the
joint target. Small saved-state diagnostics are isolating these grid errors.
The check adds one full-population averaged forward per reaction and does not
establish distribution equivalence or a general scaling law.

RA9 is the next prospective config. It requests128 cells and caps the actual
chart size by even real fit rows divided by effective metric rank. The toy
therefore retains64 cells while the grid can use128. Fixed saved-chart
diagnostics show improved grid support separation at128 and sparse toy
regions without the cap. Two matched actual toy reactions preserve RA8's
plans, updates and RNG exactly at actual64. Source/state/API qualification
passed. The fresh unchanged CUDA toy passes with the same final metrics as
RA8, and all ten saved states are VALID with exact training parity. The full
grid completed VALID/FAIL. Its five terminal clouds pass every gate except
center RMS, which ranges0.21111–0.22250 sigma against0.20. The original100k
holdout fully passes: precision0.98173, TV0.01909, center0.19269 sigma and
radial KS0.01453. Final20k precision is0.98210, all100 modes, TV0.03075,
radial KS0.01856 and maximum covariance ratio1.61258. RA9 still does not
qualify for the joint target. Completed small diagnostics isolate persistent
conditional mean bias in the averaged anchors. Reusing existing certified
copy slots cannot repair the final grid state: its last reaction has no
ordinary copy actions. The generator and latent table largely compensate
each other, so their final affine decomposition supplies no evidence for
undoing generator steps. Direct covariance correction is also unsupported
by the saved conditional covariance budgets.
The cap controls average
fit rows per cell; it guarantees no individual cell occupancy. Finer charts
increase geometry cost and count multiplicity. Optimizer, noise and quality
gates retain their RA8 definitions.

The single fixed scratch prototype completed after independent mathematical
and ownership reviews. Its grid witness fired, and 936 of 1000 bounded paired
copies had actual legal progress, reducing the declared feature mean objective
by 12.65%. Both views retained their fine cell, inside category, learned group
and supported status; source rows and optimizer/history inheritance checks
passed. The toy witness vetoed and made no actions. The saved states/files and
global RNG were unchanged. This is CPU feature-objective evidence, with no
new emitted quality result or CUDA replication claim. The learned chart and
FIFO share training data, so the witness is empirical negative evidence
rather than a population or equivalence certificate.

RA10 production implementation is selected from this fixed law. It keeps RA9's
exact config bytes, uses one common 3K+3 test family, and adds a fourth ordinary
mean-copy phase within the existing shared 5% budget. The witness is frozen
before count-driven actions. Actual copy progress uses refreshed post-action
EMA means and group counts in the same even chart. Independent source,
ownership, row-reset, population rebase, checkpoint, serving and matched
continuation contracts pass. A short actual CUDA reaction test also passes:
961 legal grid copies reduce its declared feature mean objective from
1.13930 to 1.00718; the toy mean witness vetoes. These mechanical results
do not establish emitted quality. The original 2000-update CUDA toy passes:
precision 0.965332, all 25 modes, TV 0.052114 and minimum supported mode mass
0.032715. All ten checkpoints are independently VALID. The mean phase makes
no toy moves. The same frozen package completed the original full grid100
schedule: 7000 updates, 34 observations, all five terminal clouds and the
independent 100k holdout. The independent original quality/fixture audit is
VALID/FAIL. Final precision is 0.97945, all 100 modes, TV 0.03185, center
RMS 0.19825 sigma, radial KS 0.03231 and maximum covariance ratio 1.77312
against the unchanged 1.7 limit. All five terminal checks fail covariance;
the first four also fail center RMS and the first fails radial KS. The
holdout passes its fidelity subtest but fails the frozen coverage gate.
387707 cumulative mean copies reduce the logged per-reaction feature
objective; these objectives use changing charts and are not one loss across
training. Final centering improves over RA9, while spread and tails worsen.
The separate final tensor/checkpoint audit is VALID. Historical abbreviated
JSON action lists permit scalar/count checks; full row IDs, inheritance,
lineage and resets are checked at the saved final endpoint. Required CUDA
replay and portability runs have not advanced after this quality failure.
RA10 does not qualify for the joint target. A source-reviewed fixed saved-output
diagnostic will examine clean nonlinear feature means versus the stochastic
emitted law, and all-row versus legal-copy-cohort conditioning.

- [Results, explanations and validation scope](FIXES-REPORT.md)
- [Machine-readable leaderboard](leaderboard.json)
- [Toy and grid target leaderboard](quality/REPORT.md)
- [Unchanged gates and serial run protocol](quality/PROTOCOL.md)
- [RA6 config](configs/overrides-CB64-RA6.json), [package](pkg-CB64-RA6/) and [source freeze](quality/ra6/READY.json)
- [RA7 config](configs/overrides-CB64-RA7.json), [package](pkg-CB64-RA7/) and [source freeze](quality/ra7/READY.json)
- [RA8 config](configs/overrides-CB64-RA8.json), [package](pkg-CB64-RA8/) and [source freeze](quality/ra8/READY.json)
- [RA8 hypothesis and limits](quality/RA8-PLAN.md)
- [RA9 config](configs/overrides-CB64-RA9.json), [package](pkg-CB64-RA9/) and [prospective plan](quality/RA9-PLAN.md)
- [RA9 completed toy/grid result](quality/results/CB64-RA9.json)
- [Fixed mean transport prototype and its limits](quality/MEAN-TRANSPORT-PLAN.md)
- [Completed mean prototype](integration/review/training-regression/post-ra9-quality/mean-category-transport/REPORT.md)
- [Prospective RA10 production plan](quality/RA10-PLAN.md)
- [RA10 selection and sealed evidence](quality/results/RA10-selection.json)
- [RA10 config](configs/overrides-CB64-RA10.json) and [package](pkg-CB64-RA10/)
- [RA10 original CUDA toy result](quality/results/CB64-RA10-toy.json)
- [RA10 completed toy/grid result](quality/results/CB64-RA10.json)
- [RA10 frozen full grid launch](quality/results/RA10-grid-launch.json)
- [RA10 independent completed grid quality audit](performance/training-regression/count-review/post-ra10-quality/grid-canonical-review/receipt.json)
- [RA10 final checkpoint artifact audit](performance/sampler-regression/cpu-plan-review/post-ra9-quality/ra10-grid-artifact-review/accepted-attempt2/receipt.json)
- [Package usage and checkpoint compatibility](USAGE.md)
- [RA4 config](configs/overrides-CB64-RA4.json) and [package](pkg-CB64-RA4/)
- [RA4 numerical source freeze](integration/iteration-4/READY.json)
- [Independent learned/state audit](performance/training-regression/count-review/RA4-ARTIFACT-FROZEN.json)
- [Declared indexed API metadata check](performance/sampler-regression/cpu-plan-review/indexed-metadata/README.md)
- [Independent metadata adapter review](performance/training-regression/count-review/INDEXED-OWNER-FROZEN.json)
- [Accepted RA4 screen results](integration/review/ra4-indexed-api-monitor/summary.json)
- [Copied-file manifest and original local paths](SOURCE-ARCHIVE.json)

The numerical package/config/harness were frozen before evaluation. RA4's
canonical API supplies sampled row IDs to `_generate`; the old collector
expects the previous `plain` API. A separate frozen adapter changes that one
metadata expectation to `indexed`. It preserves every other collector check
and quality gate. Original strict ERROR receipts are retained and acceptance
is explicitly scoped to the declared indexed API.

Each package is a standalone experiment selected before importing particlegan
in a fresh process. Its code and configuration must be used together. The
main package and default backend are unchanged. Raw tensors, datasets and
large profiler traces remain at the local paths recorded in the manifest.

The findings use the existing fixed seeds and saved fixture inputs. Shared-GPU
timings describe these runs; they do not establish a general scaling law.

Read compact current progress with
`python -B quality/status.py --variant CB64-RA10`, or tail the active grid log at
`validation-cb64-ra10/logs/screen-grid100.log`.
