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

- [Results, explanations and validation scope](FIXES-REPORT.md)
- [Machine-readable leaderboard](leaderboard.json)
- [Toy and grid target leaderboard](quality/REPORT.md)
- [Unchanged gates and serial run protocol](quality/PROTOCOL.md)
- [RA6 config](configs/overrides-CB64-RA6.json), [package](pkg-CB64-RA6/) and [source freeze](quality/ra6/READY.json)
- [RA7 config](configs/overrides-CB64-RA7.json), [package](pkg-CB64-RA7/) and [source freeze](quality/ra7/READY.json)
- [RA8 config](configs/overrides-CB64-RA8.json), [package](pkg-CB64-RA8/) and [source freeze](quality/ra8/READY.json)
- [RA8 hypothesis and limits](quality/RA8-PLAN.md)
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
`python -B quality/status.py --variant CB64-RA8`, or tail the original log at
`validation-cb64-ra8/logs/learned-toy-CB64-RA8.log`.
