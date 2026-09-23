# Reference calibration and test importance

These runs check what the proposed tests measure and whether fixed reference
configurations can solve them. They do not fit a controller, choose a new default,
or evaluate a reserved family.

| Development domain | Ranking cases sustained by a reference | Diagnostic cases sustained by a reference | Actual training attempts |
| --- | ---: | ---: | ---: |
| [Vector distributions](vectors/README.md) | 3/6 | 1/2 | 12 |
| [Training dynamics](stress/README.md) | 0/6 | 1/2 | 16 |
| [Procedural images](images/README.md) | 1/4 | 0/4 | 24 |

All 52 attempts use seed 0, one CPU thread, complete 24-observation curves, live
selection metrics and separate EMA. No seed sweeps were performed. A reference
failure means solvability is **not demonstrated within this setup and budget**;
it does not establish impossibility or change the declared importance tier.

The diagnostic uniform-image generator is unable to represent stripes. The
diagnostic mean-only critic cannot distinguish equally bright patch locations.
Their failures remain visible and have zero selection weight. Conversely,
realistic unequal-width and unequal-mass mixtures remain ranking challenges
even though the tested references have not sustained their spread requirements.

## Measurement correction before fitting

Review found that an average component covariance bound could accept a mixture
where several components had collapsed to their centers. Protocol v2 adds a
minimum normalized covariance eigenvalue of 0.15 for every identifiable component.
The anisotropic case already had that bound. Nonidentifiable overlapping data
continues to use distribution-level criteria.

Original v1 training runs and exact sources are preserved byte-for-byte. V2
explicitly rescores their already-recorded per-component metric; it does not
retrain them or overwrite their original verdicts. The test tiers do not change.
Both vector and dynamics archives include the original execution specification,
corrected scoring specification, source hashes, and an explicit no-retraining
record. The counts above use corrected v2 scoring. Independent target samples
and partial-collapse regression tests validate the corrected measurement.

Each domain archive contains compressed original JSON, original-byte hashes,
source bundles, logs and reproduction scripts. Reused or rescored observations
are not independent training examples. The central search uses the corrected
criteria from its initial frozen manifest.
