# Best measured native lead: paired birth/death on the sigma-floor candidate

This is the strongest **completed** grid100 directional result in the latest
structural batch. It is committed before further native variants are judged.
It does **not** pass the frozen 100-Gaussian gate.

The candidate starts from `sigfloor-v3` and changes exactly one source file:
`birth_death.py` is replaced by the previously tested `bdguard-dRpair`
implementation. The only recipe change is `particle_birth_death: false → true`.
`training.py` remains byte-identical; both use QR `batch_feature_zero`
initialization, learnable output noise initialized at `.02`, and
`total_steps: null`. `birth_death.patch`, `overrides.json` and
`source-sha256.json` reconstruct the candidate from the local LR-free harness.

The baseline was rerun first and reproduced the prior `sigfloor-v3` grid100
result bitwise. Both arms used the same frozen 7,000-update native grid100
fixture, seed, and **noisy** sampling-law score.

| Final live measure | Baseline | Paired BD graft | Frozen requirement |
|---|---:|---:|---:|
| Precision | .97325 | .97365 | ≥.97 |
| Centre RMS / data σ | .3027 | .2595 | ≤.20 |
| Covariance eigenvalue ratios | .338–2.553 | .427–2.012 | .40–1.70 |
| Mass TV | .0610 | .0388 | ≤.06 |
| Radial KS | .0553 | .0117 | ≤.04 |
| Frozen verdict | **FAIL 0/34** | **FAIL 0/34** | final five + holdout |

The independent 100k holdout also fails. The graft's holdout precision is
`.97492` and centre RMS `.24449σ`, still above the `.20σ` limit. There were
14,405 paired moves and zero
neutral-parent clones. The raw `result.json`, noisy `verdict.json`, fixture
and exact test ledger for both arms are committed here.

The gain is a measured direction, not a qualified default: rotated100,
staggered100, the other 19 toy checks, and ring/stationary were not run for
this candidate. Follow-up native experiments are isolated and should only
supersede this entry after their own frozen verdict and holdout complete.
