# Concluded R1/R2 repairs from PR #401

Preserve the completed research and stop these exact repair revisions. All 34
complete measured revisions fail Ring16. The strongest complete repair cohorts
pass five of six ordinary tasks and their own clock audit; none solves the full
original Tier 1 suite. This archive imports historical context into recall and
does not change current qualification, recipe selection or defaults.

The [current technique inventory](../technique-inventory.md) remains the only
generated leaderboard for this goal. Counts below retain the original task/view
contracts and cannot fill newer requirements. The immutable
[original PR #401 source](https://github.com/255BITS/ParticleGAN/tree/2cf7b0a4c796e29ace6cca12d2e553ad67211b5c)
preserves full reports, final metrics, reproduction sources, declarations,
certificates and actual-training GIFs. [Archive identities](archive.json) pin
each report tree and authoritative document/result blobs with SHA-256 hashes.
The remote branch `archive/pr401-2cf7b0a4c796e29ace6cca12d2e553ad67211b5c`
also retains that original head. No original commit was rewritten.

| Completed comparison | Recorded outcome and limit |
| --- | --- |
| [Scheduled baseline](https://github.com/255BITS/ParticleGAN/blob/2cf7b0a4c796e29ace6cca12d2e553ad67211b5c/reports/forge/r1r2-baseline/README.md) | 20/22 repeated view requirements, representing five PASS/two FAIL across seven distinct tasks. Ordinary 5/6; Ring16 and clock fail. |
| [Constant policies and half rates](https://github.com/255BITS/ParticleGAN/blob/2cf7b0a4c796e29ace6cca12d2e553ad67211b5c/reports/forge/r1r2-tier1-search/README.md) | Nine constant recipes pass clock; all fail Ring16. Best ordinary result 4/6. |
| [Five repair tracks](https://github.com/255BITS/ParticleGAN/blob/2cf7b0a4c796e29ace6cca12d2e553ad67211b5c/reports/forge/r1r2-five-track-repair/README.md) | AMSGrad reaches 5/6 plus clock; every track fails Ring16. |
| [Optimizer roles](https://github.com/255BITS/ParticleGAN/blob/2cf7b0a4c796e29ace6cca12d2e553ad67211b5c/reports/forge/r1r2-role-repair/README.md) | Seven new recipes pass clock; all fail Ring16. Momentum plus AMSGrad reaches 5/6. Best observed covariance .904193 still exceeds .85. |
| [Direction/denominator](https://github.com/255BITS/ParticleGAN/blob/2cf7b0a4c796e29ace6cca12d2e553ad67211b5c/reports/forge/r1r2-direction-floor-repair/README.md) | All three new recipes reach 4/6 plus clock; Ring16 and words fail. |
| [Spectral updates](https://github.com/255BITS/ParticleGAN/blob/2cf7b0a4c796e29ace6cca12d2e553ad67211b5c/reports/forge/r1r2-spectral-repair/README.md) | Both recipes reach 4/6 plus clock; Ring16 and words fail. |
| [Objective/optimizer memory](https://github.com/255BITS/ParticleGAN/blob/2cf7b0a4c796e29ace6cca12d2e553ad67211b5c/reports/forge/r1r2-objective-memory-repair/README.md) | Sliced transport and AMSGrad m110 each reach 5/6 plus clock; Ring16 fails. |
| [Native score law](https://github.com/255BITS/ParticleGAN/blob/2cf7b0a4c796e29ace6cca12d2e553ad67211b5c/reports/forge/r1r2-score-law-repair/README.md) | Non-saturating AMSGrad m110 reaches 5/6 plus clock; Ring16 fails. |
| [Weaker penalties](https://github.com/255BITS/ParticleGAN/blob/2cf7b0a4c796e29ace6cca12d2e553ad67211b5c/reports/forge/r1r2-weak-penalty-repair/README.md) | Gamma .01 regresses to 2/6 plus clock; cached gamma .1 parent retains 5/6. |
| [Raw signed energy-U](https://github.com/255BITS/ParticleGAN/blob/2cf7b0a4c796e29ace6cca12d2e553ad67211b5c/reports/forge/r1r2-energy-distance-repair/README.md) | Fresh same-source control and energy recipe both reach 5/6 plus clock. Energy restores spread while worsening covariance and HQ. |
| [Energy with frozen prior](https://github.com/255BITS/ParticleGAN/blob/2cf7b0a4c796e29ace6cca12d2e553ad67211b5c/reports/forge/r1r2-energy-prior-freeze-repair/README.md) | Same 5/6 plus clock. Freezing locations does not repair Ring16. |

The final density contrast retains its original numerical gates:

| Ring16 endpoint | Fresh non-saturating control | Energy-U | Energy-U, frozen prior | Original gate |
| --- | ---: | ---: | ---: | ---: |
| Component covariance error | 1.556546 | 6.371506 | 6.259248 | <= .85 |
| Minimum eigenvalue ratio | .094075 | 1.027582 | .748672 | >= .15 |
| High-quality fraction | .888672 | .747559 | .552246 | >= .85 |

The control fails covariance and minimum spread; both energy recipes fail
covariance and HQ. Both energy recipes fail the full gate on all 96 saved Ring16
observations, with terminal passing suffix zero against five required checks.
Endpoint diagnostics do not replace these sustained gates. Word and Gaussian
smoke passes certify acquisition under their original contracts; Gaussian's
final KS still fails for both energy recipes. Preserve acquisition and endpoint
results separately.

The actual-training illustrations and their provenance sidecars remain attached
to the original public-API numerical results, including the
[fresh control/energy GIFs](https://github.com/255BITS/ParticleGAN/tree/2cf7b0a4c796e29ace6cca12d2e553ad67211b5c/reports/forge/r1r2-energy-distance-repair/publication/media)
and [prior-freeze GIFs](https://github.com/255BITS/ParticleGAN/tree/2cf7b0a4c796e29ace6cca12d2e553ad67211b5c/reports/forge/r1r2-energy-prior-freeze-repair/publication/media).
The pinned publication JSONs retain individual result/certificate hashes, GIF
hashes, source snapshots, checkpoints and original charge owners. Cached parent
cells are reused evidence; their repeated appearance adds no independent trial
or paid cost.

The historical baseline uses executed source
`a0493ca3c213d559523eadc76e244ec9f9c1f81f`. The fresh control/energy comparison
executes `67920d9090727bd74adbd331187f0e6f22533e48`; the prior-freeze treatment
executes `3203c4e419cb7c2a4cd2021d2c48638980579901`. Those last two share
scientific source digest
`2b04f2853c511c9bcd417ef41e8fd773e97e8c67c9e057da8abdd4fef2de5f12`
and their pinned runtime, rather than the later unmeasured MMD source. The archive
head identifies retained files; it does not relabel any executed certificate.

Each original comparison declares one global recipe delta. Protocol seed 0,
public deterministic initialization, task architecture/data, initial prior
distribution/capacity, ordered target batches, sampling, full update budgets,
evaluation cadence and gates retain their recorded identities. The recipe owns
prior learning/freezing and regularization. Direct-coordinate and unsampled token
fixtures retain their explicit `not_sampled` applicability. The final freeze
contrast changes learned prior locations into buffers while preserving G/E/D
updates. Actual initial tensors and consumed streams were audited in its original
receipts. Earlier cross-family baseline results have weaker batch-alignment
evidence and differing word acquisition lengths; they do not establish a fully
matched superiority comparison. No historical fixture is substituted for a new
shared baseline by this archive.

The closed aggregate records 238 unique paid attempts and
7,168.248061007995 seconds under the unchanged 8,000-second cap. The last contrast
adds seven fresh attempts / 197.333891441056 seconds and seven zero-charge cached
parents. These totals overlap earlier publications and must not be added to them.
The remaining 831.751939 seconds cannot reserve the unchanged 900-second word
task. Bulk checkpoints, logs and streams remain in their original local artifact
archives; their availability was not re-audited here.

The [saved-only mechanism review](https://github.com/255BITS/ParticleGAN/blob/2cf7b0a4c796e29ace6cca12d2e553ad67211b5c/reports/forge/r1r2-next-repair-research/README.md)
and [MMD preparation](https://github.com/255BITS/ParticleGAN/blob/2cf7b0a4c796e29ace6cca12d2e553ad67211b5c/reports/forge/r1r2-mmd-repair/README.md)
remain deferred in the immutable archive. MMD has software acceptance and draft
studies, with zero trained task results. Its proposed new-source comparison
requires fourteen fresh cells and a 5,040-second full allowance; the earlier
proposal's 12,209-second aggregate cap was never authorized. No MMD runtime,
candidate, study or paid execution path is imported into develop. Measured energy
results supply no MMD qualification. This week's recommendation is to retain the
stronger measured control and the negative findings, defer optional runtime work,
and close the original mixed PR after this compact replacement is accepted.

To inspect a retained source without executing it:

```sh
git fetch origin archive/pr401-2cf7b0a4c796e29ace6cca12d2e553ad67211b5c
git show 2cf7b0a4c796e29ace6cca12d2e553ad67211b5c:reports/forge/r1r2-energy-prior-freeze-repair/publication/results.json
```

This reporting cleanup performs zero training updates, sampling draws or
qualification recomputation. It preserves the existing generated goal board.
