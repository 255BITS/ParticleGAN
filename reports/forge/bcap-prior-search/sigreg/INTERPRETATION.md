# SIGReg: no Tier 2 repair in the declared bracket

The matched full BCAP leader passed all six Tier 1 gates and 9/21 Tier 2 gates.
None of the three global SIGReg coefficients preserved all six Tier 1 gates,
so the ordinary protocol blocked every candidate before Tier 2. This search
provides no evidence that SIGReg repairs the remaining Tier 2 failures, including
`grid100`, `rotated100`, and `staggered100`: all three native tasks are unmeasured
for each SIGReg strength. Stop this finite bracket; do not advance a candidate,
adopt a default, or infer a repair from its separate prior-training demo.

The source-bound comparison below orders coefficients numerically. Labels come
from each admitted request's recipe overrides and the published trial settings;
request order is hash-based and supplies no strength meaning.

| Prior coefficient | Tier 1 | Gaussian smoke confirmed states (target ≥1) | Ring terminal passing suffix (target ≥5) | Ring covariance error (target ≤0.85) | Tier 2 |
| --- | --- | ---: | ---: | ---: | --- |
| Control, 0 | 6/6 PASS | 2 PASS | 26 PASS | 0.496567 PASS | 9/21 PASS, all measured |
| SIGReg 0.0001 | 4/6 PASS | 0 FAIL | 0 FAIL | 2.128039 FAIL | BLOCKED, 0/21 measured |
| SIGReg 0.001 | 5/6 PASS | 4 PASS | 0 FAIL | 0.884537 FAIL | BLOCKED, 0/21 measured |
| SIGReg 0.01 | 4/6 PASS | 0 FAIL | 0 FAIL | 1.806330 FAIL | BLOCKED, 0/21 measured |

Ring acquisition is the common regression. All coefficients reached all 16
modes, met mass-TV ≤0.15, high-quality fraction ≥0.85, and minimum component
eigenvalue ratio ≥0.15 at the recorded endpoint. Their component covariance
error exceeded 0.85. At 0.001 the endpoint miss is small (0.034537), but the
terminal passing suffix is zero; endpoint proximity does not satisfy the five
required passing checks. At 0.0001 and 0.01 covariance error is substantially
worse. This separates the observed density-shape failure from gross mode loss.

Gaussian smoke at 0.0001 and 0.01 had no independently confirmed scheduled
passing state. Endpoint KS errors were 0.066589 and 0.234117 against ≤0.05;
0.01 also failed mean error (0.498709 sigma against ≤0.2). At 0.001 four
scheduled states confirmed a full pass, even though its final KS was 0.078031.
The control likewise passes by its declared earlier confirmed states despite
final KS 0.071842. The report retains both acquisition verdicts and endpoint
measurements rather than substituting the final endpoint for the task's gate.
The other four required Tier 1 tasks passed at every strength.

The [complete score/target tables](README.md) and [machine-readable results](results.json)
include diagnostic component arrays and unavailable values. The matched
control's 12 Tier 2 failures remain baseline evidence. SIGReg's 63 unreached
Tier 2 cells remain unmeasured: empty repair/regression lists do not establish
that the control's nine passes were preserved.

## Protocol and provenance

Executed scientific source is `a2a9183f042586968fe93daad1db5984c4621ddf`, digest
`af27fe8e557defb5d97b82e8b4b3b6f48f70f22eef863da2921734e45e2adf37`.
Later report commits do not replace that execution identity. The
[specification](spec.json) and [registration](registration.json) freeze the
leader clone, protocol seed 0, public deterministic initializer, three global
coefficients, 32 projections/17 knots, ordinary revision-9 view, task conditions
and budgets. The only trainer deltas are `prior_regularizer="sigreg"` and
`prior_reg`; task architecture, initial prior distribution/capacity, sampling,
data law, update ceilings, cadence and gates stay fixed.

The saved-only [audit](protocol-audit.json), reproduced by
[protocol_audit.py](protocol_audit.py), validates the original four admitted
requests, certifications and source bytes. Eighteen candidate/control Tier 1
initialization pairs are exact, and the three 1,000-step Gaussian data digest
pairs are exact. All four ring acquisitions consumed 1,600 updates under the
same declared adaptive stopping rule. Word smoke used each candidate's own
20,001-update acquisition prefix; there is no SIGReg word-hold credit. Optional
clock diagnostics retain their own smaller probe recipe and supply no claim
that SIGReg's feature state is invariant.

The [media index and receipts](media/README.md) cover all 49 certified measured
PASS/FAIL tasks using saved real training observations. Publication adds zero
optimizer updates, sampling draws, and qualification rescoring. Original raw
logs, observation streams, tensors and checkpoints remain in
`/mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/sigreg`. GPU execution
used the common shared MPS pool; the parent archive's `infra-repair.json` retains
the unchanged-source control word continuation after GPU-context contention.
Runtime is accounting evidence, not a speed ranking: control spent
2,695.542382 seconds and all three trials spent 2,660.934959 seconds, with zero
remaining reservation, within their 45,000/135,000-second ceilings.

The fresh baseline comparison in the [same audit](protocol-audit.json) binds
the archived publication hash/source and the fresh result hash/original receipts.
Archived metric values and original grades match 28/28 retained task rows.
Full dictionaries match 19/28 because nine archived rows omitted component
arrays; no historical qualification was rewritten. Original control source and
sampling cohorts remain separate from unmeasured SIGReg native tasks.

## Publication checks

The feature's 177 focused tests, 468 host/API/search regression tests, CPU/CUDA
resume checks, disabled baseline byte parity, and bounded numerical public-API
demo remain documented in the [feature report](../../prior-sigreg/README.md).
The common saved-only publisher validation passed 36 checks and 12 fixture
GIFs before this final export. Protocol audit and the final 49-GIF export pass;
Forge memory was refreshed with `compile --summaries-only` and its read-only
`compile --check` is CURRENT, preserving historical qualification tables.
All 20 publication-memory regression tests passed, including complete trial
discoverability and committed-memory freshness without raw-log hydration.
these are software/publication checks and grant no extra research qualification.

To reproduce publication from the original archive without running research:

```sh
PYTHONPATH=. python reports/forge/bcap-prior-search/workflow.py publish \
  --name sigreg --archive /mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/sigreg
PYTHONPATH=. python reports/forge/bcap-prior-search/export_saved_media.py \
  --name sigreg --archive /mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/sigreg
PYTHONPATH=. python reports/forge/bcap-prior-search/sigreg/protocol_audit.py \
  --archive /mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/sigreg
```

The generated measurement README's compact interpretation links and explicit
BLOCKED header are report-only additions after saved-only generation.

Saved-only reproduction commands require the [sigreg feature's executed checkout](https://github.com/255BITS/ParticleGAN/tree/a2a9183f042586968fe93daad1db5984c4621ddf) and the original local archive. This aggregate branch contains reports and saved evidence; the feature PR contains its implementation.
