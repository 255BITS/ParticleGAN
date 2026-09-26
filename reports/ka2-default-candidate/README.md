# KA2 default candidate: evidence and release assessment

**KA2 is a candidate for the next default, not a qualified winner.** It is the
most promising historical challenger for combining an intact pre-shift hold
with eventual recovery. The completed extension and toy results still expose
instability. This draft makes the proposed package change reviewable; these
historical measurements do not certify the new package implementation.

The initial assessment used existing artifacts on September 26, 2026. The seed
results below already existed. A subsequent [matched implementation replay](matched-replay/README.md)
holds the research ring experiment fixed and compares the frozen mechanisms
with the public recipe factories. It introduces no seed variants. Its scope
is separate from the historical 22-task qualification below.

## Recorded results

| Configuration and scope | Hold | Recovery | Toy qualification |
|---|---|---|---|
| Exact historical KA2 | Pre-shift 120/120 checks; minimum HQ .91943 | 50/81 deadline checks; final 8 modes, HQ .99561 at update 3600 | Initial focused screens 3 PASS / 1 FAIL; later full-suite attempt stopped at unequal mass; no native gates |
| KA2, same source, extended to 4600 | Same 120/120 pre-shift checks | **105/109 checks pass from the previously reported arrival at 3520**; four new failures; final 8 modes, HQ .99927 | No additional toys in this run |
| [Historical R2](historical-reports/r2-full-suite-attempt.md) | Pre-shift 114/120 checks | 72/81 deadline checks; earlier final passing suffix starts at 2890 | Later exact-source sweep 19 transfer PASS; grid100 FAIL; remaining two natives not run |
| [Selected historical K3P](historical-reports/selected-k3p.md) | Own hold 1200/1200 + extension 300/300 | 28/81 deadline checks, FAIL | Declared 22/22 under its saved research configuration |
| [Public package 0.8.0 K3P baseline](historical-reports/public-package-baseline.md) | Own hold 1200/1200 + extension 300/300 | 0/81 deadline checks, FAIL; pre-shift 120/120 | Full suite NOT_RUN for this package baseline |

The research K3P and public-package rows are context, not a controlled direct
comparison with the draft KA2 package implementation. Historical K3P's declared
22/22 does not establish seed robustness. KA2's 120 pre-shift observations are
spaced ten updates apart; they are **not** the independent own-state hold gate
with 1200 dense passing checks and a 300-check extension. A descendant's hold
result cannot be assigned to the exact KA2 source.

KA2's matched frozen control has the same 120/120 pre-shift result and then
0/81 recovery checks with zero final modes. Its optimizer counters remain at
2400 after the target shift, while live KA2 reaches 3600. This supports active
adaptation in the historical live run, without establishing sustained recovery.
The evidence retains initialization proofs, counters, schedules and complete
diagnostic sequences.

## What the later extension changes

The older leaderboard said KA2 stayed once it arrived at update 3520 and listed
the longer run as pending. The completed run contradicts that stability claim:

| Update | Modes | HQ | Failure |
|---:|---:|---:|---|
| 4280 | 8 | .897705 | HQ below .90 |
| 4300 | 7 | .914063 | Missing mode |
| 4310 | 8 | .893066 | HQ below .90 |
| 4320 | 8 | .893311 | HQ below .90 |

It finishes with 28 passing observations starting at 4330. The original run
also passed from 3070 through 3470 and then failed at 3480–3510. Reporting only
the last passing suffix as an arrival time hides those earlier departures.
The extension therefore demonstrates **105/109**, or 96.33%, passing checks
from 3520 through 4600, rather than uninterrupted stability. Its full recovery
window remains FAIL (146/181 deadline-window checks in the longer driver).

The extended source differs from the original driver only in `steps=3600` →
`steps=4600`; mechanism, config, latent and response files are byte-identical.
Every diagnostic record through 3600 and the initial parameter proof exactly
match the original. See [the driver diff](source/extended-shift.diff) and
[the complete extension diagnostics](evidence/extended-shift.json).

## Unequal-mass failure and existing seed retests

The original full-suite attempt passes `mode_hold`, then fails
`vector_unequal_mass` and stops. At update 1000 the mean component covariance
error is **1.05889 against a .85 maximum**; the fourth component's error is
3.78498. All final live metrics pass, but only four consecutive terminal
observations pass, below the required five. This is a real stability failure.
The original failure is not a final minimum-eigenvalue-ratio failure.

| Existing host-seed offset | Verdict | Passing observations | Terminal suffix | Final covariance error (max .85) | Final minimum eigenvalue ratio (min .15) |
|---:|---|---:|---:|---:|---:|
| 0, original fixture | FAIL | 18/24 | 4 | .192825 | .592182 |
| 1 | PASS | 15/24 | 8 | .561841 | .766211 |
| 2 | FAIL | 8/24 | 0 | 1.058357 | .435548 |
| 3 | PASS | 9/24 | 8 | .357531 | .752218 |

The pre-existing retest round classified any passing offset as **FRAGILE**,
so these results lift the provisional mass disqualification under that rule.
They show 2/3 additional offsets passing, or **2/4 distinct offsets including
the original**. The reproduction of offset 0 is not another independent seed.
This does not fill the missing full-suite or native qualification.

The transfer host hardcodes construction seeds 0/1/2. The saved retest shim
remaps those to S/S+1/S+2, leaving evaluation seeds 990/991/402 unchanged.
It supplies a separately captured CPU initialization fixture for each offset.
The mechanism/config/latent/response remain exact copies. Offset 0 reproduces
the original verdict, initialization parameter proof and randomness audit.
The unequal-mass fixture hash starts `d5d6a1b3`; `cb5ddaeb` is the ring fixture.
See [the retained retest report](historical-reports/mass-retest.md) for this
historical protocol; its narrative is preserved verbatim, including caveats.

The full-suite attempt formally counted 1 PASS / 1 FAIL / 20 NOT_RUN. An
already-started unequal-width run passed after the stop and was labeled
diagnostic only. The earlier initial KA2 screen independently recorded
mode_hold, unequal_width and stripes as PASS. None of these scores is 22/22.

## Why propose KA2, and what still blocks release

### The 22/22 result is not a failed KA2 API replication

The recorded 22/22 result belongs to **K3P**, not KA2. The new KA2 API
implementation has not run the full quality suite. That is missing evidence,
not an observed failure to reproduce 22/22.

Even the K3P research/public ring comparison changes the experiment:

| Ring setup | Research driver | Public 0.8.0 baseline |
|---|---:|---:|
| Batch size | 128 | 2048 |
| Particle count | 12 | 20000 |
| Latent dimensions | 4 | 2 |
| Input noise reaches zero, shift run | Update 120 | Update 360 |
| Output noise reaches full strength, shift run | Update 240 | Update 720 |
| Generator's real batch | Fresh batch | Reuses critic's batch |

The public baseline also constructs new starting weights instead of loading
the research fixture, whose parameter shapes differ. These differences
prevent attributing the 28/81 versus 0/81 recovery result to the API itself;
they do not identify which change caused the difference or rule out a porting
bug. See the [recorded public baseline](historical-reports/public-package-baseline.md)
and the original drivers linked from the archived source files.

The [matched replay](matched-replay/README.md) holds the model, initial weights,
data and random streams, schedules, update order and evaluator fixed while
changing the optimizer and penalty implementation. It compares intermediate
losses, gradients, optimizer state and parameter updates before final scores.
This isolates the public recipe components; it does not replace the research
host with `GANTrainer` or repeat the complete 22-task suite.

### Selection rationale

The research rationale is specific: compared with R2, KA2 trades faster
relearning for an intact pre-shift hold. A gradual response to persistent
critic surprise changes the critic memory update rate, while a hysteretic
gate releases the anchor penalty. KA2 has a fixed 0.5 blend after its initial
acquisition phase, slow attack (1/60), fast release (0.5), surprise thresholds
3.0/1.75, and a guarded memory reseed after sustained release. The gate itself
does not read the training budget, LR floor or shift position. **The measured
learner still retains learning-rate and noise schedules.** Calling it
horizon-independent or schedule-free would overstate the evidence.

Measured research cost for a 3600-call shift is tied with R2 and the other
same-architecture challengers: 799 pure calls plus 2801 blended calls, or
6401 gradient-evaluation units under the historical accounting. Shared-GPU
wall-clock differences do not establish a speed advantage. Package performance
has not been established by these archived runs.

Release blockers and missing evidence:

- Recurring post-recovery failures, including four newly visible in the 4600
  extension, contradict uninterrupted stability.
- Unequal-mass seed sensitivity remains; the original declared fixture fails.
- The exact KA2 formulation lacks a completed full 22-toy qualification,
  including all three native coverage **and** accuracy gates.
- Independent own-state hold1200 + extension300, delayed/repeated target
  changes and long-term qualification are missing for the exact source.
- The package port needs evidence at its real public entry point; source
  parity and unit checks alone do not establish package-level quality.
- Schedule independence and broader robustness remain unproven. This report
  neither launches nor requests new seed experiments.

**Recommendation:** keep the proposal as an unmerged draft candidate. KA2 is a
reasonable formulation to make concrete for review, but the available data
does not support declaring it a proven replacement for K3P. Choosing it now
would be an explicit acceptance of incomplete qualification and measured
instability, rather than the result of a fully passed winner gate.

## Provenance and offline audit

[manifest.json](manifest.json) pins original paths, full SHA256 digests,
retained/omitted JSON fields and source-identity checks.
[summary.json](summary.json) contains the compact machine-readable assessment.
The exact historical mechanism is
[source/mechanism.py](source/mechanism.py), SHA256
`9f1d5eda4bb2f9e0e51832af77db27bd87fba214e29146b224d9963da390c478`.
Its companion config/latent/response hashes begin `a1475108`, `197df635` and
`7e71d60a`. These are archival monkeypatch implementations, not package imports.

Evidence files retain complete relevant JSON subtrees without rounding:
all ring diagnostic records, all mass observations, verdicts, final metrics,
configuration, fixture/worker hashes, initial parameter proofs and randomness
audits. Receipts and unrelated output fields omitted for size are enumerated
in the manifest; original complete-file hashes identify the full records.
The copied historical reports are unchanged and may contain statements
superseded by this assessment.

Run the standard-library-only audit without training:

```bash
python reports/ka2-default-candidate/verify_evidence.py
```
