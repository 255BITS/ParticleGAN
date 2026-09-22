# Controller transfer: test importance

A poor architecture or deliberately ambiguous dataset can expose a limitation
without making that limitation a requirement for choosing a default. This suite
separates **importance**, **observed solvability**, and **measured performance**.
Those are three different columns, not one PASS stamp.

| Declared importance | Selection effect | Examples |
| --- | --- | --- |
| Required | Every live behavioral test must sustain success for eligibility | The nine established behavioral regressions, including all eight ring modes |
| Ranking | Improves or lowers rank; failure does not veto eligibility | Unequal mode mass, ordinary batch/capacity changes, healthy small convolutional GANs |
| Diagnostic | No effect on eligibility, rank, or tie breaking | A generator unable to represent stripes, a critic restricted to image mean, deliberately inadequate resolution |

Tiers and their reasons are written before candidate fitting. A failure cannot
demote a test, and a surprising diagnostic success cannot promote it in the same
study. Promotion to a required test would need an explicit future protocol
revision, a supported-use rationale, and reference-solvability evidence. New
tests do not become required just because they exist.

Reference status is **demonstrated**, **not demonstrated**, or **unmeasured**.
It records whether a particular reference setup sustained the frozen target
within its budget. Failure of two references does not prove an impossible task.
Likewise a solved task is not automatically important. The predictive value of
each test for real-world transfer is currently **unmeasured**; tiers represent
declared product/research relevance, not a statistical importance estimate.

## Cases and splits

There are nine existing required regressions and 24 new development cases:

- [Eight data-distribution cases](vector_tasks.md): six ranking and two diagnostics.
- [Eight training-dynamics cases](stress_tasks.md): six ranking and two diagnostics.
- [Eight procedural-image cases](image_tasks.md): four ranking and four diagnostics.

The existing regressions and 16 vector/dynamics cases are available for fitting.
The two best eligible nonzero fitting policies and cosine advance to all eight
image cases for developmental validation. These images are not held-out transfer
evidence. Three further named families—annulus data, a changed discriminator
update cadence, and a residual upsampling image architecture—are reserved until
the selected policy is frozen. None of their scores can select or refit it.

Every GAN uses seed 0. The separate proposal RNG changes controller coefficients,
not GAN initialization. This study does not estimate across-seed reliability.

## Ranking

Eligibility requires all nine required tests to pass using live weights, for at
least the final five observations of a complete 24-observation curve. The ring
requires eight modes and HQ≥90%. The historical 29-bound report is unchanged;
the sustained ring criterion remains stricter than its old seven-mode bound.

Selection also requires that all required and ranking cases were attempted.
Recorded training errors count as failures; missing attempts cannot improve a
score. Diagnostic failures or missing diagnostic results never affect selection.

Eligible candidates are ordered by:

1. Sustained pass fraction, averaged equally across the data, dynamics and image
   domains and across ranking families within each domain.
2. Worst ranking-family pass fraction.
3. Lower final metric shortfall, using the same domain/family balancing.
4. Lower confirmation-step/budget, using the same balancing and 2 for failure.

For fitting, only the two available vector/dynamics domains enter that score.
For final developmental selection, all three domains enter equally. This keeps
many similar 2D cases from outvoting the image domain. Diagnostic scores, EMA,
and single-run timings cannot break a tie.

Shortfall is the mean positive violation of each bound divided by its absolute
value (scale 1 for zero bounds), capped at 2 per metric. Invalid/incomplete
results receive 2. It provides a visible, subordinate comparison among failures;
it cannot erase a lost sustained pass. All individual bounds and curves remain
available, so different kinds of failure are not concealed by the aggregate.

## Bounded first fit

The search retains the existing five generic gradient features and separate G/D
LR actions. Regularization weights stay zero; Adam and cosine stay in place.
It compares the fixed cosine control, the previously learned equation, and two
generations of six coefficient proposals, reusing exact duplicate policies
explicitly. Initial coefficient standard deviation is 0.008, with proposal RNG
2731. All nine required cases run first; a required failure screens expensive
new cases and leaves that row ineligible and visibly incomplete.

The full-development winner and best eligible nonzero challenger are frozen
separately: a research challenger is not automatically a new default. The
challenger's bias-only ablation is evaluated after selection. It preserves its
constant offsets and cosine while suppressing gradient feedback. Then cosine,
the frozen challenger and that ablation each run the three reserved families
once. If no nonzero challenger is eligible, the reserved families remain unseen.

```bash
python -u -m benchmarks.transfer_suite.suite \
  --output /tmp/particlegan-transfer-suite \
  > /tmp/particlegan-transfer-suite.log 2>&1
tail -f /tmp/particlegan-transfer-suite.log
```

The runner saves its manifest and exact source archive before fitting. It writes
every episode incrementally, with complete live/EMA curves, action traces,
errors, runtime, source hashes and original-byte SHA256s. It refuses to overwrite
a study. Frozen records bind the task declaration, source and development
results. Source changes during fitting are rejected before transfer evaluation.

The procedural-image tests measure a finite learned prior against 8×8 templates.
Passing them does not establish natural-image quality or large-network transfer.
Update counts and wall-clock times are reported separately; a single timing is
not evidence of a dependable speedup. Production defaults are unchanged.
