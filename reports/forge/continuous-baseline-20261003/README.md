# Frozen Atlas19 and C6 persistence evidence

Atlas19: 19/19 full protocols and 19/19 original goal GIFs.

This publication keeps the original noisy serving cohort separate from clean diagnostics and the C6 persistence extension. It supplies no current MoG qualification, family winner, shipped default or speed ranking. Missing or blocked evidence is unassessed, never a numerical failure.

| Original question | Updates | Execution | Scientific | Goal GIF |
|---|---:|---|---|---|
| `portability/img_intensity2` | 600 | PASS | PASS | [actual observations](media/atlas-original19-portability-img_intensity2.gif) |
| `portability/mode_hold` | 1200 | PASS | PASS | [actual observations](media/atlas-original19-portability-mode_hold.gif) |
| `portability/img_blobs4` | 600 | PASS | PASS | [actual observations](media/atlas-original19-portability-img_blobs4.gif) |
| `portability/img_bars4` | 600 | PASS | PASS | [actual observations](media/atlas-original19-portability-img_bars4.gif) |
| `portability/img_stripes2` | 600 | PASS | PASS | [actual observations](media/atlas-original19-portability-img_stripes2.gif) |
| `portability/vector_two_broad` | 1200 | PASS | PASS | [actual observations](media/atlas-original19-portability-vector_two_broad.gif) |
| `portability/vector_unequal_mass` | 1200 | PASS | PASS | [actual observations](media/atlas-original19-portability-vector_unequal_mass.gif) |
| `portability/vector_unequal_width` | 1200 | PASS | PASS | [actual observations](media/atlas-original19-portability-vector_unequal_width.gif) |
| `portability/vector_anisotropic` | 1200 | PASS | PASS | [actual observations](media/atlas-original19-portability-vector_anisotropic.gif) |
| `portability/vector_overlap` | 1200 | PASS | PASS | [actual observations](media/atlas-original19-portability-vector_overlap.gif) |
| `portability/vector_spiral` | 1600 | PASS | PASS | [actual observations](media/atlas-original19-portability-vector_spiral.gif) |
| `portability/stationary` | 7500 | PASS | PASS | [actual observations](media/atlas-original19-portability-stationary.gif) |
| `portability/ring_shift` | 4600 | PASS | PASS | [actual observations](media/atlas-original19-portability-ring_shift.gif) |
| `moving/grid100` | 1500 | PASS | PASS | [actual observations](media/atlas-original19-moving-grid100.gif) |
| `moving/rotated100` | 1500 | PASS | PASS | [actual observations](media/atlas-original19-moving-rotated100.gif) |
| `moving/staggered100` | 1500 | PASS | PASS | [actual observations](media/atlas-original19-moving-staggered100.gif) |
| `native/grid100` | 7000 | PASS | PASS | [actual observations](media/atlas-original19-native-grid100.gif) |
| `native/rotated100` | 7000 | PASS | PASS | [actual observations](media/atlas-original19-native-rotated100.gif) |
| `native/staggered100` | 7000 | PASS | PASS | [actual observations](media/atlas-original19-native-staggered100.gif) |

Each question, full host/seed/budget, exact observation cadence and unchanged numerical requirements are retained in `results.json`. Native scientific PASS requires both original noisy coverage and Gaussian accuracy, including the final five 20k checks and the independent 100k holdout. Clean and EMA results remain diagnostic.

| C6 continuation | Original 1200 gate | Original study | New hold gate | Added updates | Goal GIF |
|---|---|---|---|---:|---|
| atlas | PASS | INCOMPLETE | FAIL | 150 | [actual states](media/c6-atlas-broad-hold-1200-to-1350.gif) |
| e22 | PASS | INCOMPLETE | FAIL | 150 | [actual states](media/c6-e22-broad-hold-1200-to-1350.gif) |

The named hold variant restores each complete checkpoint from source `8021a1c5` and adds only 150 updates. It checks 1250/1300/1350 with the original scorer and retains the original 1200 PASS and study INCOMPLETE. Its compound requirement is five post-confirmation passing checks; it grants no ordinary eight-case credit.

Paid attempt cost: 5248.642153 seconds; unmeasured interrupt reservations: 0.000000 seconds. The two preserved H1 startup INCOMPLETE attempts contribute 6.818678 paid seconds exactly once. Summed paid cost is not elapsed study time or a cross-hardware speed comparison.

This directory is shareable with its copied GIFs and hash-bound JSON. Independent rechecking additionally needs the listed raw archives, durable supervisor receipts and exact frozen source snapshots; bulk traces and checkpoints were not copied.

Supplemental retained-cloud views (original grades unchanged):

- `atlas-original19-moving-grid100`: [actual retained clouds](media/atlas-original19-moving-grid100-retained-clouds.gif).
- `atlas-original19-moving-rotated100`: [actual retained clouds](media/atlas-original19-moving-rotated100-retained-clouds.gif).
- `atlas-original19-moving-staggered100`: [actual retained clouds](media/atlas-original19-moving-staggered100-retained-clouds.gif).
- `atlas-original19-native-grid100`: [actual retained clouds](media/atlas-original19-native-grid100-retained-clouds.gif).
- `atlas-original19-native-rotated100`: [actual retained clouds](media/atlas-original19-native-rotated100-retained-clouds.gif).
- `atlas-original19-native-staggered100`: [actual retained clouds](media/atlas-original19-native-staggered100-retained-clouds.gif).

## What the original questions verify

| Check | Question |
| --- | --- |
| `portability/img_intensity2` | Can the 8×8 image host reproduce the same centered 4×4 patch at gray levels 0.35 and 0.85? Tests intensity fidelity and high-quality coverage of both brightness modes. |
| `portability/mode_hold` | Can a host with only 12 latent rows cover all eight Gaussian modes on a radius-3 ring and finish with sustained high-quality output? Tests scarce latent support with the original coverage/HQ suffix. |
| `portability/img_blobs4` | Can the 8×8 image host place a 2×2 patch at each of four corner locations? Tests localized spatial fidelity and high-quality coverage of every location. |
| `portability/img_bars4` | Can the 8×8 image host generate two vertical and two horizontal bars at their distinct offsets? Tests both spatial position and orientation, with high-quality examples of all four templates. |
| `portability/img_stripes2` | Can the 8×8 image host generate both a centered horizontal stripe and a centered vertical stripe? Tests orientation coverage at a fixed center and width. |
| `portability/vector_two_broad` | Can two separated, equally weighted Gaussians at x = ±1 retain their common standard deviation 0.25? Tests a basic multimodal distribution through occupancy, distribution distance and within-component spread. |
| `portability/vector_unequal_mass` | Can four equal-width Gaussian components reproduce target probabilities 55%, 30%, 13% and 2%? Tests unequal probability allocation, including a minimum relative occupancy bound for the rare component. |
| `portability/vector_unequal_width` | Can four equally weighted Gaussian components retain distinct standard deviations 0.07, 0.12, 0.20 and 0.30? Tests component-specific scales through covariance error and the weakest normalized covariance axis. |
| `portability/vector_anisotropic` | Can three equally weighted Gaussian components reproduce differently oriented elliptical covariance shapes? Tests correlated, unequal axes; the minimum normalized eigenvalue checks for collapse along a narrow direction. |
| `portability/vector_overlap` | Can two broad Gaussians centered at x = ±0.35 with standard deviation 0.55 match the observable overlapping mixture? Sliced distance, mean and covariance are scored; hidden component labels and mode recall are unassessed. |
| `portability/vector_spiral` | Can a finite particle model follow a noisy 1.5-turn spiral from radius 0.3 to 2.0? Sliced distance and moment gates test continuous curved mass; they do not prove exact density or topology. |
| `portability/stationary` | Can the larger 20,000-row ring host reach all eight unchanged modes and finish 7,500 updates with sustained quality? Tests long stationary training using the original final passing suffix. |
| `portability/ring_shift` | Can the ring host reacquire all eight modes after the target translates one unit in x after update 2400? Tests recovery with a passing suffix in both the original and shifted segments. |
| `moving/grid100` | Can an initially axis-aligned 10×10 Gaussian grid follow two 30° target turns? Tests adaptation away from aligned geometry, requiring at least 95 modes and at least 90% of observed pre-turn HQ after each turn. |
| `moving/rotated100` | Can the 100-mode grid, initially rotated 25°, follow two further 30° turns? Tests moving coverage from an oblique starting layout, with the same 95-mode and relative-HQ requirements. |
| `moving/staggered100` | Can a 100-mode lattice with compressed row spacing and a half-cell offset between adjacent rows follow two 30° turns? Tests moving coverage on a staggered layout, with the same 95-mode and relative-HQ requirements. |
| `native/grid100` | Can the noisy primary sampler recover all 100 equally weighted Gaussian components on an axis-aligned 10×10 grid, including their mass and width? Tests large-mode coverage and Gaussian fidelity through the final-five 20k checks and independent 100k holdout. |
| `native/rotated100` | Can the same 100 equally weighted Gaussian components be learned on a grid fixed 25° from the coordinate axes? Tests orientation of the layout with unchanged joint coverage, Gaussian fidelity and independent 100k accuracy requirements. |
| `native/staggered100` | Can all 100 equally weighted Gaussian components retain correct mass and width on a lattice with 0.85 row spacing and a half-cell offset between adjacent rows? Tests staggered geometry with the same joint coverage and independent 100k fidelity requirements. |

The per-row `definition.original_requirements` in `results.json` are the authoritative gates; generic “mass” wording in its question descriptions is descriptive. The four image checks gate complete mode count and HQ ≥ 0.9 over the required passing suffix; mode count uses the original template-RMSE and minimum quality-occupancy cuts, while total variation is diagnostic. Overlap and spiral gate normalized sliced W1, mean error and covariance error, with no explicit component-mass gate.

These are the original historical hosts and gates. The [current ordinary leaderboard](../technique-inventory.md), [policy leaderboard](../policy-family-inventory.md), and [definition-quality audit](../../toy_audit/README.md) retain their own exact cohorts, scores and unknown requirements.

Review [measured persistence failures](HOLD_RESULTS.md), [independent goal-media QA](GOAL_MEDIA_QA.md), and [current open toy PR dispositions](REMOTE_PR_CHECKPOINT.md). The verified JSON and copied GIF bytes are unchanged from the offline publication; this README additionally explains its questions.

The [raw evidence archive](ARCHIVE.md) includes all frozen source, parent and continued checkpoints, draw arrays and terminal supervisor receipts. Its [verification receipt](archive-verification.json) confirms 11,572 individually checked members. Bulk evidence is local; the compact scores and GIFs in this directory are shareable through Git.
