# Candidate analysis: remove training-only output convolution

Read-only scientific analysis, 2026-09-30. Zero training allowance; no source,
queue, receipt, or compilation changes. This report does not establish a
positive reference or authorize additional cells.

The strongest supported next single-factor diagnostic is
`k3p-no-output-noise-diagnostic`, parent `k3p`, with exactly:

```json
{"recipe_overrides": {"output_noise_std": 0.0}}
```

Keep `requires_capabilities: ["learned_locations", "a2", "named_rng"]`,
`api_version: "forge-api-v1"`, `execution_path: "public_trainer"`, and
`claim_contract: {"schedule": "scheduled", "scoring_weights": "live",
"sampling_law": "task_declared"}`. Ordinary recipe fields require no extension;
there are no API changes or new optimizers.

## Mechanistic question

The current public trainer optimizes generated distributions convolved with
independent Gaussian **output** noise. `GANTrainer._generate` adds
`sigma * torch.randn(...)`; `step` uses it for both the discriminator fake batch
and generator objective. `output_noise_std` reaches `.029` after the first 20%
of the frozen 7k horizon. Input noise is separate and decays to zero by 10%.
Public `sample(..., output_noise=False)` and Forge's current native adapter
omit output noise while retaining learned-MoG kernel noise. Thus the optimized
fake law differs from the assessed clean public law.

The actual native sampler, `benchmarks/toy100/problems.py`, sets `DATA_STD=.03`.
For an ideal isolated Gaussian mode with independent additive output noise,
matching the noisy fake law to that target permits clean per-axis variance
`.03²-.029²=.000059`: standard deviation `.007681146`, or `.0655556` of
target variance. This is a mechanistic inference under ideal distributional
matching, **not a measured prediction**. The baseline clean holdout covariance
bias `-.265594` is much milder; finite optimization, learned affine scaling,
location spread and the evaluator's within-radius conditioning matter.

Disabling the training-only convolution asks whether the adversarial objective
can preserve terminal shape under the unchanged public sampling law. It does
not replace scoring weights, remove latent kernel noise, tune the target width,
or modify any gate. Structural classification is defensible as removal of an
entire stochastic transformation from both training objectives, analogous to
the already structural A2-off mechanism ablation. Declare the exact scalar
endpoint `.029 -> 0` prominently: this is an existing-field objective ablation,
not a newly implemented optimizer or proof of structural superiority. Choosing
another positive amplitude after this outcome would be constant tuning under
a new hypothesis, and is outside this bounded study.

## Saved evidence and compatibility

| Same named affine MoG grid host | Modes | Holdout covariance bias | Holdout radial KS | Full gate |
| --- | ---: | ---: | ---: | --- |
| K3P, output noise .029, A2 .5 | 100 | -.265594 | .120006 | FAIL |
| A2 off, output noise .029 | 74 | -.426600 | .210772 | FAIL |

The current native baseline briefly passed accuracy at 5,750, then failed
every terminal check. Keep A2; its measured removal worsened coverage and
shape and supplies no justification for repetition or a 14k extension.
Only those two attempts exist for `grid100_affine_square_named_v1` in the
durable Forge attempt receipts; both retain output noise `.029`. No exact
current one-factor learned-MoG/no-output-noise repeat was found in this scope.

The [positive-source audit](../POSITIVE_REFERENCE_SOURCE_AUDIT.md) found no
compatible full-reference positive. [PR155's archive](../supplemental/pr155-current-archive-v1/README.md)
has 93 native outcomes whose clean diagnostics all fail, while noisy outcomes
include 82 passes. Its `noout` name means no new data-space statistic, not
zero output noise: the original package retains `.029` and noisy scoring.
The [historical MoG envelope study](../supplemental/local-mog-envelope-v1/README.md)
uses different absolute widths, an EMA envelope rule and older custom loops;
none supplies current live full-native credit.

Historical [noiseless affine F5/F7 controls](../../toy100/affine-noiseless-native.md)
failed strict native gates. F5's holdout center RMS was `.49046σ`, covariance
bias `-.20372` and radial KS `.04502`. These negatives limit confidence:
noise removal alone is not sufficient in general. They differ from this
diagnostic in finite-cloud prior, Fourier bands, input-noise removal,
initializer/seed and optimizer settings, so they do not duplicate its
one-factor MoG question. Earlier [bandwidth screens](../../toy100/bandwidth-constraints.md)
also found no width-free common-host winner; do not turn this diagnostic into
an amplitude, seed, Fourier or width sweep.

Zero-training checks imported the current public API, constructed both
formulation contexts and ran native adapter preflight. Preflight returned no
blockers; the resolved recipes differ **only** in `output_noise_std: .029 -> 0`.
A2 remains requested. Preserve seed 0, named streams, identity affine G,
Fourier3/Xavier D, uniform `[-5,5]` learned locations, fixed latent sigma `.025`,
uniform masses, no standardization and the 7k schedule. Removing the output
draws intentionally leaves the named output-noise stream unadvanced; other
training streams must remain isolated. Initial tensor hashes and step-zero
arrays should match the original control. Final output-stream equality is
neither expected nor an independent requirement; record the intended change.

## Bounded decision

Measure only the full 7k `grid100_affine_square_named_v1` cell first, in the
calibration diagnostic namespace. Reuse the certified original K3P control
through explicit verified diagnostic-import bindings. The task's existing
3,600-second timeout requires a complete 3,600-second reservation; baseline
92.037 seconds is an estimate of expected work, not a timeout waiver. Keep
all five live 20k terminal checks and independent 100k live holdout. Compare
coverage, center RMS, signed/absolute covariance bias, radial KS, mass TV,
updates, memory and wall time; report EMA separately. Stop after a full FAIL.

A pass would resolve this native diagnostic obstacle only and warrant a
new bounded next-stage decision. It cannot establish the 16-task reference
positive, accepted calibration, clock-free eligibility, robustness or public
default promotion. Add this fourth lineage to the new versioned profile while
preserving the three smoke tasks, all 16 reference purposes and criteria.
Unknown cells remain unknown, and original three-lineage failures remain
searchable. No automatic whole-matrix or failed-parent continuation follows.
