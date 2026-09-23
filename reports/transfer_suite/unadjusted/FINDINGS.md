# Shared-default search results

**`shared_c6` passes all 19 live behavioral tests with one unchanged recipe and
explicit discriminator choices.** The reference discriminator profile scores
15/19. The final fix gives the discriminator smooth local neighbor-distance
features, making concentration within each generated cluster visible to it.

| Shared recipe | Required | Data | Images | Live total | Reference D profile |
| --- | ---: | ---: | ---: | ---: | ---: |
| `shared_c6`, with declared D choices | 9/9 | 6/6 | 4/4 | **19/19** | 15/19 |
| `lr00425_prior2`, cap coefficient 3 | 8/9 | 3/6 | 4/4 | 15/19 | 15/19 |
| `ratio_g34_d68_p85` | 8/9 | 3/6 | 4/4 | 15/19 | 15/19 |
| Public preset `gan` | 1/9 | 6/6 | 1/4 | 8/19 | 8/19 |
| Pinned old preset `gan_legacy` | 1/9 | 3/6 | 1/4 | 5/19 | 5/19 |

The winning recipe is unchanged from the 18/19 baseline: **Rp logistic, b_cap
coefficient 6, κ1.25, particle spread .05, no particle L2, Adam (0,.99), G/D LR
.00425 and particle LR .0085** everywhere. Rates hold for 60% of the fixed
budget, then cosine toward 5%. No host-specific optimizer or loss adjustments.
This establishes a reproducible shared-recipe baseline; package preset values
have not been changed by this research round.

## What fixed the last case

The unequal-mass test has component probabilities 55%, 30%, 13% and 2%.
[Read-only diagnostics](runs/shared-rare-diagnostic/README.md) corrected the
earlier hypothesis: the previous model's final failing components were the
55% and 13% clusters. The rare 2% cluster had sufficient spread at that point.
Occupancy alone was not the problem; covariance contracted and oscillated.

The winning discriminator has three width-96 hidden layers, per-example feature
centering without variance normalization, and Softplus activation β6. Four
smooth local-distance features join its final score head, for **19,013 trainable
parameters**. For sample `x_i` and scale `s ∈ {.1,.25,.5,1}`:

```text
k_ij(s) = exp(-||x_i - x_j||² / (2s²))
q_i(s)  = Σ[j≠i] k_ij(s)||x_i - x_j||²
          / (s²(Σ[j≠i] k_ij(s) + 1e-5))
score_i = linear(concat(trunk(x_i), q_i(.1), q_i(.25), q_i(.5), q_i(1)))
```

The features use only other samples in the current discriminator batch.
Real and fake batches compute separate features. There are no target labels,
known component centers, metric feedback, running statistics or auxiliary loss.
All feature paths stay attached to autograd.

**The discriminator is batch-dependent.** The unchanged native b_cap computes
the gradient of the sum of batch scores, including cross-sample paths; G also
receives cross-fake gradients. Thus the effective adversarial game and cap
interpretation depend on the batch. This is an explicit architecture change
under the permitted D-variant rules, not a claim of pointwise equivalence.
The four kernels cost O(4B²); the fixed training batch is 128. Kernel widths are
in raw coordinate units, so scale and batch-size transfer remain untested.
Evaluation measures generator samples without invoking this discriminator.

The model passes every live metric at **seven consecutive final observations,
steps 900–1200**. It first accumulates five of those at step 1100.

| Live metric | Worst of final five observations | Required |
| --- | ---: | ---: |
| Minimum normalized component variance | **.42385** | ≥ .15 |
| Component covariance error | .36954 | ≤ .85 |
| High-quality sample fraction | .97119 | ≥ .85 |
| Mixture mass error | .04260 | ≤ .15 |
| Normalized sliced distance | .04979 | ≤ .18 |
| Minimum component mass ratio | .67233 | ≥ .25 |

Its final minimum variance is .63294. **EMA fails the sustained gate** with
only two final passing observations; it contributes no selection credit.
An independent replay matches all ten mandatory numerical fields, including
every live/EMA observation, optimizer receipt and schedule action.
[Five-card screen and failures](runs/shared-batch-feature-search/README.md) ·
[Independent replays](runs/remaining-replays/README.md).

## Architecture support across the suite

| Case | Discriminator | Final passing observations |
| --- | --- | ---: |
| Broad mixture and spiral | Reference discriminators | 20 and 23 |
| Unequal mass | Centered 96×3 Softplus β6 plus batch neighbor distances | **7** |
| Anisotropic | Raw MLP plus small Fourier branch, 64×2, Softplus | 8 |
| Overlap | Raw-input MLP, 96×3, Softplus β5 | 10 |
| Unequal widths | Raw-input MLP, 128×3, Softplus β8 | 5 |

Required and image cases keep their original discriminators. This is one recipe
with declared architecture choices; one universal discriminator has not been
demonstrated. The earlier width fix remains intact: its worst late variance is
.25617, covariance error .62818 and quality .93042.
[Width study](runs/shared-width-search/README.md).

## Search evidence and limits

Three GPT-6 Sol agents at max reasoning, plus parent experiments, added
**89 scored episodes** this round. Replays and analytic controls add no
selection points.

| Study | Episodes | Result |
| --- | ---: | --- |
| Pointwise normalization structure | 34 | No sustained pass; centering β6 supplied the winning trunk |
| Coordinate features, residual paths and readouts | 26 | No sustained pass |
| Spectral normalization and bounded scores | 12 | No sustained pass |
| Global optimizer changes | 4 | No sustained pass; incomplete recipes |
| Four global schedules × two D choices | 8 | Best final passing streak 3; incomplete recipes |
| Current-batch discriminator features | 5 | One sustained pass; stopped at first PASS |

Three later batch-feature cards and six schedule interpolation cards were
declared but never run. No seed sweep, budget extension, checkpoint selection
or gate relaxation was used. The previous 127-episode round remains documented.

The primary importer validates **604 episodes across 37 recipe entries**:
14 complete rows and 23 incomplete screens. It checks canonical test setups,
identical per-row recipes, actual LRs/betas, schedule action multipliers,
D-only changes, source manifests and recomputed live/EMA verdicts.
All failures remain in the readable leaderboard and local raw evidence.

**125 focused tests pass** (121 integration/research tests and four new
batch-feature tests). A fresh complete-profile run passes 19/19 and matches
all ten numerical fields against each selected episode. The previous 18/19
profile also reproduces exactly. Four independent architecture replays pass
their comparison checks, including the earlier rare failure; a replay check
passing does not turn that underlying failed episode into a behavioral PASS.
[Independent winner review](runs/shared-batch-feature-review/README.md).

A [mechanism audit](runs/shared-rare-mechanism-audit/README.md) found no update
sign/detach bug. It demonstrated that the global latent spread loss can be zero
while local clusters collapse, but the archived run cannot separate latent
contraction from generator compression. That is a limitation of the diagnosis,
not proof of a unique cause.

These are inspected development tests, not unseen holdouts or demonstrated
Music/Anima/Supra transfer. Raw curves, hashes, logs and source archives remain
local under the [artifact policy](../../README.md); source, plans and readable
results remain in Git.

## Reproduce the complete baseline

The [explicit profile](leading_profile.json) runs all 19 canonical tests:

```sh
python -u -m benchmarks.transfer_suite.shared_profile_search \
  --plan reports/transfer_suite/unadjusted/leading_profile.json \
  --output /tmp/shared-c6-profile > /tmp/shared-c6-profile.log 2>&1
tail -f /tmp/shared-c6-profile.log
```

Use a new output directory. The command needs no historical generated reports.
It writes all curves, sources, optimizer receipts and a readable result.
[Contributor instructions](../../../benchmarks/transfer_suite/UNADJUSTED_SEARCH.md) ·
[Full-profile verification](runs/rare-profile-replays/README.md) ·
[Primary leaderboard](README.md).
