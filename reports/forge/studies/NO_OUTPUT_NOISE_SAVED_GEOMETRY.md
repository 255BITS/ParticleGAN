# Saved geometry after the output-noise removal failure

Both original native verdicts remain **FAIL**. This zero-training diagnosis
uses the complete final checkpoints and saved final/holdout live arrays; it
creates no sampling view, gate, random draw, image or new candidate. The
[machine-readable measurements](NO_OUTPUT_NOISE_SAVED_GEOMETRY.json) bind
the receipts and artifacts, verify both complete manifests, retain per-mode
moments and record an unchanged PyTorch CPU RNG state.

| Final live geometry | K3P control | Output noise off |
| --- | ---: | ---: |
| Affine G singular values | .989147, .979085 | .991772, .983514 |
| Analytic kernel eigenvalues / target variance | .665699, .679452 | .671736, .683064 |
| Analytic kernel mean axis variance / target variance | .672576 | .677400 |
| Saved 100k HQ mean axis variance / target variance | .697281 | .794683 |
| Saved 100k covariance bias under existing gate | -.265594 | -.163005 |
| Saved 100k radial KS | .120006 | .067130 |
| Saved 100k center RMS / target sigma | .089885 | .188341 |
| Component-mean central location variance / target variance | .030312 | .152565 |
| Component-mean central location center RMS / target sigma | .083396 | .184568 |

The frozen target sigma is `.03`; saved latent sigma is the float32
representation of `.025`. The affine generator is `Y = W(z_i + epsilon) + b`,
so its exact independent Gaussian kernel covariance is `sigma² W Wᵀ`.
Removing training output noise barely changed the global affine scale and
raised analytic kernel variance by only `.004824` of target variance. The
kernel contribution remains about two thirds of the target variance in
both states. EMA singular values and kernel covariances are similarly close;
their separate measurements are in the JSON and confer no substitution credit.

The observed spread improvement accompanies wider, unevenly distributed
learned-location spread. For this diagnostic only, transform every learned
location through its saved affine G, assign its mean to the nearest evaluator
center, and separately summarize the means within that center's 3-sigma disk.
Under this **component-mean** selection, mean location variance grows from
`.030312` to `.152565` of target variance. In the new state the lowest-variance
50 modes contain only **9.72%** of summed per-mode location variance, while
the highest-variance 10 contain **35.28%**; control values are **47.12%** and
**11.41%**. New location variance has correlation **.975913** with saved
holdout HQ covariance across the 100 modes. This describes geometry, not a
causal estimate or independent experiment.

Most modes remain narrow: **76/100** new holdout modes, versus **100/100**
control modes, have individual trace ratios below `.90` of the target's
radius-conditioned variance. This is a descriptive use of the existing
global accuracy limit, not a new per-mode qualification gate. New location
variance has median `.057891` versus mean `.152565`; its upper tail raises
the average without uniformly broadening the mixture. New component-mean
center RMS `.184568σ` closely tracks the saved-sample `.188341σ`, so the
broader table geometry also carries larger center displacement.

An exact **untruncated** source-row-group decomposition exists:

`Cov(Y | source-row group) = Cov(W z_i + b | source-row group) + sigma² W Wᵀ`.

It is **not** a decomposition of the frozen gate. The gate selects generated
points by their nearest center and 3-sigma radius, then subtracts conditional
mode means. Its target per-axis conditional variance is
`q * .03²`, with `q = .949447932838815`. Conditioning couples the component
mean and Gaussian noise and permits Gaussian draws to cross the selection
boundary; selecting component means is a different operation. Thus the
central-location covariance plus the analytic unconditioned kernel cannot
be treated as an exact gate contribution or as an alternate pass calculation.

The difference matters numerically. Source-row grouping without truncation
has mean location variance **1.588350** in the control and **2.036283** in
the new state, dominated by a small number of distant means. Only **58/20,000**
control means and **209/20,000** new means lie outside the 3-sigma radius,
but their maximum distances are **33.91σ** and **36.42σ**. Adding those raw
covariances to the kernels produces variances above the target while the
actual conditional gate still reports deficits. No untruncated covariance
claim can replace that measured failure.

The saved final 20k arrays reproduce the receipt's center and covariance
metrics within `1e-12`, as do both 100k holdouts. The original five-terminal
checks and sustained-coverage verdicts remain unchanged. The diagnosis uses
final live geometry; it does not reconstruct earlier parameter states,
estimate robustness or establish any full-reference positive.

The concrete next structural question is whether the saved critic/prior
response keeps most learned means tightly clustered while allowing a
minority to broaden or drift. A bounded read-only examination of deterministic
critic radial gradients or public prior-update residual moments on these
states could inform a mechanism before another contract. Keep latent sigma,
initialization, scoring and gates fixed. These measurements support neither
width/singular-value tuning nor another automatic search or failed-parent
continuation.
