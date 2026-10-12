# Saved optimizer-role evidence and two candidate hypotheses

The saved states justify separating momentum by player, but do not identify a
winning role. Recommend two first comparisons from the exact constant parent:
G/E-only beta1=0.5 and D-only beta1=0.5. Both keep learned-prior beta1=0, paired
R1/R2 coefficient 0.1 every update, beta2=.99, LR=.0085 and all other trainer/task
conditions unchanged. These are unmeasured global recipes; root owns their
finite registration and complete six-task plus clock comparison.

## What was inspected

The [compact diagnosis](diagnosis.json) validates three exact Ring16
certificates, checkpoint manifests/state hashes and retained endpoint sample
bytes: constant parent, AMSGrad and all-player momentum. It reads existing
Adam moment tensors and functionally reconstructs the saved critic input field
on the same 4,096 retained generated points. No model is constructed or sampled;
no optimizer is stepped and no scientific gate is rescored. The
[reproduction source](analyze_saved.py) preserves Python/Torch RNG states and
checks its analytic derivative against functional autograd on 16 saved points.
Maximum absolute derivative discrepancy is below 1e-12 in each arm.

Common scientific source is
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`.
Exact Ring16 identities are:

| Recipe | Revision | Attempt |
| --- | --- | --- |
| Constant | `35ebed78fe69b8923ead8335ea38bda209eb53ecc910b5e57ce5bac1c4b2f775` | `27c75f3d768e4b789a488a3fdc5a2e90` |
| AMSGrad | `5cb62c57a1c48ef668c2eb393fd05d255f3a4a776f95fe1ed94711d56ef78799` | `1dda7bf5c88942368359f1ac207e9e93` |
| All-player momentum | `3c9bfbf1cb961eaef0aa32e076f906d97ff2305f066ff83dc04d3d77423f34fa` | `a3991f4b7aeb452cb6a7caa299ad8ff2` |

Their existing actual-training GIFs remain linked in the
[constant readout](../../r1r2-tier1-search/README.md) and
[five-track readout](../../r1r2-five-track-repair/README.md). Neither historical
qualification nor original declarations change.

## Observed role histories

All three have G, prior and D groups with constant LR=.0085. In all-player
momentum every group actually uses beta1=.5. At the last saved update, all 256
prior rows have nonzero reconstructed Adam displacement, versus 105 with beta1=0
in both constant and AMSGrad states. Batch size is 128. This is consistent with
persistent momentum moving rows beyond the most recently sampled set; exact
last sampled row identities are unavailable, so individual rows are not labelled
as unsampled. Removing prior momentum is a substantive update-law difference,
not just a nominal beta annotation.

AMSGrad retains markedly different denominator history by role. Its median
`sqrt(max_exp_avg_sq)/sqrt(exp_avg_sq)` ratios are 7.711 for G, 7.683 for prior and
9.966 for D; the corresponding 90th percentiles are 29.10, 27.44 and 64.18. The
reported ratio includes the same small epsilon and uncorrected moments. This
is an endpoint diagnostic, not a complete applied-rate trajectory. An early
maximum can suppress later corrections unevenly while retaining a constant
nominal LR. The [native Adam law](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)
uses the maximum second moment before its existing bias correction.

Approximate last native Adam parameter-displacement norms, reconstructed from
saved moments, are:

| Recipe | G | Prior | D |
| --- | ---: | ---: | ---: |
| Constant | .123249 | .182335 | .316808 |
| AMSGrad | .072825 | .046831 | .113369 |
| All-player momentum | .050818 | .071101 | .152877 |

These float64 reconstructions omit float32 kernel rounding. They are different
endpoint states under different gradient histories; they cannot isolate a
causal optimizer effect. With beta1=0 the saved first moment is the final
parameter gradient, including the host's actual loss/penalties. With beta1=.5
it is an EMA; the final raw gradient cannot be recovered from one endpoint.

## Saved critic input fields

The functional gradient norm averages .258244 for constant, .206014 for AMSGrad
and .097265 for all-player momentum. A within-component restoring slope is
`-mean(centered_coordinate * matching_gradient) / variance(coordinate)`;
positive values describe an inward linear association. All 16 components have
positive tangent slopes in every arm. Radial slopes are positive in 15/16 for
constant/AMSGrad and 16/16 for momentum. This does not include the generator's
parameter Jacobian, relativistic loss weights or stochastic training batches,
so it is not the actual generator update.

The momentum critic field is weaker while its density is better; reducing
critic gradient magnitude alone is therefore not sufficient evidence for a
repair. All-player momentum still has covariance 1.185580 (> .85) and minimum
eigen ratio .052811 (< .15), and tangent variance remains 1.36–4.18 times target.
Its mean restoring slopes are .249694 radial and .564821 tangent. Endpoint
shape/field associations do not distinguish G, D and prior momentum as causes.

## Two globally consistent candidates

| Candidate hypothesis | G/E betas | D betas | Prior betas | AMSGrad |
| --- | --- | --- | --- | --- |
| G/E-only momentum | `[.5,.99]` | `[0,.99]` | `[0,.99]` | false |
| D-only momentum | `[0,.99]` | `[.5,.99]` | `[0,.99]` | false |

G/E-only momentum tests whether the generator's reduced local stretch survives
without prior inertia or critic momentum. Predict a lower Ring16 endpoint
covariance than the constant parent's 2.386427, with restored independently
confirmed Gaussian/word acquisition; the unchanged full density/spread and
terminal suffix gates still determine success. If covariance improves while
minimum eigen ratio stays below .15 or acquisition still fails, the complete
repair hypothesis is falsified.

D-only momentum tests whether smoothing critic feedback carries the density
improvement while keeping generator/inverse/prior updates at the original
zero-momentum law. Predict fewer density excursions and lower endpoint
covariance; loss of confirmed acquisition or failure to reach all unchanged
Ring16 bounds falsifies a full-suite repair. These predictions are unmeasured;
root's registered numerical signatures take precedence.

`Recipe.d_betas` is consumed by `make_critic_optimizer`; `prior_betas` applies to
sampled learned latent locations through public optimizer groups. G and E share
`Recipe.betas`. The pole fixture's direct generated coordinates also consume
`Recipe.betas`, even though its metadata says `forge_role=prior`; that fixture
has no sampled latent table and must not be relabelled by the role experiment.
One global role policy must apply consistently across all hosts.

The full factorial additionally controls D+G, D+prior and G+prior interactions;
the source-bound 000 and 111 parents can be reused exactly. Do not replace their
null role overrides with explicit equal tuples merely to canonicalize them:
reuse the original cards, revisions and task/job bindings. An AMSGrad plus
all-player-momentum interaction is distinct from both parents, but permanent
moment maxima may also worsen the observed minimum-spread failure. The current
snapshots cannot predict that interaction with confidence.

```sh
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-role-repair/optimizer/analyze_saved.py > reports/forge/r1r2-role-repair/optimizer/diagnosis.json
```
