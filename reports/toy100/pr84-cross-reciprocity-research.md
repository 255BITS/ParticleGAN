# Fixed-rate PR84: paired cross-field diagnostic

**Result:** the archived one-step collapse at update 1325 is not accompanied
by a large adverse symmetric cross-field term in the direction of its actual
accepted D and G/prior updates. The generator's own directional field
curvature is negative there, but is also negative at the adjacent passing
update. This is evidence against a *cross-dominance explanation for these
particular saved directions*, not evidence that the full alternating Adam map
is stable or that negative own curvature caused the quality loss. No optimizer
or training run was changed.

The host is not an exact zero-sum differentiable game: its relativistic-pair
logistic losses are `L_D=softplus(-Δ)` and `L_G=softplus(+Δ)`, where
`Δ=D(real)-D(fake)`; it also has a D-only input-gradient cap and a G-only
spatial critic stencil. The D and G host phases sample independent batches.
Consequently the off-diagonal Jacobian blocks need not be skew transposes.
This matters because own-player curvature caps measure only diagonal field
changes, while full local monotonicity depends on the symmetric part of the
whole game Jacobian [Balduzzi et al., ICML 2018](https://proceedings.mlr.press/v80/balduzzi18a.html).
The [alternating-GDA rate result](https://proceedings.mlr.press/v235/lee24e.html)
assumes strongly convex–strongly concave smooth objectives; those assumptions
are not established for this LeakyReLU/Adam host.

For one scalar pair `Δ=d(1-g)`, the D/G cross-derivative sum is
`1−2σ(Δ)−2Δσ(Δ)(1−σ(Δ))`: zero at `Δ=0`, **−0.855341** at `Δ=1`.
The two analytic tests verify this sign and that replacing the G field by the
negative gradient of the *same* D loss makes the cross sum zero. This algebra
motivates measuring the actual host but does not predict its field sign.

## One frozen-state probe

We use the exact original PR84 capture at updates 1324, 1325, 1530 and 1539.
The first real/latent D minibatch and output-noise draw are held common across
all paired perturbations and both player losses. This variance-reduced common
batch is **not** the host's actual later G minibatch. The probe rebuilds the
original network/prior at the saved accepted-D point, freezes the observed
stencil width, and perturbs D and G/prior separately along their actual
accepted updates. Every directional value below is normalized by the saved
post-update Adam metric `P=lr/(sqrt(vhat)+eps)`; G and prior keep their
different constant rates. D adversarial loss, D `b_cap`, sharp-G loss and the
G-stencil increment are reported separately. The cap was inactive on this
specific common batch at the accepted D, not necessarily on other batches.
Known mode centers enter only the archived pre/post quality grades, never a
gradient or perturbation.

| Update | HQ pre → post | Cross term | G own term | Whole directional symmetric Rayleigh |
| ---: | ---: | ---: | ---: | ---: |
| 1324, passes | .99902 → .99438 | −.006280 | −.115430 | +.102159 |
| 1325, fails | .99658 → .82422 | +.001762 | −.233698 | +.040313 |
| 1530, passes narrowly | .96899 → .91846 | +.008330 | −.458296 | −.134375 |
| 1539, already failing | .79443 → .79517 | +.001507 | unresolved | unresolved |

At 1325, cross pieces are D adversarial `+.083446`, D cap `0`, sharp G
`−.115709`, and G-stencil increment `+.034025`; they nearly cancel. The G
stencil mitigates negative own-G curvature from `−.315133` sharp to
`−.233698` in this direction. A negative whole Rayleigh at 1530 disproves
local monotonicity along that sampled direction only; a positive value at
1325 does not certify monotonicity in other directions or stability of Adam.

The first 1% and 0.5% step-fraction secants crossed LeakyReLU boundaries and
failed the true-zero-sum numerical null. They remain in the raw receipt as a
**rejected derivative calibration**. Float64 derivatives at the frozen
float32 weights with fractions `0.0001` and `0.00005` make the same-batch
zero-sum null ≤`3.2e−11` and agree on the reported terms for 1324, 1325 and
1530. The own-G values at 1539 disagree by 45.7% because a kink remains
nearby, so no own-G or whole-Jacobian inference is made there. The saved
LeakyReLU network is not globally smooth; these are local directional
derivatives, not global Lipschitz certificates.

[Mulayoff and Stich, COLT 2026](https://proceedings.mlr.press/v336/mulayoff26a.html)
shows in a single-objective setting why nonlinear minibatch trajectories can
escape an average or quadratic stability analysis; it is not an adversarial
Adam theorem. Constant-step stochastic VI methods can instead approach an
invariant *distribution*, with residual bias, under their assumptions
[Vlatakis-Gkaragkounis et al., AISTATS 2024](https://proceedings.mlr.press/v238/vasileios-vlatakis-gkaragkounis24a.html).
[Ha, COLT 2026](https://proceedings.mlr.press/v336/ha26a.html) requires
structured NC-PL or NC-concave games for high-probability SGDA guarantees;
neither structure is shown here. None of these results turns a positive or
negative one-direction secant into an indefinite HQ guarantee.

The general-sum [Follow-the-Ridge paper](https://arxiv.org/html/1910.07512v2)
uses a leader *total* derivative through a local follower response, including
an inverse follower Hessian. Simply evaluating G against an approximately
fitted D is not that update. Its local analysis assumes smoothness and an
invertible follower Hessian; neither is established for this LeakyReLU critic,
whose additive score bias is a gauge direction. Its mixture-GAN experiment
uses tanh networks and adds D L2 regularization to obtain a nonsingular
Hessian. The present D-cross Adam secant is also not a Hessian inverse.

This diagnostic does **not** warrant a new cross-coefficient sweep. The
separate observed improvement from penalized critic fitting is a more direct
reason to test one declared D-refinement mechanism, with its own field-call
cost and unchanged cold/stationary gates. It remains a hypothesis: a
nonconverged critic fit and a local output repair do not certify a best
response or a production recipe.

## Reproduction

The [hashed receipt](continuous-evidence/pr84-cross-reciprocity/manifest.json)
includes the four required saved states, the complete numerical calibration,
the operator values and an exact source snapshot. The saved-state subset is
derived from the full capture whose SHA is recorded in both the earlier exact
replay and this manifest. The output from the portable subset exactly equals
the full-sidecar output on every measurement row.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' ATEN_CPU_CAPABILITY=avx2 \
  MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 \
  /tmp/pr38-default-env/bin/python -u \
  reports/toy100/pr84_cross_reciprocity_diagnostic.py \
  --diagnosis reports/toy100/continuous-evidence/pr84-stationary-failure-diagnosis/diagnosis.json.gz \
  --states reports/toy100/continuous-evidence/pr84-cross-reciprocity/selected-four-states.pt.gz \
  --output /tmp/pr84-cross-reciprocity-reproduction.json

/tmp/pr38-default-env/bin/python -m pytest -q \
  tests/test_pr84_cross_reciprocity_diagnostic.py
```
