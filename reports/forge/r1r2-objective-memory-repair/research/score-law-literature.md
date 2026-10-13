# R1/R2 score laws and narrow-mode smoothing

**A non-saturating score-law contrast on exact AMSGrad-m110 is supported by the
API, with weak confidence. No new penalty strength is justified.** This
review registers no round and adds no implementation, training or draws.

## What the primary results establish

[Mescheder et al., ICML 2018, §4.2/Theorem 4.1](https://proceedings.mlr.press/v80/mescheder18a/mescheder18a.pdf)
prove local convergence of gradient descent with R1/R2 under realizability,
expressivity and regularity assumptions, near equilibrium at sufficiently small
steps. They also identify ill-conditioning for narrow distributions near
manifolds. Neither that theorem nor the Dirac example's critical damping
coefficient guarantees this stochastic-Adam, learned-prior, finite-budget suite.

[R3GAN, §§2.3–2.4/Appendices C and E](https://arxiv.org/html/2501.05441v1)
extends the local argument to paired relativistic training. Its StackedMNIST
demonstration favors pairing: both penalties stabilize both compared losses, but
paired training covers 1,000 modes versus 693 for classic GAN. R1 alone diverges;
lazy penalties also fail its toy demonstrations. Retain both positive penalties
every update. Its discussion of approximate Gaussian smoothing includes weighting
and Laplacian terms and explicitly does not extend that derivation to RpGAN.
Consequently, our gamma is not an established blur variance or bandwidth.

[Original LSGAN, §§3.2.2–3.2.3](https://arxiv.org/pdf/1611.04076v3)
is relevant because the API supports it. Its Pearson-divergence derivation uses
an optimal unregularized discriminator and labels satisfying `b-c=1`, `b-a=2`.
The practical alternative `(fake, real, generator)=(0,1,1)` is also in the paper.
That is the API default, but neither its labels nor our R1/R2-regularized critic
meet the stated Pearson derivation. That result supplies no repair guarantee.

## Actual score-law mechanism

The [public GANLoss](../../../../particlegan/gan_loss.py) uses raw scores. For a
scalar fake score `s_f`, real score `s_r` and sigmoid `sig`, the generator's
score derivative is:

| Supported law | Generator derivative with respect to `s_f` | Distinct mechanism |
| --- | --- | --- |
| Paired relativistic parent | `-sig(s_r-s_f)` | Real-score comparator; invariant to common score offsets. |
| Non-saturating logistic | `-sig(-s_f)` | Absolute fake-score weight; D classification anchors logits. |
| Least squares, default labels | `s_f-1` | Unbounded residual weights; different curvature/targets. |

Relativistic and non-saturating have equal per-score first derivatives at zero
and matching discriminator quadratic terms on centered scores near equilibrium.
A scalar weight cannot rotate a fixed critic's input-gradient direction.
Any shape benefit requires subsequent critic/generator dynamics.

The [saved AMSGrad-m110 score diagnostic](score-law/diagnosis.json), with its
[reproducer](score-law/probe.py) and [scope](score-law/README.md), finds no
endpoint logistic starvation: paired versus non-saturating weights have cosine
**.999997668**, with NS/RP ratios approximately **.961–.969**. These use all
16 declared target centers as an explicit diagnostic fixture, not saved
contemporaneous training pairs. A score-law change is not justified as rescuing
those endpoint generator gradients.

The distinct from-initialization hypothesis is **absolute critic score
anchoring**. Paired loss and R1/R2 are invariant to a common critic-output
offset. Non-saturating discriminator classification penalizes such an offset
and updates that degree of freedom. Although the offset itself has no spatial
gradient, its value affects classification weights and subsequent shared critic
parameter updates. Changed early dynamics might prevent later anisotropy;
endpoint observations do not establish that mechanism. The paired checkpoint's
final output-bias first, second and AMSGrad maximum-second moments are all zero.
On the center fixture, native non-saturating common-shift curvature is
**.498001**, versus zero for pairing, but its residual gradient is only
**.027667**. This demonstrates a distinct channel; it supplies weak evidence
of a harmful score offset or a future density repair.

A one-field `Recipe.loss: relativistic -> non_saturating` contrast on the exact
measured **AMSGrad-m110** parent is a concrete way to examine this hypothesis.
Keep native Adam/AMSGrad, D/G+E/prior beta1 .5/.5/0, beta2 .99, LR .0085,
gamma .1 every update, learned unregularized prior, zero noise/EMA and every
task fixed. That whole parent passes five ordinary gates and clock while failing
Ring. The field also changes G/E weighting and critic score curvature: any
observed repair cannot be attributed solely to anchoring. Confidence is weak;
the locally similar centered-score curvature, near-identical endpoint forces
and R3GAN coverage result are counterweights. The separately bounded score-law
comparison requires committed registration and independent preflight; this
review registers no round or loss grid.

## Density evidence and strength limits

The [closed objective/memory audit](../density-readout.md) records AMSGrad on
m110 covariance **1.679120 > .85**, radial variance **.413 times** target and
tangent variance **3.251 times** target. All modes are present. Full empirical
component covariance includes center dispersion and finite-noise behavior;
`sigma_z^2 J J^T` is only a local noise proxy, not its exact decomposition.
Thus neither uniform contraction nor a quieter critic establishes density.

For a score varying over spatial scale `sigma`, its squared-gradient cost scales
as `gamma/sigma^2` times squared score amplitude. This dimensional observation
motivates examining smoothing relative to narrow modes; it does not derive a
useful coefficient. Ring has component variance .01 and gamma .1, giving that
ratio 10 in its coordinate units. Setting gamma equal to variance would be an
unsupported numerical choice. Absolute score curvature, spatial density,
architecture and optimizer response matter.

Existing [constant-policy evidence](../../r1r2-tier1-search/README.md) already
rejects gamma .1/.3/1 at beta2 .9 and .99 as complete repairs. At beta2 .99,
endpoint covariance is **2.386427 / 7.872095 / 3.444710**, respectively; all
Ring gates fail. Stronger gamma is not a supported shape repair, and these
failures alone do not select a weaker value. Different-cohort
[historical weak-penalty pole failures](../../r1r2-tier1-search/evidence-diagnosis.md)
remain warnings, not grades for a new constant recipe. Historical pure-BCAP
[non-saturating](../../configuration-search/pure-bcap-non-saturating-joint-rates-v2.json)
and [least-squares](../../configuration-search/pure-bcap-least-squares-joint-rates-v2.json)
comparisons also supply no qualifying R1/R2 credit.

## Binding and falsifier prerequisites

Both alternatives already bind scalar Gaussian/Ring/clock through `g_loss`,
direct pole and token through their existing generated branch, and AE through
its original adversarial branch while retaining reconstruction/cover. Words use
`joint_g_loss`: the reversed-label encoded-real term preserves the E gradient
path. Learned prior ownership and the word's atomic softmax sampling law stay
fixed. Keep [the paired penalty](../../../../particlegan/grad_regularizers.py)
`gamma/2 * (mean ||grad D(real)||^2 + mean ||grad D(fake)||^2)` every update.
The native score laws use existing score tensors and need no additional probes
or draws; all existing stream checkpoints and actual-input/init receipts must
still match. Constant
recipe schedules remain required and clock must be measured independently.

The mechanism prediction concerns early absolute-score dynamics and subsequent
shape acquisition, not an improved endpoint force obtained by substituting a
loss derivative. Score anchoring without component-shape repair would reject
its sufficiency. A quieter critic, changed offset or lower loss is not a gate.
The eventual numerical falsifier remains the original
covariance/minimum-spread/HQ/mass/coverage bounds
at five terminal Ring checks within 1,600 updates, plus every other ordinary
gate and own clock. No task, gate, budget or sampling change follows from this
review. Any further comparison needs separate finite registration.
