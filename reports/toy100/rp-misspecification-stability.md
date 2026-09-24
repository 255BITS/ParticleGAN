# What the frozen ring can and cannot equilibrate

The 12-particle frozen mode-hold host cannot represent the eight-mode target
exactly, but this does **not** prove that a high-quality, fixed-target local
equilibrium is impossible. It does rule out applying convergence results whose
starting assumption is a realizable `p = q`, constant-critic equilibrium. The
observed coherent outward generator field at update 1,325 is a property of the
saved trained critic and game update, not a consequence of unequal component
counts alone. No production rule or training run was changed for this note.

The actual [mode-hold host](../../benchmarks/locked_shared/mode_hold.py) draws
128 real points per phase from eight equally weighted 2-D Gaussians with
centers on a radius-three ring and `t = .07`. The locked prior contains 12
uniformly sampled learned particles. The clean generator maps their learned
latents to 12 output coordinates; the frozen noise policy adds independent
`s = .029` Gaussian output noise. The selected simple candidate sets particle
L2 and prior regularization to zero, while its discriminator minimizes paired
logistic loss plus the sample-point cap

`(1/2)(E_real + E_fake) [max(||grad_x D|| - 1, 0)^2]`.

The cap is free below slope one and is not an R1/R2 zero pull. The host does
alternating D then G Adam updates at constant nominal rates in continuation.
PR84's experimental G step reads a five-point spatial average of D, while D
reads the sharp critic; this distinction matters when discussing an analytic
optimal critic. The [exact hold](continuous-evidence/pr84-stationary-failure-diagnosis/)
and [held-out minibatch audit](pr84-heldout-signal.md) already show transient
quality failures and a coherent outward G field at the first captured failure.

## Three logically separate claims

Write `m_k` for target centers and `y_j = G(z_j)` for the 12 clean output
points. The population densities are

`p = (1/8) sum_k N(m_k, t² I)` and
`q = (1/12) sum_j N(y_j, s² I)`.

**Exact matching is impossible.** If `p = q`, division of their characteristic
functions by the nonzero characteristic function of `N(0, s² I)` says that the
12-atom empirical law on `y_j` equals an eight-component Gaussian mixture with
variance `(t²-s²) I`. The former is atomic; the latter has a density because
`t > s`. This is a contradiction. Even if the widths were equal, exact equality
with all target components distinct would require `n_k/12 = 1/8`, or 1.5
particles on every target center. At the eight centers, four modes with two
particles and four with one achieve all-mode coverage and essentially perfect
HQ under the host's `.21` radius, while their mode-label probabilities differ
from uniform by total variation `1/6` (ignoring negligible cross-mode tails).
Thus the HQ gate is weaker than distribution equality.

**An unrestricted population critic retains a discrepancy signal.** For the
host's paired logistic D loss, `E_{X~p,Y~q} softplus(D(Y)-D(X))`, one may
symmetrize each ordered pair with the reversed class. The Bayes log odds are
`log[p(x)q(y)/(q(x)p(y))]`, so its unrestricted, unregularized minimizer is
`D*(x) = log[p(x)/q(x)] + constant`. Since `p != q`, a constant D has loss
`log(2)` and is not optimal: along `D = epsilon h`, its derivative at zero is
`(E_q h - E_p h)/2`. Any bounded-slope witness `h` with unequal expectations
improves it, and the host's `b_cap` remains exactly zero for sufficiently small
`epsilon h`. This proves a residual **population classification** signal in a
rich critic class. It does not prove that the finite Fourier MLP realizes that
witness as an accessible parameter direction, that its capped optimum is
`log(p/q)`, or that `grad D` is nonzero at every generated point. With D held
fixed, the generator's unregularized output descent direction is
`E_X sigmoid(D(X)-D(y)) grad_y D(y)`; its spatial gradient or pullback through
the shared G/prior Jacobian may vanish despite nonzero D advantage.

**Stable good-HQ behavior is a separate dynamical question.** A small
population counterexample uses one real `N(0,.07²)` and two equal fake
`N(±a,.029²)` components. The exact optimal unregularized critic is
`D*_a = log(p/q_a)`. Holding that critic fixed for each G update, the
positive-center G descent field is

`F(a) = E_{X~p,Z~N(0,1)} [sigmoid(D*_a(X)-D*_a(a+sZ)) D*'_a(a+sZ)]`.

The deterministic [Gauss-Hermite calculation](rp_misspecification_toy.py)
finds `F(.04293738) = +.3178173`, `F(.04393738) ≈ 0`, and
`F(.04493738) = -.3126534`. At the middle point the discriminator still has
advantage `log(2)-L_D = .0765515`, and one-dimensional HQ within `.21` is
`.999999995`. Eighty versus 160 quadrature nodes move the root only
`2.4e-7`. This is a **restoring best-response output-coordinate field** in a
misspecified, high-HQ family; it is not a proof of a local Nash equilibrium or
stability of simultaneous/alternating Adam. In fact, the example's optimal
critic has fake-sample slope RMS about 19.3, far above the host cap center one.
Its gradient field is therefore an informative foil, not the host's actual
capped best response. The host's shared generator and 12-to-eight occupancy
can change the field further.

## Primary research and theorem scope

- [Jolicoeur-Martineau, *The relativistic discriminator* (ICLR 2019)](https://arxiv.org/abs/1807.00734)
  defines paired relativistic discrimination and the corresponding generator
  objective used here. The population density-ratio derivation above follows
  from this loss and Bayes pair classification; it is not a claim that the
  paper proves the capped host's optimal critic.
- [Mescheder, Geiger and Nowozin, *Which Training Methods for GANs do actually
  Converge?* (ICML 2018)](https://proceedings.mlr.press/v80/mescheder18a.html)
  analyze local convergence at GAN equilibria and show the role of instance
  noise and zero-centered penalties. Their equilibrium analysis does not make
  a misspecified finite mixture realizable, and `b_cap` supplies no
  zero-centered damping while the slope stays below one.
- [Farnia and Ozdaglar, *Do GANs always have Nash equilibria?* (ICML 2020)](https://proceedings.mlr.press/v119/farnia20a.html)
  give restricted-generator, non-realizable GAN examples with no Nash
  equilibrium. Their Remark 1 also gives non-realizable settings *with* an
  equilibrium. Their games and generator constraints differ from this host,
  so non-realizability alone decides neither existence nor absence here.
- [Huang et al., *The GAN is dead; long live the GAN!* (NeurIPS 2024;
  arXiv 2025)](https://arxiv.org/html/2501.05441) prove local convergence for
  RpGAN with R1/R2 under assumptions including `p_theta* = p_data`, a locally
  constant equilibrium critic, and sufficiently small learning rates
  (Appendix C, Assumption I and theorem). All three are unverified or false
  for the current host: exact equality is impossible and the cap is not R1/R2.
- [Mulayoff and Stich, *On the Stability of Nonlinear Dynamics in GD and SGD*
  (COLT 2026)](https://proceedings.mlr.press/v336/mulayoff26a.html) show, for
  their single-objective GD/SGD setting, that nonlinear and individual-batch
  effects can invalidate a mean linearized stability judgment. This is a
  reason to keep dense single-update checks, not a theorem for alternating
  two-player Adam or evidence that this host is unstable forever.

## One cheap frozen-state discriminator

Use the existing exact `post_accepted_d` captures at updates 1,325 and 1,539;
do **no further game training**. Freeze q's 12 clean supports, D, G, Adam
moments and RNG. On 16 independent recorded/held-out batches, measure (i)
the mean actual **capped D-loss parameter gradient** and its across-batch
coherence, and (ii) the output force from both the learned critic and
`log(p/q)` with q detached, applying PR84's *same frozen five-point stencil*
to real and fake logits. Also record the analytic critic's sample-point cap
penalty and gradient norms. If the learned D gradient remains coherent and
large, critic lag is implicated; if it is small but its G field agrees with
the analytic outward field, misspecification is a plausible source of motion.
If fields disagree, finite critic/cap/stencil dynamics are implicated.
These are discriminating diagnostics, not a proof of a capped optimum: a small
parameter gradient can be a non-optimal stationary point, and `log(p/q)` may
violate the cap badly. The analytic density and exact host minibatches require
only 12- and eight-term log-sum-exp operations and the already archived states.
