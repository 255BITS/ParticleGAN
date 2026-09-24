# A capped critic can stabilize a mismatched generator by tracking it

This is a deterministic one-dimensional **population analogue**, not a run of
the particle GAN. Let (X\sim N(0,0.07^2)), (Y\sim N(\mu,0.029^2)), and let the
critic be (D(x)=ax+bx^2). Only the fake mean (\mu) is trainable. The widths
match the host's real and output noise scales, so (p\ne q) even at
\(\mu=0\). With \(\Delta=D(X)-D(Y)\), the two losses are

\[
L_D=\mathbb E\operatorname{softplus}(-\Delta)
+\tfrac12\mathbb E_X[(|D'(X)|-1)_+^2]
+\tfrac12\mathbb E_Y[(|D'(Y)|-1)_+^2],\qquad
L_G=\mathbb E\operatorname{softplus}(\Delta).
\]

These are the host's paired Rp logistic signs and its coefficient-one,
unit-slope `b_cap` penalty. The host uses
`sqrt(D'(x)^2 + 1e-12)` in place of `|D'(x)|`; their active penalties differ
only at numerical order `1e-12`. There is no R1, R2, or extra zero pull.
For each fixed \(\mu\), \(L_D\) is globally convex in \((a,b)\): its logistic
argument is affine, and the cap is squared distance from the interval
\([-1,1]\) applied to an affine slope. Its paired-logistic Hessian is positive
definite under these nondegenerate Gaussians, so a stationary finite critic is
the unique global minimizer in this restricted family.

At \(\mu=0\), the critic optimum is \((a^*,b^*)=(0,3.301866103)\), with
\(L_D=0.687290202\), including cap penalty \(0.000754371\). The cap is active
on about 3.052% of real draws and 0.0000177% of fake draws. The critic gradient
has infinity norm below \(2\cdot10^{-18}\), and its Hessian eigenvalues are
\(0.003873824,0.031954008\). The generator's partial mean gradient is zero
at this point, but the response matters:

| Critic during a small positive mean displacement | Slope of the generator's partial mean gradient at zero | Effect of a small direct mean descent step |
| --- | ---: | --- |
| Frozen at \(D^*(0)\) | \(-3.314817\) | Moves farther from zero |
| Reoptimized to \(D^*(\mu)\) | \(+4.508826\) | Restores the mean toward zero |

The critic's best-response derivative is
\(da^*/d\mu=-15.58615\), \(db^*/d\mu=0\). The fast-critic slope above means
\(d[\partial_\mu L_G(D^*(\mu),\mu)]/d\mu\), **not** a derivative of the
composite value \(L_G(D^*(\mu),\mu)\). A direct mean update with a sufficiently
small positive rate has local multiplier \(1+3.314817\eta\) with the frozen
critic, versus \(1-4.508826\eta\) after complete critic reoptimization.

Critic tracking must actually be fast in this toy. With identity metrics and
equal infinitesimal learning rates on \((a,b,\mu)\), the simultaneous
gradient-flow Jacobian has eigenvalues \(+3.238373,+0.044490,-0.003874\): it
is unstable. In these coordinate units, increasing the critic speed above
\(103.737\) times the generator speed makes the local continuous-time
linearization stable. That number is **not** a prescription for the host's
neural-network Adam rates or discrete updates.

The first term of \(L_D\) and all \(L_G\) expectations use tensor-product
Gauss–Hermite quadrature. The cap's one-dimensional Gaussian expectation is
integrated analytically; low-order Hermite sampling of its moving activation
boundary produced a misleading Hessian, so it is not used for the cap. Orders
64 and 96 agree on the reported optimum and field slopes to roughly
\(10^{-11}\). Off-equilibrium finite differences agree with analytic
gradients/Hessians within \(4.5\cdot10^{-10}\); an independent direct cap
integral agrees within \(2.3\cdot10^{-15}\). Re-fitting the critic at
\(\mu=\pm10^{-5}\) agrees with the implicit best-response derivative within
\(2.7\cdot10^{-6}\).

The example proves a narrow point: variance mismatch can coexist with a
stationary coupled field and a restoring **fast-critic** mean response under
the actual capped, paired-loss structure. The frozen-critic generator has
negative curvature here, so this stationary point is not a local Nash
equilibrium. The full host has 12 shared-network particles, eight modes,
finite minibatches, a Fourier MLP critic, output-noise sampling and Adam;
none of their stability follows from this toy. It does make critic-response
lag a specific, falsifiable mechanism rather than an impossibility caused
solely by unequal noise widths.

The [source](capped_critic_tracking_toy.py) and
[machine-readable receipt](continuous-evidence/capped-critic-tracking-toy.json)
record the formulas, both quadrature orders, root and derivative residuals,
environment versions, and source SHA-256. Reproduce with:

```bash
/tmp/pr38-default-env/bin/python reports/toy100/capped_critic_tracking_toy.py \
  --output reports/toy100/continuous-evidence/capped-critic-tracking-toy.json
```
