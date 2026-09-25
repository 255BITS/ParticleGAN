# Read-only shared-c6 mechanism audit

**No implementation bug was identified in the reviewed rare-case training
path.** The evidence does expose a structural gap: the existing particle
regularizer can be exactly zero while every local particle cluster has zero
variance. This does not prove that the gap caused the archived failure.

The native update uses Rp logistic `softplus(D(real)-D(fake))` for G. Its fake
gradient is `-sigmoid(D(real)-D(fake))*grad_x D(fake)/batch`; G can only get
sample-shape feedback from the critic's input gradient and the global latent
regularizer. The D update correctly detaches fake samples, and the G update
freezes D parameters while retaining input gradients. The b_cap term is
`6/2 * [mean relu(||grad D(real)||-1.25)^2 + mean relu(||grad D(fake)||-1.25)^2]`.
It has zero gradient when the critic's input slope is at or below 1.25, by
design. The archived run does not record those slopes, so cap inactivity in
that run cannot be inferred.

The latent spread term in `ParticleRegularizer(weight=.05)` checks only the
standard deviation and off-diagonal covariance of **all** 256 latent rows.
The [static counterexample](analytic_counterexample.py) creates eight sets of
32 identical rows at `±2 e_j` in four dimensions. Global standard deviations
are all 1.00196, global off-diagonal covariance is zero, and the native spread
loss and every latent gradient are exactly zero. Each cluster nevertheless has
zero variance. The same script differentiates native b_cap against linear
critics: input gradient norms .5 and 1.25 give zero cap value/gradient; norm 2
gives penalty 3.375 and parameter gradient 9. The [recorded output](analytic_counterexample.json)
includes source hashes matching the archived diagnostic's numerical sources.
These are algebraic checks, with no training or benchmark selection point.

The [diagnostic replay](../shared-rare-diagnostic/README.md) records final
sampled minimum eigenvalue ratios `.1335, .2100, .1253, .3317` for the 55%,
30%, 13% and 2% components; the corresponding ratios from unique particle
outputs are `.1324, .2255, .1111, .3793`. Thus the failure in the common
components persists without replacement-sampling duplication. Their unique
output positions are recorded, but latent positions, G weights/Jacobians and
critic input-gradient norms are not. Output covariance can contract because
latent particles locally cluster, because G compresses a direction, or both;
the retained episode cannot separate those mechanisms. The lower-rate public
rare-case witness confirms the same formulation can pass under a different
optimizer recipe, so this gap is not an impossibility result.

One principled **new global formulation candidate** is a small normalized
energy-distance term on the existing real and generated G minibatches:

`L_E = [2 mean_{i,j} ||fake_i-real_j|| - mean_{i!=j} ||fake_i-fake_j|| - mean_{i!=j} ||real_i-real_j||] / s_real`,

where `s_real` is the detached root-mean-square real pairwise distance. The
fake-fake term supplies direct output-space repulsion even if the learned critic
gradient is flat, while the real-fake term anchors the full data distribution;
no component labels, target centers or evaluation metrics enter training. The
static script confirms that for narrower generated pairs, descent on this term
pushes the two groups apart. A single fixed coefficient would be required on
all 19 hosts, and the complete original-budget live curves would have to pass.
No such candidate was trained here. Its quadratic pairwise cost and Euclidean
pixel geometry on image hosts are material risks.
