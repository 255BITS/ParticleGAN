# Generator Jacobian conditioning: feasibility, not a next recipe

**Do not admit a global clamping recipe from the current evidence.** Continuous
Ring geometry motivates controlling excessive stretch and lost spread, but a
shared mathematical rule, scale and public binding remain unresolved. This
read-only review adds no implementation, candidate, training, draws or grades.

[Odena et al., ICML 2018, §4 and Algorithm 1](https://proceedings.mlr.press/v80/odena18a/odena18a.pdf)
penalize a sampled finite-difference quotient
`Q = ||G(z + delta) - G(z)|| / ||delta||` outside two bounds. This adds perturbation
draws and another generator forward; it does not guarantee singular-value bounds.
Their image experiments use bounds 1–20 and perturbation norm 1, with bounds
selected from good unregularized runs' quotients. These settings establish no
shared coefficient or scale for the present suite. The
[supplement, §B.1](https://proceedings.mlr.press/v80/odena18a/odena18a-supp.pdf)
also exhibits a poorly performing, well-conditioned generator: conditioning alone
is insufficient.

## Fixed-host compatibility

| Original host | Mathematical applicability and ownership |
| --- | --- |
| Ring16 | The generator maps 4D latent inputs to 2D outputs. Its input nullspace makes a positive lower sensitivity bound in every input direction impossible. Conditioning the two output-space directions through `J J^T`, or the nonzero singular spectrum, is a different regularizer from the paper's sampled quotient. |
| Scalar Gaussian | The generator maps 2D inputs to one output and also has an input nullspace. Its sole nonzero singular value has condition number 1; ratio-only conditioning supplies no width control. Absolute sensitivity bounds need a separate scale justification. |
| AE hold | The continuous 2D decoder is applicable. Its generated and reconstructed branches share parameters, so decoder protection does not establish reconstruction. Encoder and learned-prior gradient ownership must be declared. |
| Own clock | Continuous generator geometry is applicable. Any law must ignore external step labels/horizons and preserve checkpointed state and probe parity. |
| Five-word joint | The actual [WordGenerator](../../../../benchmarks/toy_audit/api_images.py) uses per-character **softmax**, not sigmoid. The task's zero-width five-atom prior needs no within-row variation. A positive local sensitivity floor can compete with confident categorical output; off-atom perturbations introduce an auxiliary neighborhood assumption. This is a possible conflict, not a proof that every finite-confidence gate is mathematically incompatible. G, inverse E and prior gradients need separate ownership. |
| Direct pole | The host optimizes generated coordinates without a sampled continuous generator-input map. Treating the map as identity gives an inert Jacobian constraint; introducing a decoder changes the task. Declare non-applicability explicitly. |
| Unused token | The host edits parameter slots without a sampled latent map. Differentiating its adapter control scale would constrain a different quantity. Declare non-applicability explicitly; retain the original unused-slot objective. |

These dimensions and exceptions belong to the unchanged task declarations and
public hosts, not to selectable trainer settings. A partial applicable-host rule
cannot silently become an active seven-host mechanism.

## What the saved Ring geometry supports

The [saved spectral prior/geometry audit](../../r1r2-spectral-repair/prior/README.md)
reports full-spectral local tangent noise **2.171 times** target variance and
radial noise **.199 times** target variance. Its observed full output variances
are **2.586** and **.635 times** target, respectively. For fixed isotropic
within-row noise, the first-order local covariance is `sigma_z^2 J J^T`.
This supports investigating directional generator sensitivity rather than
uniform contraction.

It does not justify unit isometry or a clamp weight. Full empirical component
covariance also includes mapped-center dispersion, finite-noise activation
crossings and mode reassignment. The local proxy is not an exact additive
covariance decomposition; making its eigenvalues equal does not establish the
original density gate. A well-conditioned but uniformly too wide or too narrow
map also fails that gate. Scalar Gaussian, AE and atomic words further prevent
inferring one absolute scale from Ring's target/prior widths.

## Prerequisites before implementation or admission

The current [Recipe](../../../../particlegan/recipes.py),
[GANTrainer](../../../../particlegan/training.py) and
[Forge API registry](../../../../experiments/forge/api.py) expose no reusable
generator Jacobian/clamping regularizer. The existing polar optimizer constrains
parameter updates, not the generator's input-output Jacobian.

A future proposal must specify the constrained spectrum/coordinates, a justified
shared scale and coefficient, explicit host applicability, and whether auxiliary
gradients reach G, E or learned prior locations. No settings are selected here.
Random finite-difference probes require their own named, checkpointed stream;
they must not advance existing data, latent, training-noise or evaluation streams.
They must reuse the original training samples without extra target draws and
record the additional forwards. Exact derivatives can avoid perturbation draws
but require differentiable Jacobian computation and declared additional work.
Either approach needs public host bindings, source-bound comparison receipts and
the original task allowances. Any eventual hypothesis still fails as a complete
repair if one ordinary gate or its own clock fails. This review registers no new round.
