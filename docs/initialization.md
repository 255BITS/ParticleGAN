# Tested initializer: construction, math, and architecture mapping

`batch_feature_zero` has two parts: **deterministic, RMS-matched QR weights**, and
**zero initial coefficients for explicit batch-distance features in the critic**.
It also retains the original recipe's patterned biases and R2 particle prior.
The measured result is 22/22 fixed benchmark gates plus a passing long hold;
target-shift recovery still fails. This document describes the exact tested
construction first, then proposed extensions. Transformers and LoRA have not
been trained in this study.

| Component | Initialization | Evidence in this study |
|---|---|---|
| Linear/MLP weights | Deterministic semi-orthogonal matrix at the host's declared RMS | Tested in the toy suite |
| Ordinary conv and transposed-conv weights | Flatten all axes after the first; use the same QR rule; reshape back | Tested in the four image tasks |
| Explicit batch-distance readout | Zero just its coefficients; retain the per-point readout | Fixes unequal mass; repeated bit for bit on both A6000s |
| Transformer projections | Apply the matrix rule to logical projections while retaining host scales | Proposed; no transformer training result |
| LoRA | Keep the pretrained weight; deterministic nonzero A, zero B | Proposed; algebra checked, no LoRA training result |

## Usage in the package

The recipe now defaults to `initialization="batch_feature_zero"`. Its
`make_optimizers` factory initializes supported G/D/E parameters before optimizer
and EMA setup; `GANTrainer` uses that same path. `make_prior` gives learnable
particle tables an R2 cloud before any MoG calibration. Supplied priors are kept.
For pretrained/custom weights, use `get_recipe(initialization=None)`.

Loading weights into G/D/E *before* `make_optimizers` would let this default
overwrite them, so `make_optimizers` emits a `UserWarning` naming the network
(G, D, or E) and the affected parameters. Fix it by calling `make_optimizers`
before `load_state_dict`/checkpoint restore, or by setting
`initialization=None`. A constructor that runs its own init (e.g. Xavier)
also triggers it; when replacing that init is intended, call
`initialize_(network, key=...)` (G=0, D=1, E=2) before `make_optimizers`. That
gives the same weights and makes the overwrite explicit.

Detection compares each supported parameter's in-place write counter
(`tensor._version`) with the count PyTorch's own constructor leaves (one
write, two for an `nn.Embedding` with `padding_idx`). It catches
`load_state_dict`, `copy_`, optimizer steps, and custom in-place init. Fresh
modules, `.to(device)`, dtype casts, and parameters the initializer keeps
(frozen, constant, identity, zero bias, already initialized) do not warn.
Known gaps (no warning, weights still overwritten): writes that replace the
tensor or bypass its counter, namely `load_state_dict(assign=True)`,
`param.data` writes such as `param.data.copy_(...)`, `copy.deepcopy` of a
loaded network, and a device/dtype move after loading with
`torch.__future__.set_swap_module_params_on_conversion(True)`.
The prior has no such hazard: `make_prior` initializes a table as it creates
it, and a prior passed to `make_optimizers` is never reinitialized.

For a standalone network and any optimizer:

```python
from particlegan import initialize_
initialize_(network)  # before optimizer construction or loading trained weights
```

This direct API uses standard PyTorch layer declarations and stable per-call
keys (G=0, D=1, E=2 in the recipe), without global hooks or RNG consumption.
It handles standard linear, convolution, embedding, and attention parameters;
keeps normalization, frozen parameters, constant/identity matrices, and zero
biases; and leaves unknown custom parameters alone. See the
[full API contract](api.md#initialization), including checkpoint migration.
Packed `nn.MultiheadAttention` QKV uses one QR over its stored tensor. Splitting
logical Q/K/V blocks, custom depth scaling, and architecture-specific LoRA
initializers remain extensions requiring their own evaluation.

### Exact historical replay

The research hook captures arbitrary normal/uniform declarations and uses the
original optimizer-index/parameter-index keys. It is separately available for
replaying the frozen experiments:

```python
from particlegan.init_registry import install
install("batch_feature_zero")
# Construct fresh models and Adam optimizers next.
```

Or launch an unmodified script with a compatible runtime (historical drivers
still require the historical training APIs they import):

```bash
python -m particlegan.init_registry --init batch_feature_zero -- path/to/train.py --your-args
```

An explicitly installed registry initializer takes precedence over recipe
initialization. The research hooks are process-wide and Adam-specific; use one
family per process. Reinstalling resets the construction keys for fresh models.
`particlegan.batch_feature_init.uninstall()` restores its hooks without changing
weights. The direct API is the normal entry point for new applications.

The frozen 22/22 result qualifies the captured-declaration hook on its recorded
runtime. Direct initialization matches its tensors for standard G/D witnesses;
this does not turn the historical training result into a full qualification of
the current public trainer or of every architecture extension.

## 1. Exact deterministic QR construction

For a weight tensor, let `m = shape[0]` and `n = product(shape[1:])`. Interpret
it as an m-by-n matrix. The tested implementation derives a key from the
optimizer index, parameter index, and tensor shape using SHA-256. A fixed
SplitMix64 counter hash gives values u in (0,1); applying the inverse normal
CDF gives a deterministic Gaussian-shaped source matrix. No RNG is consulted
for these replacement values, and no seed or hash salt is selected by search.

Form a tall source matrix of shape `max(m,n) × min(m,n)`, compute its reduced
QR decomposition in CPU float64, and fix the sign ambiguity:

    H = Q_raw R
    Q_tall = Q_raw diag(sign(diag(R)))     # replace a zero sign with +1

Use `Q = Q_tall` if m >= n, otherwise `Q = Q_tall.T`. Thus, in exact arithmetic,

    Q.T Q = I_n     when m >= n,
    Q Q.T = I_m     when m <= n.

Let rho be the root mean square of the distribution the host originally
declared for that parameter:

    Uniform(a,b): rho² = (a² + ab + b²)/3
    Normal(mu,sigma): rho² = mu² + sigma²

Initialize

    g = rho sqrt(max(m,n))
    W = g Q.

Because `||Q||_F² = min(m,n)`, this yields exactly

    ||W||_F² / (m n) = rho².

All nonzero singular values are g. This preserves declared entry RMS; it does
not preserve the whole random distribution, its realized sample values, or a
nonzero weight mean. The Gram identities are approximate after floating-point
QR and conversion to the model dtype.

For example, if a linear host declares `Uniform(-1/sqrt(n), 1/sqrt(n))`, then
`rho = 1/sqrt(3n)` and `g = sqrt(max(m,n)/(3n))`. The tested recipe inherits its
host's scale rather than replacing every layer's gain with 1 or sqrt(2).
PyTorch's orthogonal initializer uses the same trailing-axis flattening
convention; our source construction and gain selection are specified above.
[PyTorch initialization reference](https://docs.pytorch.org/docs/2.14/nn.init.html#torch.nn.init.orthogonal_)

### Biases and particle prior are part of the recipe

The evaluated variant is `w=qr,s=std,b=pattern,p=qmc` in the frozen original F
implementation. **Its randomly declared biases are not universally zero.**
For a bias vector with more than one element, a deterministic hash-uniform
vector v is standardized to population mean 0 and variance 1, then mapped to
`mu_b + sigma_b v`, using the declared distribution's mean and standard
deviation. A scalar bias retains a deterministic hash value at the declared
scale; a scalar cannot have that vector's exact empirical variance.

For N particles in d dimensions, the R2 prior uses a generalized golden ratio
phi > 1 satisfying `phi^(d+1) = phi+1` (64 fixed-point iterations in the code):

    u_ij = fractional_part(0.5 + i phi^(-j)),  i=1..N, j=1..d.

Map u affinely for a declared uniform prior, or through the inverse normal CDF
for a normal prior. This is the tested quasi-random construction; **it is not
an exact finite-cloud whitening operation**. Its empirical covariance is not
guaranteed to be identity. Constants and host-written identity/zero parameters
are preserved according to the implementation's initialization-tag checks.

### What repeatability means here

The hook observes the host's normal/uniform initialization declarations, lets
the original draws occur, and replaces supported fresh parameters when Adam
is constructed. Keeping those draws preserves the original subsequent sample
stream in these comparisons. Repeatability depends on parameter/optimizer
order, shapes, recipe, software and numerical environment, as well as training
randomness. The empirical cross-GPU repeat used the same A6000 model and stack;
it is not a promise of identical floating-point QR across library versions.

The frozen hook is a research adapter. Untagged nonconstant parameters can be
left untouched (`UNTAGGED_RANDOM_KEPT`); that must be audited when extending it.
It is not an automatic deterministic initializer for every module or optimizer.
The direct API replaces the global optimizer counter with an explicit network
key. It uses the same tensor key for ordinary G/D parameter order; new custom
layouts and encoder ordering require their own evaluation.

## 2. The batch-feature correction and its guarantee

The evaluated batch-distance critic has the form

    D_i(X) = a.T h(x_i) + b.T phi_i(X) + c.

Here h is the learned per-point MLP, and phi consists of four differentiable
distance statistics over the current batch. The additional operation is
`b = 0`: only four scalar entries of the output weight change. The per-point
weights a, bias c, hidden layers, generator and prior retain their QR recipe.

At initialization, the batch branch contributes neither a score nor any input
derivative. Because this h has no cross-sample operations,

    partial D_i / partial x_j = 0   for i != j.

The per-point score and its derivatives remain available. The loss gradient
into b is a weighted feature difference and can be nonzero even when b=0, so
the unchanged optimizer can learn batch-dependent scores immediately. This
guarantee applies at initialization; it does not persist after b is updated.
If another model has BatchNorm or some other cross-sample operation in h,
zeroing b removes only this branch's contribution, not every cross-sample
derivative in that model.

Why four weights matter: each distance statistic divides by a kernel scale
s_k². For a nearly coincident cloud of n samples, define

    q = sum_k b_k / s_k²,
    P = I_n - 11.T/n.

The Hessian of the summed batch score at a coincident cloud is exactly

    Hessian_X sum_i b.T phi_i(X) = 4 n q/(n-1+eps) (P tensor I_data).

In the failed QR run, q = -4.349527. Ascending that initial summed score
contracts the nontranslation directions locally. On the actual initialized
cloud, the batch contribution's input-gradient RMS is 25.79 times the
per-point contribution. Initializing b=0 removes this arbitrary preference.
The derivation and measurements are in [the mechanism report](../reports/toy100/batch-feature-init/BATCH_FEATURE_MATH.md)
and [its diagnostic](../reports/toy100/batch-feature-init/batch-force-diagnostic.json).

This is a guarantee about the initial function and derivatives. The actual
trainer updates D first and uses sampled relativistic logistic losses and
Adam. Neither this Hessian argument nor semi-orthogonal weights constitute a
convergence theorem for that complete training process.

## 3. Linear layers

For `y = W x + bias`, apply `W = g Q` directly. In a tall layer, the linear
part preserves all input norms up to g; in a wide layer, its rows are
orthogonal but it necessarily has an input nullspace. A nonlinear stack's
Jacobian also contains activation derivatives, so these individual matrix
identities do not imply an isometric network or convergent training.

For a layer that concatenates ordinary and explicit batch features,

    y = W_main h + W_batch phi + bias,

retain the deterministic nonzero `W_main` and set `W_batch=0`. The tested
implementation does this after constructing the full QR head, so the modified
head itself need no longer have the original QR Gram matrix or RMS. Ordinary
linear layers with no such auxiliary branch receive no selective zeroing.

## 4. Convolutions

For a standard kernel `K[C_out, C_in/groups, kH, kW]`, the current rule flattens
to `W[C_out, (C_in/groups) kH kW]`, applies QR with the captured rho, then
reshapes. The image hosts use ordinary groups=1 convolutions. The same
stored-tensor rule is applied to ConvTranspose2d, whose first dimension is
**C_in**, with storage `K[C_in, C_out/groups, kH, kW]`; it must not be described
as an output-channel flattening for that layer.
[Conv2d storage and groups](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html),
[ConvTranspose2d storage](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose2d.html)

These are Gram identities for a flattened kernel, not for the complete spatial
convolution operator. For example, the unit-norm 1D kernel `[1,1]/sqrt(2)` has
`W W.T = 1`, but on a length-three input its valid convolution matrix is

    T = [[1,1,0], [0,1,1]]/sqrt(2),
    T T.T = [[1,0.5], [0.5,1]].

Overlapping windows already break full-operator orthogonality; strides,
padding and boundary conditions matter as well. This study does not establish
convolutional dynamical isometry.

For grouped/depthwise extensions, an architecture-aware implementation could
initialize each independent group block separately, using a distinct key and
the host's declared scale. That differs from the current whole-stored-tensor
QR rule and has not been benchmarked here. If a convolutional critic adds
explicit batch-statistic channels to its score head, the targeted correction
would zero their connecting coefficients while retaining the ordinary path.
It does not mean zeroing ordinary convolution kernels.

## 5. Transformers — proposed adaptation, not a measured result

Attention uses learned projections and a scaled score matrix:

    Q_att = X W_Q.T,  K_att = X W_K.T,  V_att = X W_V.T
    S = softmax(Q_att K_att.T / sqrt(d_k))
    Attn(X) = (S V_att) W_O.T.

This formula comes from the original transformer construction.
[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

For training from scratch, the proposed QR adaptation initializes each logical
Q/K/V/output and feed-forward projection at its existing host-declared RMS,
with distinct deterministic keys. Fused QKV storage needs an explicit policy:
one QR over a packed tensor does not give each Q/K/V block the same guarantee
as three separate QRs. Splitting the blocks is a proposed extension, not
behavior provided by the tested generic flattening hook. Retain host attention
scaling, depth-dependent gains, normalization constants, position conventions,
embedding/output tying, and task-required output paths.

Orthogonal projections alone do not control attention entropy, query/key
correlation, softmax saturation or the full block Jacobian. Token-to-token
attention is not the toy critic's sample-to-sample batch-distance branch, so
the toy result does not justify zeroing all attention outputs.

A related, **optional experiment** is a neutral existing residual branch. For
a pre-normalized block of the form

    Y = X + F(LN(X); theta) C.T + 1 c.T,

setting `C=0, c=0` makes `Y=X` and `dY/dX=I` initially, with theta initialized
nonzero. C can learn on the first update; the branch's internal theta has zero
task-loss gradient on that first update and starts receiving it after C moves.
For attention, C is its output projection; for an FFN, it is the last affine
projection. A post-normalized block instead starts as `LN(X)`, not identity.
Zeroing a nonlinear/normalized branch requires checking its actual output,
including biases. This is a mathematical analogy, not a trained recommendation
for replacing the standard transformer initialization.

ReZero studies a related zero-initialized residual gate. Adding such a gate
changes the architecture and was not part of our initialization-only search.
[ReZero](https://arxiv.org/abs/2003.04887)

For a pretrained transformer, preserve its learned base weights and existing
normalization parameters. Apply any new initialization only to newly added
trainable modules. The direct API skips frozen parameters. For a trainable pretrained base, call
`initialize_` only on the new modules and use `initialization=None` in the recipe.

## 6. LoRA — proposed deterministic adapter initialization

For conventional LoRA, using column-vector notation,

    W_eff = W_0 + s B A,
    A has shape [r,d_in], B has shape [d_out,r], s = alpha/r.

W_0 is the pretrained base. The original LoRA paper initializes A nonzero and
B zero; PEFT also documents a default zero B. Our proposed extension replaces
the random A with deterministic, RMS-matched QR while retaining the zero B.
[Original LoRA paper](https://arxiv.org/pdf/2106.09685),
[PEFT initialization reference](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora)

For the usual `0 < r <= min(d_in,d_out)`:

    A = g_A Q_A,  Q_A Q_A.T = I_r,
    g_A = rho_A sqrt(d_in),
    B = 0.

Use the existing LoRA recipe's rho_A and scaling s; a recipe using a different
rank scaling keeps that scaling. Keep W_0 and any existing base bias untouched,
and initialize any new additive adapter bias to zero. Then `s B A = 0` exactly:
the adapter initially leaves the base function unchanged. The nonzero A
provides a deterministic rank-r feature subspace for B to learn from.

Let `G = partial L / partial W_eff`. Differentiation gives

    partial L / partial B = s G A.T,
    partial L / partial A = s B.T G.

At B=0, A's task-loss gradient is zero while B's can be nonzero. Setting both
A and B to zero would make both task-loss gradients zero. Keeping A nonzero is
therefore essential to this particular trainable neutral-branch construction;
it does not guarantee that every data gradient has a component in A's subspace.

For one plain SGD step from B=0, with P_A = Q_A.T Q_A,

    Delta W_next = -eta s² G A.T A = -eta s² g_A² G P_A.

P_A is a rank-r orthogonal projector. This describes the initial projected
update, not the elementwise-preconditioned Adam update and not a convergence
guarantee. The identities are our derivation. A different A changes subsequent
learning even though the initial forward function is the same. We have not
measured whether this QR A beats the existing LoRA initializer.

For attention LoRA, apply the rule to each declared target projection and
preserve tied/fused weight semantics. For convolutional LoRA, respect the
implementation's factorization and groups: a down projection A followed by
an up projection B is initially neutral when B is zero. Treating it as one
flattened `B A` requires verifying that the convolutional composition actually
has that matrix representation; the linear-layer derivation is not automatic
for arbitrary spatial kernels.

## Reproduction and validation boundary

The tested sources are frozen under
`/ml2/hypergan/pr194-init-search-20260926/batch-feature-full-suite/source/`:
`ortho_init.py` supplies the original F QR/pattern/R2 construction, and
`failure_init.py` applies `batch_feature_zero`. `failure_worker.py` installs it
before models and optimizers are created. The existing public registry name
`qr_pb_pq` resolves a different construction; the candidate is now registered separately as `batch_feature_zero`.

The retained research script [`verify_initialization_math.py`](../reports/toy100/batch-feature-init/verify_initialization_math.py) checks the matrix scaling/Gram identities, the
convolution counterexample, and the proposed residual/LoRA gradient identities
on fixed tiny tensors, without training or seed experiments. Such algebra
checks do not extend the 22/22 training evidence to transformers or LoRA.
Full training results and audits are summarized in
[the research report](../reports/toy100/batch-feature-init/README.md).
