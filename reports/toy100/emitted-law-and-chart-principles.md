# Emitted-law geometry, critic response, and a fixed affine chart

The smallest useful next test is an **exhaustive remaining donor-exchange
certificate on the actual emitted-law objective**, followed by the unchanged
mass and shape measurements. A particle cloud that stops moving under local
gradients can still have an improving allocation. Conversely, a certified
small distribution-objective gap can coexist with a failed quality metric.
Neither outcome should trigger another neural training run without attribution.

This note contains mathematical controls and source inspection, not a new
training result. The separate [finite-bank study](pr84-finite-bank-variance-geometry.md)
and actual mean-Adam control establish why critic variance is still a live
dynamics hypothesis. Their measured finite-mean game also has an expanding
generator direction. The two questions are distinct: removing estimator
noise need not repair mean-field geometry, and changing the data objective
is not an optimizer-only fix.

The subsequent native continued-state mean16 replay failed **24/44** checks:
12/12 at 1324–1335, 11/16 at 1380–1395, and 1/16 at 1530–1545. Independent
read-only review verified its archived/live source hashes, unchanged native
RNG/noise, rates, and once-advanced Adam clocks. All D-bound factors were one.
The later first failure moved HQ `.911865 -> .863281` with G factor `.824`
and clean RMS `.02364`; the severe branch began `.921875 -> .876953` with
factor `.425` and RMS `.02504`. Thus avoiding the original update-1325 event
did not prevent directed drift elsewhere. This is not evidence of numerical
explosion or effective freezing. The source-bound result lives in
`artifacts/continuous-learning/round8/mean16-continuous-filter-v2` in the
integration worktree; no warm run follows its failure.

## A specific no-Nash example, with a restricted scope

Consider the unregularized population Rp logistic game with unrestricted
critic functions. Let the target be a finite Gaussian mixture with common
covariance `t² I`, and the generator be a finite Gaussian mixture with common
covariance `s² I`, where `0 < s < t`. All centers are finite and all retained
weights are positive. Generator strategies permit translating the whole
cloud while retaining its weights and output-noise scale.

For fixed positive densities `p,q`, balanced classification of the ordered
pairs `(real,fake)` and `(fake,real)` gives a discriminator optimum

`D*(x) = log p(x) - log q(x) + C`.

Its difference across a pair is the optimal log odds. The additive constant
does not affect either Rp loss. For the stated mixtures,

`D*(x) = [1/(2s²) - 1/(2t²)] ||x||² + O(||x|| + 1)`.

This follows by factoring the common quadratic out of each Gaussian mixture;
the remaining log-sum-exp of finitely many affine functions grows at most
linearly. Hence `D*` is continuous, coercive, and bounded below by a finite
number `m`.

The non-saturating generator minimizes

`L_G(q';D*) = E_{X~p,Y~q'} softplus(D*(X)-D*(Y))`.

Translate any fixed finite generated cloud by `v`, with `||v|| -> infinity`.
For each fixed noise realization, `D*(Y+v) -> infinity`, so the integrand
tends to zero. It is dominated by `softplus(D*(X)-m)`, which is integrable
under the Gaussian target. Dominated convergence gives `L_G -> 0`. Every
finite cloud has strictly positive loss, so no finite generator strategy
attains a best response to this optimal critic. **There is no pure Nash
equilibrium within these stated strategy classes.**

This is an elementary result for this population Rp game, not a theorem
about the repository's `b_cap`-penalized finite Fourier MLP. It also does not
prove that a bounded high-quality trajectory is impossible, or that a proper
distribution objective has no finite best approximation. The related
[Farnia–Ozdaglar analysis](https://proceedings.mlr.press/v119/farnia20a.html)
studies failures of Nash existence for neural GANs and proposes proximal
equilibrium; its assumptions and equilibrium concept should not be silently
substituted for this example.

The included one-dimensional quadrature test uses `t=.07`, `s=.029` and a
generated mean at zero. The fixed ratio critic gives zero first derivative
but **negative** generator own curvature there, and translating the generator
reduces its loss toward zero. In the same misspecified family, population
negative log likelihood has mean-dependent term
`(t² + theta²)/(2s²)`, a finite minimum and positive curvature. Thus
misspecification alone is not a universal learning obstruction; the game
and the optimization objective matter.

## An allocation certificate for noise-aware MMD

Let `k` be a positive-definite kernel and define the feature of a clean
particle center by its **emitted** distribution,

`psi(c) = E_eta k(c+eta, .)`, with `eta ~ N(0,sigma_out² I)`.

For `N` equal-weight particles, `m = N^-1 sum_j psi(c_j)`. For a fixed
empirical real-data embedding `m_P`, use the squared MMD
`F(m)=||m-m_P||²`. The real-real term is constant but must be retained if
reporting an absolute value or gap.

Suppose a complete read-only check of every occupied donor `a` and every
candidate center `b` in a fixed set `C` finds no replacement improving `F`
by more than `epsilon >= 0`. One replacement changes the embedding by
`(psi(b)-psi(a))/N`. Expanding the square gives

`2 <m-m_P, psi(a)-psi(b)> <= ||psi(a)-psi(b)||²/N + N epsilon`.

Let `D²` bound those feature differences. Average the inequality over the
current equal-weight donors and over any probability distribution on `C`.
Convexity of the squared norm then gives

`F(m) - min_{nu in Prob(C)} F(m_nu) <= D²/N + N epsilon`.

The comparator allows arbitrary nonnegative mixture weights on `C`.
Current particles need not lie in `C`; the same convexity argument still
applies. At exact exchange stationarity, the bound is `D²/N`. A bounded
search that merely used all its allowed moves has not established
stationarity: report its **best remaining improvement** as `epsilon`.
Candidate subsampling only certifies the sampled set.

For the normalized Gaussian kernel
`k(x,y)=exp(-||x-y||²/(2h²))`, analytic noise integration gives

`<psi(a),psi(b)> = A exp(-||a-b||²/[2(h²+2sigma_out²)])`,
`A = [h²/(h²+2sigma_out²)]^(d/2)`.

The inner products are nonnegative and all squared feature norms equal `A`,
so `D² <= 2A`. No unknown target centers, group count, or radius enter this
certificate. This is a discrete equal-mass counterpart of a convex-hull
linear-oracle gap argument; see [Jaggi's primary Frank–Wolfe treatment](https://proceedings.mlr.press/v28/jaggi13.html).
The finite exhaustive test checks both zero-improvement and residual-gap
forms against a known convex mixture optimum.

[MMD's characteristic-kernel theory](https://www.jmlr.org/papers/v13/gretton12a.html)
identifies distributions at zero population discrepancy. It does not turn a
small finite `2A/N` bound into the host's mode, mass-TV or radial-KS guarantee.
A narrow local bandwidth can leave a weak certificate for a distant absent
component; a wide bandwidth can hide local shape error. A cumulative real
embedding is a statistical estimator, not a changing optimizer gain, but
its empirical optimum still moves when a new bank arrives. A fixed-dataset
oracle and a fresh-Gaussian-law estimator require separate claims.

The new [AISTATS 2026 particle-MMD work](https://proceedings.mlr.press/v300/belhadji26a.html)
jointly treats particle locations and quadrature weights. Its unconstrained
real-valued quadrature weights cannot automatically be used as probability
mass or as this host's equal-weight prior. The donor certificate above keeps
the emitted probability law explicit.

## The actual production affine chart removes one avoidable drift mechanism

The declared production configuration has 20,000 free two-dimensional
particles and `toy100_model="affine_square_v1"`. In
[`make_trainer`](../../benchmarks/toy100/train.py), that policy creates a
two-dimensional affine generator initialized to `W=I, b=0`, after drawing
the prior in `[-5,5]²`. Its config still contains LR schedules; this source
inspection is not evidence of a constant-rate production run.

Fix any invertible affine reference `(W_ref,b_ref)`. A desired finite clean
output cloud `Y` has the exact representation

`Z = (Y-b_ref) W_ref^{-T}`.

Thus fixing this production generator to its identity reference and learning
the prior cloud loses no finite clean-output expressivity in this particular
architecture. It removes the shared G/prior coordinate redundancy and the
parameter holonomy of incremental nonlinear output fits. Every fit uses the
same reference: a closed output loop returns to exactly the same prior.
Bounded `Y` implies bounded `Z`, with bound proportional to
`||W_ref^-1||`. This does not bound the critic or an unbounded sequence of
empirical real outliers.

It changes which parameters own learning and therefore is not the unchanged
GAN optimizer. The argument does not transfer to the conditional trajectory
MLP or an image generator. It avoids a structural gauge problem in the
actual affine host rather than proving global boundedness for general neural
Gauss–Newton. The September 2026
[Gauss–Newton drifting paper](https://arxiv.org/abs/2609.17167)
connects least-squares output fitting and natural gradients; its asymptotic
result needs additional function-class, capacity and fitting-error conditions.
It supplies no fixed finite-network parameter-boundedness theorem here.

## A generic likelihood alternative and its cheapest falsifier

For a fixed candidate dictionary `C`, let
`K_ij = Normal(x_i; c_j, sigma_out² I)` and probability weights `w`.
The emitted-law likelihood

`J(w) = -M^-1 sum_i log((K w)_i)`

is convex on the probability simplex. This is a generic kernel mixing law;
it does not fit a Gaussian to each inferred group. The fixed-dictionary
EM/MM update
`w'_j = w_j mean_i K_ij/(K w)_i`
preserves nonnegative unit mass and decreases the empirical objective.
Its update gain is not scheduled by elapsed training time. Exact zero
weights never revive under this formula, so dictionary expansion or explicit
mass reallocation remains essential for discovery. A new atom's first-order
score is `1-mean_i k_c(x_i)/q(x_i)`, obtained by differentiating
`(1-alpha)q + alpha k_c`. This tests likelihood signal anywhere, including
inside an old partition, without fixed group identities.

The [JMLR 2024 mixture-NPMLE work](https://www.jmlr.org/papers/v25/22-1120.html)
analyzes and solves the finite-support convex weight problem. The
[April 2026 density-estimation result](https://arxiv.org/abs/2604.12087)
adds rate results under assumptions such as known bounded support and,
for its faster stated rate, a finitely discrete true mixing measure. Those
are not finite-sample host guarantees. In particular, matching a wider
Gaussian with narrower fixed emitted kernels can require a continuous
mixing distribution.

The cheapest falsifier would use an actual saved production real bank,
declare its finite dictionary once, solve only its probability weights to
a reported convex gap, and materialize the resulting law in the fixed
identity chart. Inspect the **original** coverage, mass and shape metrics
on a separate native evaluation bank before building a neural adapter.
Failure at a small optimization gap attributes the limitation to dictionary,
sample, representation or objective—not neural pullback accuracy. No such
production solve or training run is claimed in this note.

Equal-weight particles only approximate arbitrary mixture weights. Largest
remainder allocation gives per-dictionary-coordinate error less than `1/N`
and mass-TV error at most `|C|/(2N)`; `|C|` is the full dictionary size, not an
unobserved true group count. Under a separated eight-group assignment, twelve
equal atoms cannot give eight equal masses (minimum assignment TV `1/6`),
whereas 20,000 atoms are divisible by the production's 100 equal groups.
That arithmetic fact does not guarantee within-group spread.

For every emitted law with fixed isotropic noise,
`Cov(Y)=Cov(clean)+sigma_out² I`. A target covariance below that floor is
unrepresentable, independent of the controller. Production `.03` data noise
exceeds `.029` output noise, so this particular floor does not rule out the
production law. Centroid-only anchors omit the additional spread and cannot
stand in for full distribution matching.

## Verification and next decision

Six deterministic mathematical tests cover the affine chart, the scoped
Rp tail-escape example, convex weight descent/rest, new-atom derivative,
integer mass/noise-floor restrictions, and the exhaustive MMD exchange
certificate. All six pass; together with the three finite-bank VR identity
tests, the focused run is **9/9**. They do not substitute for a host gate.

The evidence supports two separately labeled paths: test the mean-critic
dynamics on continued native states, and test an explicit emitted-law
objective with a global allocation certificate. Neither a failed pure-game
example nor a successful toy support controller establishes a universal
impossibility or a general GAN solution.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' \
  /tmp/pr38-default-env/bin/python -m pytest -q \
  tests/test_emitted_law_principles.py tests/test_pr84_finite_bank_vr_diagnostic.py
```
