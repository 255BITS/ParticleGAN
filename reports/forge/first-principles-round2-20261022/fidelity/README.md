# Learned Bregman c-transform game

Candidate **`bispf-bregman-game-v1`** is admitted by cheap analytic/software controls for the shared bounded comparison. It has **no trained result yet**. Its only recipe delta from the original `bispf-optimized-quintic-v1` parent is `bispf_bregman_game: false → true`. This replaces both adversarial payoffs and their generator force; it is not a coefficient adjustment or another auxiliary discrepancy. `output_energy_weight=fidelity_transport_weight=0`. The existing learned prior, SBNP policy/coefficient, BISPF critic penalty/observer, optimizer and direction protection remain unchanged.

The implementation base is `96af7be246261f27db9794a4fa787dbe2a0487a9`. Scientific execution must bind the later common committed source; this proposal is not an execution-source receipt. The parent protocol owns the all-15-task comparison, original architectures/data/initial priors, seed0 public initializer, complete consumed streams, fixed update budgets, metrics and actual-training GIFs. No new leaderboard, gate, seed sweep or paid training is created here.

## Why this mechanism

The [completed first round](../../three-theme-repairs-20261022/fidelity/README.md) rejected “global marginal agreement implies local fidelity.” Quartic transport repaired overlap and strongly improved width allocation, but width passed only 2/24 complete checks with a maximum consecutive run of one. Its narrow `.07` component still had 54/1012 samples beyond four standard deviations, accounting for 85.31% of scatter. The clean `G(location)` census already contained bridges/tails. Intensity retained balanced nearest-template counts while clean samples fell 19→9. Blobs collapsed to almost-zero sigmoid outputs. These outcomes motivate a different *force and game*, rather than another aggregate witness.

The hypothesis is that a critic used as a target-price field can supply the marginal correction, while an explicit output-space potential supplies a restoring force to actual target samples. Bernoulli Bregman geometry cancels the sigmoid derivative analytically, before floating-point sigmoid saturation loses the signal. This remains a hypothesis about the measured failure mechanisms; shared parameter updates can still distort that force.

## Declared payoff and pullback

For the existing fake batch `x_i=G(z_i)`, real batch `y_j`, and scalar critic prices `C(y_j)`, maximize in C and minimize in G:

\[
 V(C,G)=m^{-1}\sum_j C(y_j)+n^{-1}\sum_i\min_j[c_h(y_j,x_i)-C(y_j)].
\]

D minimizes `-V` with cost coordinates detached; the actual existing BISPF penalty is added unchanged. Both existing real/fake critic forwards are retained. G minimizes the second term with target coordinates and prices detached. First-index hard argmin ties are deterministic. There is no assertion that one minibatch update is balanced: the negative-dual price gradient is selected fake mass minus empirical target mass, which supplies the subsequent correction.

Identity output uses `c=mean((x-y)^2)/(2*s²)`, where `s²=mean((y-mean(y))²)` is detached, and exactly constant real batches use `s²=1`. This is a batch-normalized cost, not a claim of a fixed population OT metric.

For a declared sigmoid output `x=sigmoid(a)`, use mean Bernoulli KL from the **actual finite pre-sigmoid tensor**:

\[
 c_h(y,x)=d^{-1}\sum_k[\operatorname{softplus}(a_k)-y_ka_k+y_k\log y_k+(1-y_k)\log(1-y_k)],
 \qquad \partial c_h/\partial a_k=(\sigma(a_k)-y_k)/d.
\]

`xlogy` defines zero-endpoint entropy terms. No `logit(x)` reconstruction, clipping of G outputs, surrogate gradient, new output activation or forward/tensor change is introduced. At rounded output zero/one the prelink force can still be order one. Apply the existing observed BISPF thermal scale once to this direct G gradient; it would otherwise bypass the critic hook. No new dense-batch cooler is added. The scalar `bregman_game` diagnostic identifies the actual active payoff despite retaining the parent `loss='non_saturating'` compatibility field.

[Genevay et al.](https://proceedings.neurips.cc/paper/2016/hash/2a27b8144ac02f67687f76782a3b5d8f-Abstract.html) motivate semi-dual stochastic optimization; their convergence statements do not transfer to this regularized neural critic with one update. [Banerjee et al.](https://jmlr.org/papers/v6/banerjee05b.html) develop the Bregman/exponential-family connection. Stable prelink evaluation follows the numerical principle documented for [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html), while here BCE is a pairwise geometric cost rather than discriminator labels. [Chen et al.](https://arxiv.org/abs/1906.03471) alternate computed transport and generator regression; this candidate learns prices in the existing one-step critic schedule instead. [Choi et al.](https://proceedings.mlr.press/v267/choi25a.html) establish spurious-solution risks for semi-dual neural OT in their setting; our finite-batch game makes no general convergence or exact transport-map claim.

## Distinction from prior negatives

| Earlier mechanism | Substantive difference here |
|---|---|
| Energy/quartic/global marginal witnesses | Replace the two-player payoff; learned price correction plus a local restoring force, without an additive witness competing with unchanged BCE. |
| BCAP round5 balanced/local transport | No Hungarian blocks, balanced finite-batch coupling, component feature frame, sorted projections or local bandwidths. Hard routes are unbalanced until prices learn; this can itself fail. |
| SBNP, tethers and prior-only routing | Their policy is retained. The new force acts through both existing G and prior paths, without relabeling a latent regularizer as output fidelity. |
| Projected/filtered output forces and settling | No inverse empirical kernel, projection solver, step shrinkage or parameter-move modification. Existing direction protection is retained. |
| BCE and existing SRpGAN/Sinkhorn formulation | Critic learns target prices from a c-transform, rather than logistic real/fake labels or softplus score differences weighted by a detached coupling. No Sinkhorn iterations or entropic coefficient. |

The [BCAP round5 negatives](../../bcap-physics/round5/README.md) remain active evidence: balanced matching repaired width while losing rare/native quality, prior-only routing lost rare retention, and four-step output-kernel filtering did not repair native quality. This proposal does not claim those forces were already sufficient or silently combine their best endpoints.

## Cheap admission and runtime

[Numerical receipt](cheap-controls.json), [deterministic source](cheap_controls.py), and [public software tests](../../../../tests/test_bispf_bregman_game.py) cover:

- `a=±40` opposite-endpoint targets: finite loss40 and absolute per-row prelink gradient .5 in both directions.
- Equal mean/variance, wrong-shape 1D laws: cost .292893 at explicitly zero prices. This is a fixture at those prices, not critic-independent discrimination.
- Missing mode at zero prices: G cost **zero**, while D price gradient is `[.25,0,0,-.25]`. The control explicitly catches the blind fixed-price explanation.
- Matched constant laws: zero restoring force, zero shared-price gauge gradient, exact unit fallback without epsilon amplification.
- Detached real/prices in G, critic gauge invariance, no global Torch RNG advancement.
- Actual public scalar and selected image constructors, both payoffs consumed once on the original batches, same consumed named streams as inactive baseline, exact full-state resume, rejection before mutation for tampered capability metadata, and inactive inherited component-host compatibility.

The targeted software run, including the pre-freeze observer checks below, passes **57 tests in 1.72 seconds** (2.86 user + .67 system CPU seconds including pytest startup). All cheap-control/test commands together consumed under 30 CPU seconds, below the assigned 120-second cap. These tiny public updates are software fixtures, not task qualifications or substituted scientific cohorts.

One device-local n×m cost matrix is 16MiB at native `n=m=2048,d=2`; several such temporaries can coexist, so 16MiB is not a peak-memory claim. Complexity is O(nmd), with row minima, no n³ CPU assignment or iterative solve. Three single-thread CPU measurements including both D/G cost construction and backward were **.1314/.0670/.0922 seconds**; projecting the maximum to 7000 updates gives 919.92 seconds of cost-only CPU work, excluding the existing trainer. Image `n=32,d=64` measured .00703/.000498/.000488 seconds. CUDA/full-task runtime is unmeasured; the common frozen task caps still apply. The matrix uses a prelink/real matrix product for images and direct `cdist` for vectors, without n×m×d expansion. No extra model forward, optimizer update or sampling draw is added.

## Capability and evidence plan

The host declares `particlegan_output_link`; it does not depend on task IDs or oracle geometry. `SimpleMLPGenerator` declares identity. The actual image constructor declares sigmoid at its existing `output` layer; scoped hooks capture and verify its exact tensor relation, then are removed. Post-sigmoid `uniform_generator` averaging and unknown hosts are rejected. Actual G autocast that would break full-precision correspondence is rejected; selected ≤32-channel image hosts retain the original `bf16_channels_last` recipe declaration but the existing autocast predicate is false, so no precision delta is needed. Active checkpoints bind the capability metadata; inactive serialized packets omit every new field/key. Frozen `public_components` admission refuses a mode it does not consume.

With `collect_stats=True`, `GANTrainer.step` returns a `bregman_stats` scalar dictionary in addition to the unchanged loss. The observer reuses the already computed G cost, detached prices and that same hard argmin. It reports price minimum/maximum/population standard deviation, fraction of real **rows** selected, route entropy in nats, mean selected geometric cost, actual output range, and canonical descent-force RMS/maximum before and after the existing thermal scale. Sigmoid additionally reports actual prelink range, fraction with `|a|≥16`, exact output-endpoint fraction, and lower/upper near-saturation fractions at `x≤1e-6` / `x≥1−1e-6`. These thresholds are descriptive diagnostics, not gates, architecture changes or numerical clipping.

Canonical force means the negative derivative of the **batch-mean adversarial G payoff**, including `1/n` and `1/d` but excluding SBNP, with respect to the declared output coordinate (identity) or actual prelink coordinate (sigmoid), holding the selected route fixed. It is `(y_selected−x)/(n*d*s²)` for identity and `(y_selected−x)/(n*d)` for sigmoid, scaled by the observed thermal factor for the after-thermal values. It is **not the realized parameter update**, an inverse-kernel estimate, or a claim that the optimizer faithfully realizes that force. Selected-row fraction/entropy are not oracle mode coverage; duplicate/nearby target samples can distort them. Price spread alone does not establish dual convergence. Read their trajectories alongside the original whole-curve gates and image quality/saturation.

The observer gathers only n×d target coordinates and transfers the scalars together. It adds no cost matrix, n×m×d tensor, network forward, sampling draw, optimizer update, autograd graph or backward call. Tests compare enabled/disabled observation with exactly identical scalar loss, analytic/autograd canonical force, loss gradients, RNG states and complete public scalar/image checkpoint trajectories. It runs only when statistics are explicitly collected; inactive/ordinary recipes expose no new diagnostic key.

Prespecified interpretation for the common all-15 comparison:

1. Primary success remains all eleven formerly failing tasks passing their original whole-curve gates while retaining all four passing controls. No per-task recipe union or endpoint-only promotion.
2. Image prediction: stable prelink pullback should recover clean intensity spatial/photometric quality and avoid the almost-black blobs collapse. Persistent zero clean modes or collapse despite finite prelink force rejects sufficiency of the pullback/game. Balanced nearest counts alone do not support it.
3. Width/native prediction: explicit unbounded geometric restoring forces should reduce center-output bridges and full-law tail scatter, then sustain full covariance/native precision gates. Core improvement without full-law/stable-curve repair repeats the negative and rejects the mechanism.
4. Missing-mode prediction is conditional on learned prices. Failure to recover rare/all-mode quality rejects this fixed one-critic-step recipe; zero fixed-price cost is not evidence that the distributions match. Price oscillation, hard-route ties, real minibatch normalization and the retained BISPF penalty are declared risks.

Use the unchanged task traces and saved samples/checkpoints. If the parent publishes its already declared center census, keep it supplementary to original gates and isolate any constructor RNG; it adds no sampling draw or optimizer update. Do not expand this proposal into smoothing/coefficient/seed sweeps after seeing outcomes. Archive stdout/JSONL/tensors and paid costs outside Git; the common publisher owns the final compact receipt and actual-training GIFs.

Reproduce software/analytic checks from the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_bispf_bregman_game.py tests/test_bispf_three_theme_contract.py tests/test_bispf_fidelity_repair.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python reports/forge/first-principles-round2-20261022/fidelity/cheap_controls.py
```
