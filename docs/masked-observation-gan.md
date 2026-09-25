# A discriminator over whatever was observed

Status: proposed experiment on feature/masked-observation-gan; no training runs.

## Central idea

Learn one complete generator from heterogeneous observations. Each record supplies
what was measured and its value. The discriminator makes its ordinary real/fake
judgment on that observation, without requiring a reconstruction target or a
separate task for each missingness pattern.

A sample-level discriminator accumulates distributional evidence through shared
parameters trained on many records. Being sample-bound does not mean it ignores
population statistics. The proposed change is its observation interface.

Two hypotheses: many partial records jointly constrain a useful complete model;
a small full reference set can select structure those records leave ambiguous.
The latter captures "pointing at the target distribution": partial data provide
much of the statistical detail, while scarce complete examples guide the remaining
structure. This is a sample-efficiency hypothesis, not a guarantee for arbitrary
high-dimensional targets.

## Formulation

A record is (T, y), with y = T(x). T can select coordinates, crop an array, apply a
known linear projection/downsampling operation, or be the identity for full data.

```text
real score: D(T, y)
fake score: D(T, T(G(z)))
```

Use the same T for real and fake, with independent latent draws. G produces full
samples without receiving T. Gradients pass through the observation operation;
shared generator parameters accumulate constraints from different observations.

For example, a vanilla logistic objective (D denotes a probability) is:

```text
max_D E_(T,y)[log D(T,y)] + E_(T,y),z[log(1-D(T,T(G(z))))]
min_G -E_(T,y),z[log D(T,T(G(z)))]
```

Implement using logits and stable softplus losses. Other existing GAN losses can
use this same interface. This is the AmbientGAN principle with mixed observation
types, including scarce complete observations.

For coordinate masks, start with one shared MLP over [m*x, m]. The mask distinguishes
an observed zero from missingness. D recognizes the view, but the view itself gives
no real/fake clue because both sides match. Hiding T can allow different per-view
errors to cancel in the aggregate, unless the observation encoding identifies T.

A later architecture could encode a variable set of (coordinate, value) tokens
and combine them into one score. It must allow interactions: merely summing scores
computed independently for each coordinate cannot learn joint dependence. Projection
or crop metadata likewise specifies what was measured. This is an architecture
ablation, not a prerequisite for the first 2D toy.

Initially assume measurement type is independent of sample content, and all sources
are compatible observations of the same target. Different source populations or
missingness depending on unseen values require an expanded observation model.

## General 100-Gaussian toy

Reuse the current 10-by-10 grid with coordinates {-4.5,...,4.5} and isotropic Gaussian
component standard deviation 0.03. Keep centers, labels, and target formulas out of
G and D; they are only for synthetic data creation and evaluation.

1. Uniform 100 Gaussians: mechanics check matching the existing benchmark.
2. Weighted 100 Gaussians: primary test of learning joint structure, using
   w[i,j] = (1 + a * (-1)^(i+j)) / 100, initially a = 0.8.

All 100 modes remain: alternating weights are 0.018 and 0.002. Every row and column
sums to 0.1. Targets a = 0, +0.8, and -0.8 therefore have identical x-only and y-only
distributions. The uniform target factorizes, so reproducing that grid from partial
views would not demonstrate learned dependence. The weighted version exposes this.
Do not parameterize G or D with the alternating pattern: the toy tests whether sparse
full references can convey shared structure to generic networks.

Create a fixed partial pool of 10,000 records, half x-only and half y-only, plus a
separate fixed pool of complete references (anchors). Start with 128 anchors; later
compare 0, 32, and 512, with nested anchor subsets and a fixed partial pool. Use one
fixed seed and no seed-only experiments.

Store only visible values, masks, and full anchors in the incomplete-data trainer.
Keep masks fixed across epochs. Do not resample clean target data during training
or remask hidden source vectors, which would increase the information budget.
The oracle separately receives full source vectors. Evaluate on independent complete
data and do not use hidden target knowledge to choose training checkpoints.

## Sparse references and sampling

Initially draw half each batch from anchors, one quarter x-only, and one quarter
y-only; with zero anchors use equal partial-view counts. Report unique records and
reuse counts. Natural-frequency sampling is a later targeted ablation, because
rare full observations may otherwise contribute too few updates.

Positive weights on compatible view objectives preserve their common exact solution
at population level. In finite training, oversampling anchors changes the compromise
and can amplify memorization. Reweighting here assumes view assignment independent
of content; it does not correct selection bias automatically.

## Comparisons

Use the existing MLP building blocks and one fixed GAN recipe/prior. Extend D for
mask input without simultaneously sweeping priors, losses, or network sizes.
Existing Fourier value features can be retained with mask metadata appended.

| Method | Training evidence | Question |
| --- | --- | --- |
| Naive zero-filled GAN | Partial pool + 128 anchors, no observation simulation | Does it learn holes? |
| Anchor-only GAN | The same 128 full references | What can scarce complete data achieve alone? |
| Observation GAN, no anchors | Partial pool only | What marginals and assumed coupling emerge? |
| Observation GAN + anchors | Partial pool + 128 anchors | Can partial evidence and sparse guidance combine? |
| Full-data oracle GAN | Complete versions of the same source records | Reference with more information |

Match generator/discriminator update budgets. The oracle is a reference, not a
guaranteed optimization upper bound. Check mechanics on uniform data, then compare
methods on the weighted target. Vary anchor counts afterward, comparing against
anchor-only training at each count.

Analytically check normalization and identical coordinate marginals for a = 0 and
+/-0.8. This establishes zero-anchor ambiguity without optimization or seed sweeps.
A subsequent target-switch experiment can keep the same partial pool and replace
only the anchors with draws from a = -0.8. Does the learned joint mass follow them?

## Evaluation and reporting

Evaluate complete G outputs with the existing sliced Wasserstein-1 metric using
fixed evaluation projections. Also report separate x/y Wasserstein distances,
10-by-10 mode histogram total variation from actual target weights, mode coverage,
and within-mode spread/high-quality fraction. Do not score weighted targets against
uniform mode frequencies.

Track alternating-mode mass contrast C = sum_ij (-1)^(i+j) * estimated_w[i,j].
Its target is +0.8 (or -0.8 for target-switch), versus 0 for the uniform product.
Contrast alone is insufficient: a collapsed model can match it. Interpret alongside
histogram error, coverage, and quality. Show target, anchors, and generated samples
with identical-axis scatter plots and mode heatmaps.

Monitor nearest-anchor distances and generated within-mode spread against held-out
data for memorization diagnostics; no single distance establishes its absence.
Rank a leaderboard by joint sliced Wasserstein-1, retaining histogram error,
marginal errors, contrast, coverage, quality, and unique anchor count as columns.
Success means better joint held-out fit than anchor-only training at the same anchor
budget, including correct joint weights rather than only accurate marginals.
One-seed rankings are exploratory, not robustness evidence.

Follow the existing experiment runner's config and summary contracts. Flush metrics
to stdout and per-run log.txt; use results/masked_observation/PIPELINE.log for
aggregate progress and tail -F. Summarize the completed leaderboard, explain what
improved or failed, and recommend the next experiment.

## Later extensions

- Mix in known 1D projections at different angles: these constrain more than axis
  marginals, though finitely many angles do not identify an unrestricted joint law.
- Extend to longer arrays with coordinate/block observations.
- Compare a mask-conditioned MLP with a shared observed-token discriminator.
- Test additional deletion only on visible coordinates, using the same effective
  mask on both sides. It may regularize D or discard valuable joint evidence.

## References

- [AmbientGAN](https://www.cs.utexas.edu/~ecprice/papers/ambientgan.pdf): compares
  real measurements with simulated measurements; recovery depends on identifiability.
- [MisGAN](https://arxiv.org/abs/1902.09599): adversarial modeling of complete data
  and missingness.
- [Weakly supervised multimodal learning](https://arxiv.org/abs/1802.05335): related
  framing for partially paired observations.
