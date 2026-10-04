# Saved N11 word trajectories: original rate versus half-base

The half-base run changed the trajectory but did not recover the full word
question. Its accepted result is **COMPLETE / original numerical FAIL**. The
older N11 run remains **INVALID / accepted numerical UNAVAILABLE**; its saved
measurements below are observations, not a re-certified FAIL. Neither fills
unrun cells or confers ordinary Atlas, default or speed credit.

This comparison aligns all 24 actual saved updates, 834 through 20,001. It
verifies 99 consumed input identities before and after reading, including both
sets of 24 NPZs and the earlier 46-pin source-law proof. NumPy reads only saved
arrays with `allow_pickle=False`. No checkpoint is deserialized; no model,
forward, restore, sample, optimizer, official scorer or CUDA context is used.

## What is matched

The complete applied Recipes differ only in `lr`: .0053125 versus .00265625.
Both retain prior multiplier 1.5 and critic multiplier 1. This halves nominal
G/E/table/noise/D rates; it does not preserve endogenous applied rates or isolate
the generator. Recorded named initialization is exactly equal. Seed 0, N11 raw
2D ParticlePrior, batch 256, all original G/E/joint-D degrees, original joint
RpGAN, gates, 20,001 updates, 24 reads of 1,024 samples, and terminal-five cadence
are matched. Both observe selected `fast` G/E/prior, with output noise off and
actual DV12 retained. Every recorded G/E/prior/D owner completed 20,001 updates.

The frozen model, loss, public policy, continuous sampler, raw-prior and scorer
source bytes are equal. ASTs of the complete joint callback, words-only noise,
fake generation, generator objective and training step are equal. The newer
producer adds explicit rate/cohort binding and metadata labels; its observer
math and training path remain the same. The health-guard repair changes evidence
acceptance, not these word measurements. Exact source hashes are in
[comparison-final.json](comparison-final.json).

Original origin `fb7acc775b3a1a6184d36b55e035b9da04531492`, digest
`f380eed990931bacb205e6537beaf387fdbb676ffc97b6f32f19ff903ae1cfed`.
Half-base origin `f9f7ed9d7a06c48d4ec56999107658983d7e8efc`, digest
`5995590d3c303207c664fe3b0ba7dc1e09dd3da7a90abf87086634c789175026`.

## Coverage and inverse are different observations

The generated-mode counts below are copied from the original recorded,
confidence-qualified measurements. Individual inverse counts are separately
derived argmax display diagnostics. The all-five inverse flag is zero at all
24 saved checks in both runs; that does not mean zero individual reconstructions.

| Saved update | Original: recorded modes; individual inverse | Half-base: recorded modes; individual inverse |
|---|---|---|
| 834 | 2/5; 2/5 | 2/5; 2/5 |
| 6,667 | 3/5; 3/5 | 4/5; 4/5 |
| 10,001 | 5/5; 1/5 | 3/5; 2/5 |
| 11,668 | 3/5; 3/5 | 4/5; 4/5 |
| 15,835 | 1/5; 1/5 | 4/5; 2/5 |
| 17,501 | 2/5; 2/5 | 1/5; 2/5 |
| 20,001 | 2/5; 2/5 | 2/5; 1/5 |

Original recorded five-mode coverage occurred only at update 10,001. Half-base
never reached five modes. It nevertheless briefly reconstructed four individual
words, versus the original run's maximum three. Current accepted independent
grading records zero passing reads and a zero passing suffix. An earlier good
word or checkpoint cannot replace the original terminal-five requirements.

At update 20,001, original recorded quality is .9248046875 and TV .6; half-base
quality is .546875 and TV .622265625. The current generated argmax spellings are
464 apple, 182 grape and 378 melon, but recorded qualified apple mass is zero.
The best minimum correct-token probability of any saved apple row is only
.628198683, below the original .90 confidence rule. Lemon and berry are absent
even as canonical argmax spellings. Argmax counts must not replace the recorded
quality, mode masses or numerical decisions.

All six tokens, including underscore padding, enter the paired probability
measurement. These are the terminal reconstructions and minimum probabilities
of the *correct target* tokens, not confidence in the predicted spelling:

| Known input | Original decoded inverse; minimum correct-token probability | Half-base decoded inverse; minimum correct-token probability |
|---|---|---|
| apple | `berre_`; 0 | `melon_`; 1.76242e-12 |
| grape | `grape_`; 1 | `gpape_`; .0802794 |
| lemon | `berry_`; 0 | `gpape_`; 8.79188e-10 |
| melon | `berry_`; 0 | `melon_`; .982160 |
| berry | `berry_`; 1 | `melon_`; 6.79940e-7 |

The report retains complete correct-token probability vectors at updates 834,
10,001, 15,835 and 20,001, and per-word minimum/mean descriptors at every check.
Confident wrong tokens, including exact saved zeros in the original run, are
finite outputs. Their source scorer clips probabilities for its diagnostic NLL;
this comparison calls no scorer and assigns no replacement grade.

## Code geometry and relative noise

All 24 prior tables in both runs have eleven distinct rows, and all encoded
five-word queries are distinct. Distinctness is insufficient to establish
useful separation. Half-base ends with much more compressed code geometry:

| Terminal descriptor | Original | Half-base |
|---|---|---|
| Prior axis standard deviations | (.404529, .602379) | (.136372, .056435) |
| Closest pair of encoded words | lemon/berry, .326228 | apple/berry, .0201052 |
| Median encoded-word distance | 1.00508 | .158209 |
| Closest prior-row distance | .142378 | .000890051 |
| Saved inverse DV12 displacement RMS | .0954885 | .0327382 |
| Inverse RMS / closest encoded-word distance | .292705 | 1.62834 |
| Saved generated-code DV12 displacement RMS | .152399 | .0150122 |

These are saved geometric descriptors, not new gate thresholds. Half-base has
less absolute inverse perturbation but more relative to its closest encoded
words. Its retained perturbed apple and berry inverse queries are each nearest
to the original melon encoder code; both actually decode as melon. Original
perturbed queries remain nearest to their own encoded labels, yet three still
reconstruct incorrectly. Thus neither geometric overlap nor DV12 alone explains
both failures. Same-state DV12-off reconstructions were not retained.

The learned *training* word-coordinate sigma recorded at both endpoints is
.0289999992. Its analytic root-mean-square L2 magnitude over 168 coordinates is
.375883, or .153454 of the canonical one-hot word's L2 norm. This is a scale
calculation, not a new random draw. Output noise is off in all saved evaluations.
Per-checkpoint training output-sigma history is unavailable; the report does
not assume that the endpoint amplitude held throughout training. The final
recorded latent bandwidths and last two training perturbations are kept
separately from the saved observation displacements.

Every saved generated effective code differs from its raw sampled row in both
runs (24,576 / 24,576); none equals any corresponding deterministic E(word).
Every saved inverse effective query also differs from its E(word) input
(120 / 120). The earlier source-law report explains the discrete real encoded
joint versus perturbed fake joint and fake-only word noise. This persists under
half-base. It does not prove that the finite gates are impossible, that the
finite critic exploits this distinction, or that noise caused the failures.

## One proposed causal repair, not an implemented candidate

Use a **new explicitly named objective/family variant** that adds paired
six-token reconstruction NLL on all five known words, through the existing
stochastic public `G(E(word))` path, with fixed `reconstruction_weight=1`.
Retain the current half-base rates, original joint RpGAN and regularizers,
architecture, free E, N11/raw prior, initialization, all enabled policy owners,
same-code generation callback, DV12 and words-only noise laws, observation law,
seed, full horizon and every original numerical gate.

The current caller's `generator_objective` gives E gradients through the real
joint critic, and G/table gradients through the generated joint. It never
computes an inverse loss; `reconstruction_weight=1` is presently unused by
this caller. The proposed term supplies a direct word → E → public generator
→ correct-token-probability gradient that bypasses D. Its stochastic query must
retain DV12, not silently train or evaluate a noiseless replacement.

For training, the proposed path must use the actual owned differentiable G/E
and public `UpdatePolicy.generate` within the generator update, with `sigma=0`
for this inverse term; it must not backpropagate into a copied inference
snapshot. Stable log probabilities must come from the same generator logits,
with forward/gradient parity proved. Logging already underflowed softmax zeros
or hard-clamping probabilities can lose the intended gradient. A separately
declared auxiliary training-noise stream must retain the original fake and
evaluation stream identities rather than silently shifting their draws.

This is the one objective change, and it needs fresh source/gradient/RNG/checkpoint,
resource and capacity review before execution. It is not row supervision or
unseen-word generalization, and it is not credit for either existing variant.

The falsifiable prediction is persistent improvement of the original all-five
paired exactness and minimum correct-token probability at unchanged terminal
checks. Generation coverage, quality and TV must still pass independently.
No persistent inverse improvement, or continued full original gate failure,
rejects this as a complete repair. A direct inverse term could leave prior
coverage unresolved or upset the joint game; compressed neighborhoods, moving
support, discriminator behavior and asymmetric noisy joints remain competing
explanations. This proposal does not repair or prove exact joint-law equality,
establish stochastic capacity, identify DV12 as causal, or imply a default.

This one failed global rate reduction **does not show that LR is irrelevant**.
Its different geometry, mode history and individual inverse peak demonstrate
changed dynamics. No second rate candidate, training, source preparation,
admission or unchanged replay is performed here.

## Reproduce only the saved-array comparison

The raw archives are LOCAL_ONLY. [inputs.json](inputs.json) pins the old diagnosis,
old source-law review and new immutable card/passive publication. All consumed
file identities and all 24 aligned descriptors are in
[comparison-final.json](comparison-final.json). The reader refuses missing
or changed pins; it never hydrates or regenerates missing evidence.

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /ml2/hypergan/.venvs/particlegan-develop-integration/bin/python \
  compare_retained.py --inputs inputs.json --output fresh-comparison.json
```

The command runs from this directory. Any compatible NumPy environment can
inspect the saved arrays; that reader runtime is not a scientific speed or
qualification comparison. Original artifacts and verdicts remain unchanged.
