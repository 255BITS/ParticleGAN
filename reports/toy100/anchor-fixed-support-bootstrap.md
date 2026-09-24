# Confirmed fixed support for a stationary target

A data-derived support model with fixed identities is a legitimate simpler
alternative to perpetual group discovery for the present fixed-target task.
Conditional on complete and correctly grouped initial support, it removes
the identified minibatch-omission failure and the continuing possibility of
creating identities from isolated tail groups. It does not establish those
initial conditions without evidence, and it does not turn finite statistical
ambiguity into an impossibility claim about GAN learning.

The bootstrap estimates the data model, not the generator. To keep acquisition
honest, it must use only explicitly counted real samples, never an acquired
G checkpoint or true component centers. G/prior retain their original cold
initialization. Bootstrap data, work, and RNG effects are recorded, and cold
acquisition, own-state holding, and same-target model-error response remain
required tests. This remains an explicit additional support objective; it
does not prove the unchanged adversarial objective or production mass/spread
accuracy gates.

## A minimal source-reviewable confirmation rule

Use two distinct, independently drawn native-sized real banks A and B.
Apply the same declared grouping function to each. Let their centroid arrays
be `(a_i)` and `(b_j)`, with inferred counts M_A and M_B. No configured group
count or true-center radius enters the rule.

Require:

1. Both arrays are finite and `M_A=M_B=M≥2`.
2. Their within-array minimum separations `s_A` and `s_B` are positive.
3. Nearest-centroid maps A→B and B→A are unique, reciprocal, and form a full
   bijection π.
4. The strict margin is positive:

   `min(s_A,s_B) − 2 max_i ||a_i−b_π(i)|| > 0`.

This margin certifies unique geometric correspondence. Write the maximum
paired distance as d. For any competing j, the triangle inequality gives
`||a_i−b_j||≥s_B−d>d≥||a_i−b_π(i)||`; the reverse direction follows from
s_A. Therefore every selected pair is strictly nearest in both directions
and this pairing is also the unique minimum-sum assignment. The certificate
concerns correspondence, not whether both groupings found the true mixture.

For a first seven-group bank followed by an eight-group bank, confirmation
stays unresolved. Replace the pending proposal with the current bank; only
a subsequent matching eight-group bank can confirm that pair. Pool exactly
the two confirmed banks' sufficient statistics, preserving the first bank's
identity order. Do not freeze seven identities, silently combine disagreeing
models, or treat a repeated copy of one bank as independent confirmation.
Single-group, duplicate-center, tie, and nonpositive-margin cases lack this
certificate and remain explicitly unresolved. A predeclared maximum data
budget can return unresolved rather than an unqualified support model.

Bank hashes and IDs detect replay mistakes; actual independence comes from
the source-bound distinct sampler draws without a stream rewind. Different
hashes do not prove independence. Conversely identical arrays can arise
under independent discrete sampling, so rejecting identical bytes is a
declared conservative rule, not a theorem that such samples are dependent.

## Coverage risk and what confirmation cannot prove

For a mixture with K components and probabilities p_k, the probability that
at least one component is unobserved in B independent real samples obeys

`P(incomplete samples) ≤ Σ_k (1−p_k)^B`.

If grouping is correct on all observed components, false confirmation due
solely to a component absent from both banks requires that component to be
absent from 2B samples. Thus

`P(confirm AND incomplete) ≤ Σ_k (1−p_k)^(2B)`.

This is an unconditional false-confirmation bound under the stated grouping
premise. A probability conditional on confirmation would also need its
denominator. For up to L adaptively inspected adjacent-bank pairs, a union
bound is L times the single-pair bound; no independence between those pair
events is needed. Do not apply a one-pair bound to unlimited attempts.

The actual uniform samplers give the following **offline** values. Their K
is used only for this analysis, not by the confirmation rule.

| Fixed sampler | One bank: any component absent | Same omission across two banks |
|---|---:|---:|
| Ring, K=8, B=128 | ≤3.0208e−7 | ≤1.1406e−14 |
| Production, K=100, B=2,048 | ≤1.1505e−7 | ≤1.3236e−16 |

These bounds do not include incorrect clustering. The existing offline
production first-bank receipt observes all 100 groups with pure MST clusters;
that is evidence about the fixed draws, not a guarantee for every new bank.
Two matching incomplete banks also cannot rule out an arbitrarily rare
unseen component in a distribution with unknown weights. A distribution-free
complete-support claim would require more assumptions, such as a positive
minimum component mass and an identifiable grouping model. The stationary
bootstrap need not claim that stronger result.

Unlike current-bank-only deletion or unrestricted online births, bootstrap
misidentification is a one-time model-selection risk. Correct fixed identities
are not removed when later data omit a component and are not enlarged by
every later tail cluster. The subsequent centroid estimator still needs its
own accuracy assumptions.

## Fixed identities, continuing statistics, full model correction

The selected stationary assignment uses the **fixed pooled bootstrap reference
centers** to label each subsequent native real sample by nearest reference.
Accept that assignment only if its distance is strictly less than half the
fixed minimum reference separation. These open acceptance balls are disjoint;
every accepted sample therefore has a unique identity. Reject points outside
all balls. Update the accepted identity's ordered sums, counts, and
squared-norm sums; create or delete no identities and never rerun MST after
confirmation. Keep references and the fixed radius separate from the resulting
running centroids and serialize them. Record per-bank accepted and rejected
counts. This fixes the data partition and avoids an additional moving-boundary
online k-means feedback mechanism.

For independent stationary data, positive cell probability, and finite
moments, these per-cell estimates target the conditional means of the fixed
acceptance balls. They include any other mixture components' tails that enter
the ball and exclude samples outside it. Those truncated means are close to
the mixture centers only under suitable separation and bootstrap accuracy;
Gaussian overlap does not give exact latent labels. The output-invariance
argument uses the resulting centroid-error bound and the existing fit/rest
conditions. It does not prove parameter boundedness or every future finite
evaluation result.

Running-mean weights express accumulation of data evidence. They must not
multiply the MM target, GN fit, or accepted model-error correction. At the
same cached centers, proportionally scaling sums/counts leaves the objective
and full repair of a perturbed G unchanged. Explicitly test that count
invariance, then test an actual same-target model perturbation from a mature
learner state. No nominal LR claim substitutes for those tests. Target shifts
are a different task and are not an obligation of this fixed-support policy.
If a bank has no accepted samples, retain the data statistics while still
allowing full G-error correction from the remembered support targets.

The next cheap checks are therefore the seven/eight/eight bootstrap sequence,
same-support confirmation under distinct native banks, permanent retention
under a later omitted-component bank, no birth under a later isolated outlier,
and full count-invariant model-error correction. They are diagnostics before
any new native training gate, not proof of an all-time learner.

This note implements no helper and runs no training.
