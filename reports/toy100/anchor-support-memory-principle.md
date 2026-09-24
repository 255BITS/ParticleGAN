# Remembering stationary support: conditional association and invariance

The current-bank-only objective has a specific failure: one real minibatch
that omits a population component deletes that component from its target
set. The actual neural update-2,401 counterexample moves from eight modes
to seven although the dataset is unchanged. A persistent data-support
memory addresses this failure by treating absence as no observation.
It is an explicit stateful estimator, not a claim that every bank reveals
the full distribution.

This note analyzes the proposed rule without changing its implementation:
infer current groups by the existing MST rule; match each current group
to its nearest **pre-bank** cached centroid only when closer than half
the minimum cached separation; add matched samples to running sums/counts;
retain unobserved cached groups; append unmatched current groups. Multiple
current groups can match one existing group. Newly appended groups do not
participate in matching other groups from that same bank.

## Association with an existing group

Let true centers be μ_i. Suppose every cached center c_i represents one
distinct true center and `||c_i−μ_i||≤e`. Let

`s = min_{i≠j} ||c_i−c_j|| > 0`, `r = s/2`.

Assume a current group is pure and its centroid b represents cached center
μ_i with `||b−μ_i||≤q`. If

`e+q < s/2`,                                                   (1)

then `||b−c_i||≤e+q<r`. For every other cache center,
`||b−c_j||≥s−||b−c_i||>r`. Thus the match is unique, accepted, and correct.
This proof permits several pure current groups from the same true component:
all match the same cached identity. It does not assume every individual
Gaussian sample lies within q; the bound concerns the corresponding pure
group centroid.

For cache count n and current pure groups with sizes m_j, the updated
centroid satisfies

`e_new ≤ [n e + Σ_j m_j q_j] / [n + Σ_j m_j]`.                 (2)

In particular, a common error bound ε is preserved by the weighted mean.
An absent cached group has no update to either sum or count and is retained
bitwise. Matching must use a frozen pre-bank cache to make (1) apply to all
groups under the same radius; this is the declared helper behavior.

## Discovery is a separate condition

For an uncached true center μ_new, define its distance to the represented
true set by `d_new=min_i ||μ_new−μ_i||`. A sufficient correct-append condition
is

`d_new − e − q > s/2`.                                        (3)

Then every current centroid for that new component lies outside the
matching radius. To append exactly one new identity, require at most one
inferred current group per uncached component in this bank. The helper
otherwise appends duplicate new groups separately; it does not coalesce
them against groups added earlier in the same call.

Half the cached separation is **not** universally safe for discovery.
With exact cached centers `{0,4}`, the radius is 2, and a genuinely new
center at 1 is incorrectly absorbed into the center at 0. There is no
observation noise in this counterexample. The minimum separation among a
proper subset of true centers can exceed the global minimum separation.
Consequently an upper bound `s≤Δ+2ε` is valid only if the cache includes
a globally closest true-center pair, particularly when it is complete.

Once that pair is represented and all errors are at most ε, the stronger
condition `ε<Δ/6` gives

`2ε < Δ/2−ε ≤ s/2 ≤ Δ/2+ε < Δ−2ε`.

It simultaneously guarantees correct existing-group association and correct
append of distinct new groups, provided those new groups are not split
within the bank. The seven-of-eight ring cache retains a closest pair,
so this condition can cover the intended omission/rediscovery diagnostic.
An arbitrary two-group cache may not. The one-cache-group bootstrap using
the new bank's MST threshold needs a separate argument and is not covered.

For full discovery of a finite stationary target, each uncached component
must eventually supply a correctly inferred group satisfying (3). Positive
component probability gives eventual *sampling* with probability one under
independent draws; it does not by itself prove correct MST grouping,
association, or discovery. Initialization errors, mixed groups, far-tail
false groups, and duplicate unseen groups can persist because the rule
does not merge or remove established identities. For example, a spurious
far-out MST group is appended just as a real new component would be. These
are explicit limits of the current novelty rule, not evidence that support
memory is impossible.

## Revised observed-width association

The next declared rule uses an observed within-group RMS width rather than
cache separation alone. If a group has sum S, squared-norm sum Q, and count
n, its width is

`w = sqrt(Q/n − ||S/n||²)`.

For current group j and its nearest cached group i, the proposed radius is

`r_ij = min(w_i+w_j, s/2)` for at least two cached groups,

and `r_ij=w_i+w_j` for one cached group. Match when distance is at most this
radius. The exact zero-width, same-center Dirac case consequently matches;
the exact `{0,4}` cache plus new center 1 does not.

This is an observed **support-scale overlap rule**, not a confidence
interval for the centroid. Raw within-group spread does not shrink as
1/√n. A sufficient existing-group condition is

`e_i+q_j ≤ w_i+w_j` and `e_i+q_j < s/2`.                     (6)

The second inequality ensures that the correct group is uniquely nearest;
the first ensures that its observed-width threshold accepts it. At the
half-separation equality boundary, deterministic tie breaking does not
imply a unique correct association. For discovery, require the new group's
distance to the nearest cached group to exceed that group's radius. A
stronger population-level sufficient condition is

`||μ_new−μ_i||−e_i−q_j > r_ij` for every cached i.             (7)

For a singleton cache, the separation argument is unavailable; (6)'s width
condition still suffices for an already correctly identified same group,
and (7) suffices for a distinct new group. Neither is automatically certified
by the observed widths. Two noisy singleton observations from the same true
component have zero empirical RMS and can have different means, so they
would incorrectly create separate identities. A far-out pure component
subgroup can similarly exceed its cached overlap radius. A statistical
confidence claim would need explicit tail assumptions and control of the
repeated grouping/association decisions; empirical RMS alone supplies neither.

The weighted-mean identity and complete-cache output induction below are
unchanged when association is correct. The new width rule repairs a concrete
failure of the separation-only rule; it does not certify every future novel
group. No source or training change is made by this note.

## Change to the output invariant

Once the cache is complete and correct, let E bound all cached-centroid
errors. Use **all cached groups**, including absent groups, in the existing
distinct-anchor objective. The [output invariant](anchor-invariant-region.md)
then uses E in place of the current-bank error ε. With N>K, clean cloud in
the covered radius-R region, minimum true separation Δ, numerical objective
error η, comparison tolerance τ, and eligible fit error δ, its conditions are

`Δ > (1+√K)(R+E)`,                                           (4)

`E + √[N(2δ²+τ+2η)] ≤ R`.                                   (5)

The existing failed-fit → pre-G rest guard preserves the region when fitting
does not converge. If (1) and (2) keep the memory correct and bounded, then
(4)–(5) yield conditional output invariance even when arbitrary subsets of
components are absent from individual minibatches. Every objective still
contains the absent components, so omission no longer changes the target's
support cardinality.

This is a two-part induction: first preserve the memory's identities/error
bound, then preserve the generated output region. It does not require every
current bank to contain all K components. It still requires sufficient
grouping/association accuracy; a newly appended false group changes K and
can invalidate N>K or the region premises. During initial discovery, adding
a true previously unknown center is an acquisition event. The complete-cache
local holding theorem does not guarantee that one MM/neural update reaches
the expanded covered region. The fixed-center free-output result and actual
cold acquisition tests remain distinct evidence.

Uniformly bounded errors of all future Gaussian bank centroids are not
established. Correctly assigned cumulative means can reduce centroid noise,
but the helper's novel-group decisions still inspect each current bank.
Neither this memory lemma nor the earlier output theorem proves an
unconditional pathwise guarantee for infinitely many unbounded Gaussian
batches. They also do not bound internal parameters, Adam state, or
Jacobian conditioning.

## Why the running mean is not learning-rate decay

The center update can be written
`c_new=c_old+[m/(n+m)](b−c_old)`. The factor is the exact weighting of new
evidence in a stationary empirical mean. It is not a gain multiplying the
G/prior correction: the same output target, MM rule, GN budget, convergence
tolerance, and native Adam rates are used regardless of n. At fixed cached
centers, multiplying sums and counts by a common integer leaves the target
and correction for the same model error unchanged. A perturbed model can
therefore require the same full output repair after many observations.

This distinction should be tested, not inferred from nominal rates: hold
the memory's centers fixed, perturb only G/prior, and compare target/proposal
behavior with proportionally scaled sums/counts. Then use an actual matched
same-target model-error response from the old learner state. If the target
distribution itself changes, cumulative counts can slow centroid adaptation;
that is a separate scenario, outside this stationary-target rule.

## Exact learner-state and resume contract

The current host snapshot has complete models, Adam, EMA, RNG streams and
NoisePolicy state, but no recorder memory. A memory method must extend the
learner-state contract instead of relying on that host snapshot alone.

Save a versioned outer envelope containing:

1. The existing complete host snapshot, including all noise counters/history,
   absolute update position, and all four RNG streams.
2. The adapter method/schema version and the ordered memory sufficient
   statistics: exact float64 sum tensors, float64 squared-norm sums for the
   revised RMS rule, and positive integer counts.
   Preserve group order; nearest-neighbor and Hungarian ties use that order.
   Saving means alone is insufficient to reproduce future weighted updates.
3. An observed-bank count and the last committed absolute update, or
   equivalent receipt-backed guards, to detect duplicate/missing observations.
   Save any future persistent bootstrap, threshold, or association state too.
4. The explicit snapshot stage, active/inactive correction scope, and source
   hashes for grouping, memory update, controller, and state serialization.

Observe the native D bank exactly once per completed outer update. The
three same-batch replay phases, Jacobian trials, live evaluation, and EMA
evaluation must not add observations. Fitting failure may rest the model
while committing the valid real-data observation; that transaction policy
must be declared and serialized. Fatal incomplete-update snapshots should
be labeled non-resumable unless their stage has an explicit recovery rule.

At an ordinary boundary, memory is committed before EMA/checkpoint and the
next bank has not been observed. Restore the host before the next `set_step`
as today, and restore the recorder's memory before any next observation.
Do not reconstruct memory from generated supports or the next minibatch.
Warm forks must clone memory with the model/optimizer/RNG state. A legacy
snapshot without memory can be used only for an explicitly declared new
bootstrap diagnostic, not an exact continuation of the memory algorithm.
The conditional trajectory and disabled control must retain their declared
inactive/empty-memory behavior.

Hash the full envelope. The reviewed warm/hold diagnostic reconciliation may
adjust only its two proven host evaluation counters; it must never omit or
normalize the memory. Minimal tests are uninterrupted versus split-resume
exact equality of the full envelope and per-update memory/target/correction
receipts, once-per-update observation accounting, source/method mismatch
rejection, and rejection of invalid counts/nonfinite sums. Preserving an
omitted group after resume should be part of that check.

This note changes no candidate source and launches no training.
