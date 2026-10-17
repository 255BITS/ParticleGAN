# Small-batch critic-memory handoff

## Question and constraints

Can an O(N) rolling signal from a moving critic guide particle birth/death
with N=16–64 and batches as small as two, without a real-sample reservoir,
raw `G(z)`/`X` geometry in the regularizer, a second classifier, or tuning to
the data domain? We used N=32, batch two, 1,600 steps for one fixed symbolic
preflight. This is a negative result for the tested mechanisms, not a proof
that all critic-guided transport is impossible.

The source E19 split-conformal isolation action has a sharper limitation:
with N rows and N/2 calibration real samples, the smallest p-value is
`1/(N/2+1)`. BH at Q=.05 needs at least
`ceil[N/(.05*(N/2+1))]` simultaneously minimal p-values before it can flag
anything. The action guard permits at most `.05*N` flagged rows. For N=16,
32, and 64, these bounds cannot both hold; the E19 isolation path therefore
cannot *act* at those table sizes. This follows from `_isolated` and
`_isolation_pick` in
`/ml2/hypergan/gan-attempts/noout-20260928/pkg-E19/particlegan/birth_death.py`.
It does not say that the separate ordinary birth/death path cannot act.

## Fixed experiment

See [SPEC.md](SPEC.md), [FLOW_SPEC.md](FLOW_SPEC.md),
[MASS_SPEC.md](MASS_SPEC.md), and [OPTIMISM_SPEC.md](OPTIMISM_SPEC.md) for the
prospective decisions. A four-category relativistic logistic GAN has 32
trainable particle distributions and one learned critic score per category.
All arms share one random stream, initial rows, critic/generator rates, and
1,600 steps. Six cases probe exact equality, an overrepresented unsupported
category, a legitimate rare category, a law shift, a frozen wrong critic,
and a function-preserving critic-feature rescaling. Only the evaluator uses
true category probabilities. Candidate signals use the current critic and
learned particle distributions. The toy can directly average each row's
four-category distribution; doing so in a real high-dimensional generator
would require a different estimator. No native performance or A2 claim
follows from this toy.

The immediate and rolling controllers clone the lowest-payoff row into the
highest-payoff row once per 16 steps. Rolling uses an O(N) EWMA with a
16-step turnover. The guarded version adds a global prequential e-process
for real-vs-fake inequality. A second experiment uses payoff-proportional
continuous-time pair reactions; a third uses continuous row probabilities
with `log(pi_i) <- log(pi_i) - .0085 L_i`. The latter has no cloning.

### Outcome leaderboard

Entries are **final total-variation distance** to the true law; lower is
better. `flow` is the magnitude-sensitive reaction. `mass` is continuous
mass under the learned critic. The oracle-mass column is an evaluator-only
diagnostic with access to the true law, never a deployable candidate.

| Case | Ordinary | Current BD | Rolling BD | Guarded rolling | Direct mass Adam | Flow | Mass | Oracle mass |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Exact null | **0.000** | 0.156 | 0.094 | 0.031 | 0.100 | 0.031 | 0.031 | 0.000 |
| Overmass | **0.021** | 0.059 | 0.273 | 0.027 | 0.390 | 0.028 | 0.025 | 0.024 |
| Rare | **0.011** | 0.146 | 0.160 | 0.015 | 0.289 | 0.052 | 0.036 | 0.013 |
| Shift | 0.142 | 0.207 | 0.171 | 0.141 | 0.157 | **0.116** | 0.119 | 0.026 |
| Frozen wrong D | **0.943** | 1.000 | 1.000 | **0.943** | 0.996 | 0.967 | 0.953 | 0.948 |
| Feature gauge | 0.023 | 0.065 | 0.217 | 0.030 | 0.153 | **0.005** | 0.028 | 0.014 |

Final TV alone hides churn. Current and rolling BD each made 100 moves in
*every* case, including exact equality. The guard cut that to one false
null move but did not validate which row to clone. Payoff-proportional flow
made only 7 null moves, yet the null ended one row (1/32) wrong. Continuous
mass also ended 1/32 wrong under learned D. On the rare case, ordinary GAN's
cumulative TV was 74.7 versus 86.7 for flow and 83.1 for mass. On the
shift, mass's final TV improved but its cumulative TV was 165.8 versus
165.4 for ordinary training; it did not recover earlier. Full final,
last-quarter, cumulative, critic-regret, row-retention, and time metrics
are in [results.json](results.json), [flow_results.json](flow_results.json),
and [mass_results.json](mass_results.json).

The exact-real/payoff flow diagnostic was nearly identical to the two-real
stream payoff, so fresh-real minibatch averaging alone did not explain the
failure here. Oracle critic payoffs improved the shift substantially. This
implicates learned critic information, but the oracle uses an artificial
`1e-12` probability floor where the real law is zero, giving very large
scores; its gains are not a practical bound. A frozen wrong D also sends
the generator gradient the wrong way, so transport alone cannot fix that
case even when its own payoff is oracle-derived.

### Moving-critic lag check

After Astra reviewed mass flow, [lag_diagnostic.py](lag_diagnostic.py)
replayed it without changing any update and exactly reproduced saved TV.
For each step it compared current centred per-row payoff against
`2 L_t - L_(t-1)`, using exact toy payoffs only as an evaluator. Ratios of
integrated direction error (extrapolated/current) were:

| Case | Online two-real ratio | Exact-real, learned-D ratio |
|:--|--:|--:|
| Exact null | 1.0016 | 1.0019 |
| Overmass | 0.9999 | 0.9999 |
| Rare | 0.9975 | 0.9973 |
| Shift | 1.0034 | 0.9951 |
| Frozen wrong D | 1.0000 | 1.0000 |
| Feature gauge | 1.0000 | 1.0000 |

The predeclared go/no-go in [OPTIMISM_SPEC.md](OPTIMISM_SPEC.md) failed:
the available online extrapolation increased null and shift error. The
near-one ratios do not establish a useful predictable critic lag. The
oracle-direction metric is numerically dominated in some cases by the
evaluator's near-infinite unsupported-category score, another reason not
to promote this idea on these data. No optimistic controller was trained.

## Interpretation and next move

The tested O(N) rolling payoff is a memory of *different critics and moving
particles*. It worsened the ordinary baseline, and the global e-process
mostly helped by suppressing actions. A global test of `p_real != p_fake`
does not tell which particle has inadequate support. Continuous mass avoids
cloning damage but still acts on an untrustworthy direction. A method that
receives only an uninformative or wrong critic cannot in general distinguish
two real laws that the critic maps to the same signal; the frozen-D case
illustrates this limit.

**Recommendation:** keep ordinary GAN as the small-batch control and do not
promote any of these transport arms. The next investigation should improve
and *measure the critic's current row-level directional information* before
adding a controller. Use prospective prequential/held-out critic metrics and
rare-support/shift cases; require lower cumulative TV than ordinary training
while preserving exact-null and rare rows. A short memory may help D's own
training, but these results do not justify storing old D payoffs or training
an auxiliary classifier on its moving features. The symbolic toy and one
fixed stream are a mechanism filter; a successful new idea would still need
native 16/32/64-row tests and A2 source review.

Reproduce from this directory, in order, with a Python environment that has
NumPy:

```sh
python -u toy.py > run.log 2>&1
python -u flow.py > flow.log 2>&1
python -u mass_flow.py > mass.log 2>&1
python -u lag_diagnostic.py > lag.log 2>&1
```

The logs are concise and can be tailed while each job runs. They are ignored
by Git; the JSON metrics and full trajectories are committed. No seed sweep
or post-result rate tuning was performed.
