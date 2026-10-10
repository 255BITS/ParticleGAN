# Prospective follow-up: magnitude-sensitive row reaction

Written after the fixed [first toy](SPEC.md) showed that unconditional
highest-to-lowest cloning performs one move at every 16-step opportunity,
including under an exact null. Astra reviewed that outcome before this design.
The first toy's cases, N=32, batch size two, 1,600 updates, initial state,
real/fake quantiles, D/G optimizer settings, and evaluation metrics remain
unchanged. No reaction multiplier or gate is tuned to those results.

For current pre-D-update row generator payoffs `L_i`, define the directed
pair rate for replacing child i with parent j as
`lambda_ij = max(L_i-L_j, 0)/N`, with `lambda_ii=0`. In a time interval
`dt=.0085` per training step (the inherited applied prior/G rate), simulate
this continuous-time reaction with exponential waiting times and categorical
pair draws. After a move, copy the parent row and reset the child's G Adam
state; replace the child's payoff by the parent's payoff and recompute all
pair rates for the remaining part of the interval. Do not retain payoffs
between steps. When all payoffs are equal, the rate is exactly zero. Check
algebraically that the expected mass drift of this event generator equals
`pi_i*(mean(L)-L_i)` for uniform pi.

Run three reaction arms on all six original cases:

1. **stream_payoff:** use the two fresh real observations and the current D,
   exactly as the first toy's prequential row-payoff signal;
2. **exact_real_payoff:** replace the two-real average by the toy's exact real
   expectation, while keeping the *learned current D* (diagnostic for
   minibatch noise); and
3. **oracle_critic:** use the toy's true real/model category probabilities to
   form `D*(c)=log((p_c+1e-12)/(q_c+1e-12))`, then use its exact real
   expectation (diagnostic upper bound, not a candidate online method).

The oracle's `1e-12` is solely a finite approximation to a `-infinity`
score for an absent real category. D and G still train from the unchanged
batch-two stream in every arm. The private reaction generator starts at seed
771234 for each arm/case; this is one shared private stream, not a seed
sweep. Record every event, exact TV, critic regret, rare-row retention,
prequential D loss, and elapsed time. Re-run the ordinary control and assert
its deterministic metric fields equal the first toy's saved result.

Interpretation is fixed: a reaction that makes nearly no moves and matches
ordinary training is safe but has no demonstrated benefit. If the oracle
helps and the learned-payoff arm fails, D signal/tracking is the likely
blocker. If both fail, this reaction mechanism lacks evidence. To advance,
the learned-payoff arm must improve non-null TV over ordinary training while
keeping exact-null churn and rare-component loss low. Do not raise the
reaction rate after observing the result. This is a mechanism preflight, not
a native or A2 verdict.
