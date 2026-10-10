# Prospective follow-up: continuous mass flow, no cloning

Written after [FLOW_SPEC.md](FLOW_SPEC.md) and its results. Astra advised
testing whether discrete one-row jumps caused the flow reaction's null churn
and rare-component regression. This is the last predeclared arm in this toy
line. Keep all six cases, one seed, N=32, batch size two, 1,600 updates,
real/fake random quantiles, D/G model, optimizer rates, and exact evaluator
unchanged. Do not tune the transport speed from the preceding outcomes.

Maintain a log probability for each of the 32 existing rows. Start uniform.
Before D trains on each fresh batch, compute each row's paired generator
payoff `L_i` as in the first toy, using the current D and the two fresh real
observations. Update the log probabilities by

`log_pi_i <- log_pi_i - .0085 * L_i`, then normalize with softmax.

This is the exact frozen-payoff solution for one step of the same replicator
flow used in the magnitude-sensitive stochastic reaction. `.0085` is the
inherited applied generator/prior rate; there is no extra multiplier,
clipping, or probability floor. No row is cloned; its generator parameters
and optimizer history continue normally. D/G fake rows are sampled from the
current probabilities. A function-preserving gauge change must preserve the
instantaneous payoffs exactly.

Compare two arms:

1. **learned_payoff:** fresh-batch payoff from the current learned D;
2. **oracle_payoff:** the toy's exact real/model probabilities produce
   `D*(c)=log((p_c+1e-12)/(q_c+1e-12))` and its exact real expectation,
   as in FLOW_SPEC.md. This is evaluator-only knowledge, not a candidate.

Re-run the ordinary baseline and assert its deterministic metrics match the
first toy. Verify algebraically that the small-step derivative of the
softmax update is `pi_i*(mean_pi(L)-L_i)`. Report exact TV trajectories,
critic regret, prequential D loss, row effective sample size, minimum mass,
rare mass, and compute time. Compare with the prior direct-mass Adam arm,
flow reaction, and ordinary baseline. If the oracle mass flow fails, reject
this objective-flow route in the toy; if only learned payoff fails, D
tracking is the blocker. If both improve and preserve rare mass and null
stability, that supports a further preflight, not native promotion.
