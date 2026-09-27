# Shared relative-step adaptation experiment

This research rule caps each ordinary Adam proposal by the parameter tensor's
pre-update RMS. It is one shared optimizer mechanism, used on every G, D and
particle tensor, with no task identity or evaluation feedback:

```text
delta = ordinary_Adam_proposal(parameter, raw_gradient, moment_state)
factor = min(1, fraction * max(RMS(parameter_before), 0.1)
                / (RMS(delta) + 1e-12))
parameter_after = parameter_before + factor * delta
```

The ordinary Adam moments are unchanged. The factor can only attenuate a
proposal, so zero/small gradients and the existing cosine schedule still permit
vanishing steps. Tensor shape influences its RMS, but role labels only annotate
the report. There are no per-role or per-example factors. This is an altered
Adam update rule, not plain Adam with different declared learning rates.

The fixed base recipe is `lr00425_prior2`: G/D LR .00425, particle LR .0085,
Adam(0,.99), Rp logistic, b_cap3/κ1.25, spread .05 and no particle L2. All existing
architectures, particles, data, batches, budgets and metrics are retained. The
three predeclared fractions are .01, .025 and .05; all other rule constants are
fixed. Role-specific attenuation is measured only for interpretation.

The bounded study first checks an identity wrapper against three complete
archived trajectories. It screens six predeclared tasks for each candidate,
then completes the other13tasks for the candidate with most sustained passes,
lowest mean metric shortfall and finally lexical candidate name. No new fitting
occurs after screening. Seed0 only; EMA remains separate.

The runner (`shared_adapter_search`) and the Adam-patching implementation have
been removed: the rule changed Adam's update from outside the optimizer, and
optimizers now come only from the recipe. The exact sources and their contract
test are archived in
`reports/transfer_suite/unadjusted/runs/shared-adapter-search/reproduction/`.
`relative_step_adapter.mechanism` keeps the declared card so archived rows still
validate. A new attempt belongs in the recipe-built optimizers as an option.

The output recorded the full shared recipe plus an
explicit `mechanism` card, all observations/actions, tensor-level gradient and
proposal norms, actual attenuation, source archive and runtime fingerprints.
Importing these rows into the primary leaderboard must retain the mechanism
identity; matching recipe fields alone is insufficient.

Two initial control attempts stopped on reporting comparisons: in-memory tuples
versus JSON lists, then an image result's creation timestamp. Their numerical
trajectories match exactly. Canonical JSON comparison and removal of runtime
timestamps corrected the assertions; original results, logs and exact source
archives are retained separately. These were reporting failures, not numerical
training failures.

The completed experiment is negative: fractions .01 and .025 each pass only 1/6
screen tasks; .05 passes 2/6 and was selected. Its complete score is **12/19**
(7/9 required, 2/6 data, 3/4 images), below the base recipe's 15/19. It fixes none
of the four original failures and additionally loses trajectory, anisotropic and
stripes. Stripes meets final metrics but has only four final passing checks.

The selected cap attenuates 2.23% of D tensor-updates, 5.56% of G updates and
3.87% of prior updates. Mean factors are .9955, .9889 and .9896, respectively;
rare large changes can still alter the trajectory substantially. Measured
adaptation plus tracing overhead is 8.19 seconds across 108.18 seconds of summed
full-suite runtime. This includes instrumentation and concurrent CPU effects;
it is not a production speed estimate. Four adapter contract tests and ten
existing integration tests pass.

This experiment changes no production defaults. The complete artifact report
retains all 31 candidate episodes and seven identity-control episodes across
attempts. A partial screen cannot claim an overall pass.
