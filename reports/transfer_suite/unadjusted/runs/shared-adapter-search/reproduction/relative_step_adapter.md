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

```bash
/tmp/pr38-default-env/bin/python -m pytest -q tests/test_relative_step_adapter.py
/tmp/pr38-default-env/bin/python -u -m benchmarks.transfer_suite.shared_adapter_search \
  --output /tmp/shared-adapter-replay > /tmp/shared-adapter-replay.log 2>&1
tail -f /tmp/shared-adapter-replay.log
```

Use a fresh output directory. The output records the full shared recipe plus an
explicit `mechanism` card, all observations/actions, tensor-level gradient and
proposal norms, actual attenuation, source archive and runtime fingerprints.
Importing these rows into the primary leaderboard must retain the mechanism
identity; matching recipe fields alone is insufficient.

The initial control attempt stopped after exact numerical parity because a
reporting assertion compared in-memory tuples with JSON lists. Canonical JSON
comparison fixed that assertion; the original result, log and exact source
archive are retained separately. This was a reporting failure, not a numerical
training failure.

This experiment changes no production defaults. Scores and limitations belong
in the generated artifact report; a partial screen cannot claim an overall pass.
