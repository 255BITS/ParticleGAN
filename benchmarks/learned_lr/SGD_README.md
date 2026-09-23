# Plain SGD and raw-gradient LR feedback

This study keeps the inner update exactly

`parameter_next = parameter - learning_rate * gradient`.

It uses `torch.optim.SGD` with zero momentum, dampening, and weight decay,
Nesterov disabled, and no gradient clipping or coordinate normalization. The
generator and particle prior share one LR; the discriminator has its own LR.
Analytical tests verify the update across unequal coordinates and successive
gradients, including that the optimizer retains no momentum state.

The initial sweep tests 16 independent G/D rate pairs:

- G: `0.01, 0.05, 0.25, 1.25`
- D: `0.001, 0.005, 0.025, 0.125`

Each pair gets constant and delayed-cosine schedules on the four-mode ring and
3×3 Gaussian grid training distributions. Every run uses the same seed 0,
1,200 updates, networks, particles, losses, and regularizers. The Adam/cosine
control uses its existing settings. Failed/nonfinite runs remain in the report
with their exceptions and partial traces.

Unlike the first LR-controller experiment, selection explicitly values
sustained coverage and HQ. The per-task objective is:

```
20 * no_sustained_pass
+ 3 * mean_last_five_deficit
+ 2 * final_deficit
+ 0.2 * normalized_SW1_rollout
+ 0.1 * stable_progress
```

Here `deficit = missing_mode_fraction + max(0, 0.9 - HQ) / 0.9`. Sustained PASS
requires every mode, at least 90% HQ, and five consecutive passing observations
through the final budget. Both training tasks receive equal weight. Failed or
incomplete runs receive objective 1000. Coverage/HQ metrics are outer-loop
training targets; they never enter the controller's observations.

If rate tuning establishes a useful regime, the optional controller fit uses
at most 32 policies, each tested on both training distributions. Its inputs are
bias, normalized progress, log gradient RMS relative to the initial observation,
alignment with the previous observed gradient, and log parameter RMS relative
to initialization. It predicts one positive scalar LR multiplier per optimizer
role, bounded to `[0.05, 4]`, every 20 updates with log-space smoothing. Only the
LR changes; the actual parameter update remains raw-gradient SGD.

```bash
python -u -m benchmarks.learned_lr.sgd_study sweep --output /tmp/sgd-study \
  > /tmp/sgd-sweep.log 2>&1
tail -f /tmp/sgd-sweep.log

# Optional second stage, selected only after reviewing the rate sweep:
python -u -m benchmarks.learned_lr.sgd_study fit --output /tmp/sgd-study \
  > /tmp/sgd-fit.log 2>&1
```

Existing eight-mode and full behavioral-suite results are validation data now;
they are not fresh held-out evidence. Neither these results nor any new task or
architecture is used to tune this study's rate sweep or policy fit. New transfer
tests are declared separately before final evaluation.

The subsequent per-tensor study assigns each parameter tensor its own SGD
group. It uses `alpha = target * max(parameter_RMS, .01) / max(gradient_RMS,
1e-12)`, bounded to `[1e-4, 100]` times its selected role's base LR. The physical
update remains `-alpha * raw_gradient`. This is per-tensor adaptive gradient
descent, not ordinary common-LR SGD. Its follow-up tests cosine decay of target
fractions and a causal running RMS denominator with beta .9 or .99. The running
quantity is one scalar per tensor, initialized from the first gradient; there
are no coordinate-wise Adam moments or momentum in the update direction.

```bash
python -u -m benchmarks.learned_lr.relative_sgd_study \
  --sgd-output /tmp/sgd-study --output /tmp/relative-sgd \
  > /tmp/relative-sgd.log 2>&1
python -u -m benchmarks.learned_lr.relative_sgd_followup \
  --preceding /tmp/relative-sgd --output /tmp/relative-followup \
  > /tmp/relative-followup.log 2>&1
```
