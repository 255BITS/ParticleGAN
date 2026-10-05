# Pure BCAP initial loss comparison

This family starts from develop `5f2ac0116de93ab697a7d2faa840245430a3af90`.
It uses native PyTorch Adam, constant learning rates and moments, and one fixed
critic-input gradient penalty on both real and generated samples:

```text
D loss = adversarial_D(real_scores, fake_scores)
       + coefficient / 2 * (mean(max(norm(grad_x D(real)) - cap, 0)^2)
                          + mean(max(norm(grad_x D(fake)) - cap, 0)^2))
G loss = adversarial_G(fake_scores, real_scores)
Adam.step(D loss); Adam.step(G loss)
```

The coefficient and L2 cap are both 1; the penalty runs every critic update.
There is no gradient clipping, A2 damping, adaptive particle gain, EMA anchor,
prior regularization, averaged serving, additive input/output noise, or cosine
annealing. Learned MoG kernel noise remains part of each task's distribution.

The public formulation is small:

```python
from particlegan import GANTrainer, get_recipe

recipe = get_recipe("bcap", loss="hinge", lr=.0010625)
trainer = GANTrainer(recipe, generator, discriminator)
```

The five supported losses are `relativistic` (paired logistic),
`non_saturating` (logistic), `hinge`, `wasserstein`, and `least_squares`.
They consume raw critic scores. Loss choice is independent of the BCAP penalty;
Wasserstein plus BCAP does not use weight clipping or an interpolation penalty.
The [API guide](../../../docs/api.md#ganloss) defines their exact reductions.

The initial round evaluates each loss at global LR `.0010625` and `.00425`.
D uses the same LR; learned latent locations use twice that LR. Adam betas
remain `(0, .999)`. These are ten complete global recipes, not per-task tuning.
Five [search declarations](../../../configs/forge/searches/pure-bcap-relativistic-rates-v1.json)
share one campaign. Loss changes have structural cards; each numeric search
varies only LR. The [frozen plan](plans.json) records the complete roster.

All six required Tier 1 tests retain their original numerical gates,
initialization, resources, prior, sampling law, and training allowance. The
three CPU behavioral hosts retain their task-owned particle L2, hold or
reconstruction objectives. The candidate supplies their adversarial objective,
optimizer and BCAP penalty. Gaussian, ring, words and the clock measurement
diagnostic run across GPU 0 and GPU 1, one worker per GPU; one CPU worker runs
the behavioral tests. All runnable Tier 1 peers finish after a scientific
failure. Higher tiers are outside this round.

The ceiling is 2,520 seconds per candidate (2,220 required plus 300 diagnostic),
25,200 seconds overall. The shared campaign does not replenish budgets between
loss searches. The overall display choice uses required PASS count, descending,
then configuration hash, ascending. Only one whole candidate supplies a family
row. A good final endpoint cannot replace the sustained terminal gate.
Calibration remains provisional, and this round cannot change public defaults.

Historical K3P-derived BCAP stays a separate family and retains its evidence.
Removing several mechanisms at once defines a baseline, so comparison with that
history cannot isolate the causal effect of one removal. Generated Forge
admission metadata is separate from the simple public training recipe; its
numerical prediction is not the scientific pass/fail criterion.

Reproduction preparation is read-only with respect to training, and refuses
to overwrite a frozen round:

```sh
python reports/forge/pure-bcap/prepare.py
python reports/forge/pure-bcap/run.py --expected-commit EXECUTED_COMMIT
tail -F runs/forge/pure-bcap-losses-v1/coordinator.log
tail -F runs/forge/pure-bcap-losses-v1/queue/events.jsonl
```

Raw logs, saved samples, source snapshots and checkpoints stay local or in the
artifact archive. Publication exports certified final metrics and GIFs from
the actual saved training observations, without new updates or sampling.
The [single current leaderboard](../technique-inventory.md) is updated in place.
