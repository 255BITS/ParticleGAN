# Pure BCAP initial loss comparison

This family starts from develop `5f2ac0116de93ab697a7d2faa840245430a3af90`.
Develop's candidate/study separation at `4f3a70e7` is integrated on the PR
branch. Executed v1 declarations retain their original identities; the
upstream schema-3 example uses the successor ID `bcap-pure-adam-example-v3`
to avoid replacing the executed v1 parent.
It uses native PyTorch Adam, constant learning rates and moments, and one fixed
critic-input gradient penalty on both real and generated samples:

```text
D loss = adversarial_D(real_scores, fake_scores)
       + coefficient / 2 * (mean(max(norm(grad_x D(real)) - cap, 0)^2)
                          + mean(max(norm(grad_x D(fake)) - cap, 0)^2))
G loss = adversarial_G(fake_scores, real_scores)
Adam.step(D loss); Adam.step(G loss)
```

On a joint BiGAN host, the second update uses `joint_g_loss(fake, real)` and
updates both G and E: generated pairs should look real, encoded real pairs
should look fake. Standalone GANs use `g_loss`. The paired relativistic
expression is identical in both cases.

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

The original round evaluated each loss at global LR `.0010625` and `.00425`.
D uses the same LR; learned latent locations use twice that LR. Adam betas
remain `(0, .999)`. These are ten complete global recipes, not per-task tuning.
Five [search declarations](../../../configs/forge/searches/pure-bcap-relativistic-rates-v1.json)
share one campaign. Loss changes have structural cards; each numeric search
varies only LR. The [frozen plan](plans.json) records the complete roster.

The [independent initial audit](audit-initial.json) verifies 53 paid attempts
from source `af75a3fea19aa6e4d1ca2be867b9c50a02931e33`, costing 1,038.752
seconds. Relativistic BCAP at `.00425` passes two-pole, unused-token hold and
AE hold: **3/6 required**. At `.0010625`, it passes AE hold and word acquisition:
**2/6 required**. Both pass the separate clock diagnostic. Neither qualifies
Tier 1. These are two complete recipes; their successful cells cannot be mixed.
The [saved final metrics and media receipts](original-initial-readout.json)
include 51 actual-training GIFs; the two cancelled word attempts retain their
original partial evidence. The [verified archive receipt](archive-initial.json)
binds all 53 paid attempts and 2,012 byte-exact files in the local artifact bundle.

The nonrelativistic objectives exposed an integration bug: the word host
called the standalone generator objective, which ignored real scores and
therefore never updated E. Both completed non-saturating word attempts correctly
failed the encoder-update guard. The eight nonrelativistic candidates were
cancelled; their completed and partial receipts and costs remain original-source
context. Unexecuted cells remain unmeasured. `GANLoss.joint_g_loss` now supplies
the missing reversed-label encoder term. Software controls verify G, E and
learned-prior updates, a frozen critic during that update, and exact unchanged
relativistic behavior.

Gaussian failures concern terminal CDF shape and stability. The lower-rate
relativistic run briefly achieves KS `.04932` and `.03851`, then finishes at
`.06821`, above `.05`. The higher rate finishes at KS `.09674` and width ratio
`1.26168`, above `1.2`. Its ring reaches all 16 modes but misses high-quality
fraction (`.82275 < .85`) and component covariance error (`5.3665 > .85`). More
updates are an untested hypothesis; transient passing points do not justify
weakening the sustained criteria.

The corrected family uses [small reusable schema-3 cards](../../../configs/forge/ideas/bcap-pure-adam-v2.json).
Four schema-2 searches own the finite rate grids, hypotheses and budgets.
Candidates carry no decision contract, study hypothesis or task prior.
The [correction preflight](repair-preflight.json) reviews eight whole recipes,
with up to 20,160 new paid seconds, retaining the original cost within the
25,200-second aggregate ceiling. Each corrected candidate needs all seven of
its own source-bound Tier 1 cells. The two unchanged relativistic recipes will
not be rerun for this merge.

The current session cannot access CUDA or GitHub. No corrected GPU run has
started, no corrected execution cohort is frozen, and no new family winner has
been registered. The current leaderboard retains its registered scientific
rows and selections. Navigation marks the updated word source contract as
changed; old word grades remain attached to their original sources. Complete
the corrected loss comparison before choosing another experiment angle. Then
consider a separately budgeted Gaussian rate/duration comparison and inspect
the saved critic/shape diagnostics, retaining the existing numerical gates.

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

Prepare and freeze the correction from a clean reviewed commit on the actual
two-A6000 host, using the same project Python 3.12.13 environment as the original
round. Preparation never trains and refuses to overwrite frozen declarations:

```sh
python reports/forge/pure-bcap/prepare_repair.py --freeze
python reports/forge/pure-bcap/run.py --plan repair-plans.json --expected-commit REVIEWED_COMMIT \
  > runs/forge/pure-bcap-joint-loss-repair-v2/coordinator.log 2>&1
tail -F runs/forge/pure-bcap-joint-loss-repair-v2/coordinator.log
tail -F runs/forge/pure-bcap-joint-loss-repair-v2/queue/events.jsonl
```

Without `--freeze`, preparation only materializes reusable declarations.
The runner refuses unavailable GPUs, dirty tracked files, changed source
bindings or an unexpected commit before admitting work. Original v1
reproduction sources are retained at `af75a3fe`; `prepare.py` intentionally
refuses to rewrite the already-frozen original plan.

Raw logs, saved samples, source snapshots and checkpoints stay local or in the
artifact archive. Publication exports certified final metrics and GIFs from
the actual saved training observations, without new updates or sampling.
The [single current leaderboard](../technique-inventory.md) is updated in place.
