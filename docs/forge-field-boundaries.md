# Forge field ownership

An experiment task defines the problem and comparison conditions: data, model
architecture, prior implementation and width, initialization, resource budget,
evaluation and sampling law. A technique defines the training mechanisms. Its
configuration supplies numerical settings within those mechanisms. The protocol
binds seed, named RNG streams and comparison policy. View policy selects the
required tasks and tiers; it does not change their scientific contracts.

[`boundaries.py`](../experiments/forge/boundaries.py) assigns every public
`Recipe` field an explicit owner. Adding a public field without updating this
registry fails validation. The configuration-search whitelist comes from the
same module; being a numerical hyperparameter does not automatically authorize
searching it. A configuration must preserve its technique's mechanisms. Setting
a penalty coefficient to zero, adding a schedule or disabling an update control
can change the technique even though the field accepts a number.

The formulation family is the reusable solution axis; the technique signature
is the stricter boundary for numerical configuration search. An ordinary idea
may change existing controls within its formulation family with explicit
structural provenance, but a changed optimizer/loss formulation needs a new
family. Each candidate supplies one global recipe across eligible tasks.
Task applicability can make a role or control inactive; it cannot silently
choose a bespoke optimizer recipe. Publication pins one complete ordinary
candidate/cohort row per family and retains other cohorts unranked. A family
label never authorizes pooling source, runtime, prior or sampling identities.

| Owner | Binding |
| --- | --- |
| Task | Prior, architecture/data, initialization, update limit and scheduled horizon, particles, latent dimension, batch size, evaluation and sampling |
| Technique | Optimizer/penalty family, loss and encoder modes where supported, update and serving policy, structural switches |
| Hyperparameter | Learning rates, moments, coefficients and schedule settings within a fixed technique |
| Protocol | Seed, named RNG derivation and streams, fixed comparison and robustness policy |

The task owns external model architecture independently of `Recipe`. Recipe
fields such as `model`, `conditioning` and `encoder_mode` describe technique
controls on the scalar trainer path. Existing behavioral tasks instead own their
original component topology and objectives. In particular, scalar trainer
`prior_reg` is a numerical configuration setting, while behavioral hosts retain
their own objective and particle regularization. AE routing settings remain
configuration fields where its public encoder consumes them. Other behavioral
hosts own their original routing/objective definitions.

`total_steps` records the task's original schedule horizon for scheduled
techniques. A schedule-free technique retains `total_steps: null`; the separate
task execution limit still bounds its run. A continuation preserves the original
schedule horizon and declares its additional budget separately.

The task's prior always selects the actual public code path: `mog` selects
`MoGParticlePrior` and `particle_cloud` selects `ParticlePrior`. Forge binds an
absolute task sigma directly and sets the public recipe's relative calibration
field `sigma_rel` to zero. The receipt therefore records both the task's actual
width and its distinct implementation. Candidate and protocol prior declarations
are labelled references and cannot supply an effective task prior.

Every resolved task can emit an ownership receipt with each recipe field's
effective value, owner and source, plus its prior, initialization, architecture,
budget, evaluator and protocol. The receipt consumes the actual resolved public
recipe rather than constructing another recipe. It rejects contradictions in
the task prior, declared resources or schedule horizon.

Inspect a declaration's effective task bindings before execution:

```sh
python -m experiments.forge plan k3p --through-tier 1 --show-boundaries
```

This adds ownership receipts to the ordinary read-only plan; it launches no
training and makes no qualification claim.

Behavioral sources contain some effective values outside `Recipe`. Those fields
are labelled `host_owned` with a null effective recipe value; the unused recipe
reference is retained separately. AE encoder resources bound by its public
recipe remain explicit effective fields. `Recipe.name` is a metadata label.
These statuses prevent an unused recipe default from being reported as an actual
trained component or objective.

Registered extensions retain their explicit binding source. Public recipe
normalizations are also labelled; for example, a fixed `reg_arm` forces the
resolved critic formulation to K3P. Old frozen native requests with undeclared
resources retain their candidate/API values as `legacy_reference`; an old
initializer fallback is similarly labelled. Current declarations make those
task bindings explicit without rewriting the old evidence.

Task-recipe adaptation retains the original candidate's delegated reference
values separately from effective task values. Historical `host_definition`
optimizer and penalty constants likewise remain labelled inactive provenance;
they do not override the candidate's current public optimizer settings. Changing
the retained reference does not make it an active tuning axis.

Technique grouping and scientific comparison cohorts serve different purposes.
The same training mechanisms may operate on separate MoG/cloud, initialization,
architecture, runtime or serving cohorts. A shared technique never pools those
qualification results. Existing evidence keeps its frozen identity and verdict;
adding ownership declarations does not retroactively qualify or regrade it.

The boundary tests check resolution and provenance without training. Scientific
toy qualification still requires the public-API numerical metric and actual
training GIF contract in `EXPERIMENTATION.md`.
