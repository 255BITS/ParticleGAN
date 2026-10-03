# Creating a new Forge experiment

Use this guide when adding a new research question to the runnable suite. Start
with [EXPERIMENTATION.md](../EXPERIMENTATION.md), the
[compiled memory](../reports/forge/EXPERIMENT_MEMORY.md), and the
[public-API toy contract](../reports/toy_audit/api_contract/README.md). They define
the current gates, budgets, evidence rules and provisional calibration status.

Two worked examples show the process:

- [Sixteen Gaussian clusters](../reports/toy_audit/api_contract/ring16/README.md):
  reuse a vector host, add acquisition gates and scorer controls, and publish a
  full-budget numerical failure with an actual-training GIF.
- [Five-word joint BiGAN](../reports/forge/five-word-joint/README.md): bind an
  existing successful API example into Forge, preserving paired reconstruction
  and keeping a short integration demo separate from full qualification.

## 1. Decide what is new

| What you want to change | Forge declaration | Example |
| --- | --- | --- |
| The question, target, host or measurement protocol | Task in `configs/forge/tasks/` | Acquire sixteen Gaussian clusters within 400 updates |
| The proposed solution or complete training configuration | Idea in `configs/forge/ideas/` | A substantive formulation change evaluated on existing tasks |
| Which tasks support a claim, their tiers and required/diagnostic roles | Existing view in `configs/forge/views/` | Add ordinary acquisition tasks to Tier 1 of `discriminator_stability` |

`python -m experiments.forge new` scaffolds a schema-v2 idea with a draft
[hypothesis-to-decision contract](forge-decision-contract.md). It cannot enter
ordinary execution until its evidence, actual delta, task/runtime bindings,
numerical prediction, falsifier and one-round budget are reviewed and ready.
Task and view registration require editing their declarations; the worked
examples show those files and any necessary adapter changes. Their historical
v1 cards retain their original identity and do not replace the new contract.

Search existing tasks, API cases, readouts and memory before adding a task.
An example can have useful successful evidence while lacking Forge registration;
the five-word investigation found exactly that gap. Keep the original source and
evidence identity when exposing it. Changing only the seed is not a new idea.

Write one falsifiable question. Declare the target distribution or behavior,
conditioning inputs, held-out queries where relevant, and the intended scope.
Separate acquisition, retention, adaptation and paired correctness. A target
with sixteen clusters needs both meaningful coverage and local distribution
quality; a joint encoder/generator needs correctly paired reconstructions as
well as the generated marginal.

## 2. Reuse a host and the public API

Look for a compatible factory and adapter before implementing training code.
The ring uses the shared vector MLPs and `transfer_vector`; the word task injects
Forge components into the existing `WordFixture` joint update. Neither needs a
candidate-specific copy of a trainer.

Scalar hosts use the public `GANTrainer`. Joint, conditional and routed hosts
use supported public recipe, prior, loss, optimizer and policy components with
explicit row/context semantics. A new adapter must check unsupported recipes,
priors, objectives and serving policies before reservation. Silent substitution
of a supported mechanism changes the question.

Bind the actual resolved recipe, architecture, initialization, random streams,
prior and sampling law. Ordinary new tasks use a learned MoG with explicit
positive width. A finite particle-cloud exception must declare its reason in
the task, as the five-word task does. Recipe-owned fields and fixed host-owned
objectives must have an explicit compatible adaptation; disclose those
exceptions rather than tuning them separately for each toy.

## 3. Establish the scorer before training

Freeze numerical bounds, sample count, observation cadence, terminal requirements,
training updates and wall-time allowance before the scientific run. Validate the
scorer using independent target/oracle samples and destructive controls that
address the question. These controls do not require training.

Useful controls include missing modes, uneven mass, centers without Gaussian
spread, continuous-circle impostors, wrong reconstruction pairs, low token
confidence, wrong padding and nonfinite output. Pick the relevant ones. Report
the failed bounds, including an explicit rejection of nonfinite values.

Nearest-center assignments alone do not establish acquisition. The ring counts
meaningful occupancy inside each declared three-sigma ball, divided by the
entire generated sample count. Its final sixteen-mode coverage passed while HQ
and component covariance failed. Likewise, correct argmax words do not establish
confident probabilities or a correct inverse mapping.

Use the existing grader when its measurement contract fits. The current
`transfer_sustained` grader requires its exact 24-check schedule and five passing
terminal observations. A new measurement contract needs a matching evaluator;
adding inert JSON fields does not implement different grading behavior.

## 4. Register the task and its view

Use a nearby task as a structural example, then explicitly materialize the new
target and protocol. Do not inherit execution settings invisibly at runtime.

| File or binding | What it owns |
| --- | --- |
| Task JSON | Description, host and steps, explicit prior, evaluation/sampling contract, source bindings, resources, capabilities and dependencies |
| View JSON | Goal, required/ranking/diagnostic assignments, tier order, eligibility and provisional calibration policy |
| Adapter and sampling registry, when needed | Public execution, supported mechanisms, recorded sampling law and raw evidence for independent grading |
| API caller or scoped reproduction script | Runnable numerical test and actual observed training states |
| Idea and campaign, when needed | Complete candidate configuration and bounded reservation policy for ordinary execution |

Required tiers in an ordinary view must be contiguous from Tier 1. A Tier 2
task therefore needs meaningful required Tier 1 prerequisites. Existing failed,
blocked or missing prerequisites stop remaining tasks in that tier and
higher-tier work. Diagnostics use their declared role and cannot supply ordinary
qualification by bypassing a gate.

For an ordinary new question, add the task as required Tier 1 in the existing
`discriminator_stability` view and increment its revision. Both worked acquisition
examples follow this policy: `ring16_acquisition` and
`five_word_joint_acquisition` are required Tier 1 tasks in revision 3, with a
5/19/2 denominator. Create a new view only for an explicitly different claim or
diagnostic scope, such as clock-free eligibility or architecture transfer.
Keeping an existing denominator unchanged is not sufficient reason for a new
view. Saved campaign requests, calibration/search contracts and recorded revision-2
3/19/2 results retain their original task sets and evidence identities.

Initial placement remains a hypothesis, even for a successful source example.
Scientific adoption needs independent positive/negative references, false
rejection/acceptance and cost measurements under a preregistered compute cap.
Oracle controls or a single fast run do not calibrate a tier. Check the expanded
reservation budget: these five Tier 1 tasks total 2,100 seconds, so the historical
900-second smoke campaign is insufficient. A later tier change gets a new view
revision; changed task budgets, gates or sampling laws need a new protocol
identity with the original evidence preserved.

## 5. Validate, inspect cost, then freeze

Run from the checkout root in the project Python environment. Check the rendering
dependencies as well as the training dependencies before the bounded run.

```sh
python -m experiments.forge validate
python -m experiments.forge plan k3p --view discriminator_stability \
  --through-tier 1 --device cpu
```

Both commands are read-only. The plan shows blockers, compatible reuse,
prerequisites and worst-case reserved cost; it launches no training. This example
plans all five required Tier 1 tasks for an existing candidate and does not
authorize executing them. Inspect every task's compatibility and the full
reservation ceiling before choosing a campaign.

For a new v2 idea, plan its final view, tier and compute cohort after the actual
formulation change. Inspect `decision_contract.actual_bindings` and `expected`,
bind exact prior evidence with byte hashes and identifying fields, and follow the
[draft-to-ready steps](forge-decision-contract.md#prepare-one-reviewable-question).
Planning must show `READY` before submission with that same scope. A changed
task, source, runtime or job map requires a newly reviewed binding; copying an
old ready card supplies no authorization.

Use focused tests for scorer counterexamples, shared API/adapter behavior,
incompatibility preflight, initialization/RNG isolation and grading incomplete
evidence. Preserve the relevant existing view contracts. A changed host need not
fit a test asserting an exact historical task set; keep that assertion scoped to
its original policy and add meaningful checks for the new task.

Commit the frozen scientific setup before running. Keep actual source/recipe/prior/
initialization/budget/sampling/runtime identities in the receipt. A historical
pass with similar settings cannot qualify a different cohort; clean, noisy,
averaged and policy-served results retain separate identities.

## 6. Choose integration proof or ordinary qualification

A short API demo verifies execution, update roles, guards, scoring and rendering.
It can publish instantaneous numerical FAIL and an actual-training GIF while
remaining **INCOMPLETE against the declared full task**. The word task's
32-update demo retains its 20,001-update requirement. Check the full update budget
as well as the observation schedule so a fabricated complete curve cannot turn
a short run into qualification.

Ordinary execution uses a declared campaign and Forge's prerequisite/budget
checks. Submission freezes work; the worker performs execution. The command and
tailing workflow are in [EXPERIMENTATION.md](../EXPERIMENTATION.md#inspect-cost-then-submit).
Use an unused ignored output directory for standalone runs, and keep stdout easy
to tail. Do not rerun an unchanged failure, tune bounds after observing training,
or launch a seed sweep to obtain PASS.

Render actual observed targets and outputs on fixed comparison axes, with update
and numerical status labels. Save the observation arrays before rendering. If
rendering fails after numerical execution, retain that error and raw hashes;
recover media from saved observations with a separately recorded renderer, as
the ring example does. Media recovery needs no additional training or sampling.

## 7. Publish the review artifacts

Commit a concise explanation, final/terminal metrics, declared failed bounds,
cost, compact provenance receipt, reproduction source and final GIF. Keep bulk
stdout, JSONL traces, checkpoints, observation arrays and state dumps ignored or
archived. Keep the original failure and incomplete status visible.

New API evidence can live in a separate compact publication under
`reports/toy_audit/api_contract/`; preserve the frozen campaign indexes. A task's
`research_artifacts.api_publication` names a supplement containing `cases`,
`readouts` and `runs`; evidence rows must belong to that supplement's case IDs.
GIF paths are relative to its publication directory. Optional
`research_artifacts.readout` points to the experiment's explanation in the
checkout.

Task `description` provides the question text. Explicit `retained_question_ids`
and API-case `legacy_ids` connect variants to original questions; host variants
also declare their execution host/problem. The report discovers these links
without claiming that related API demonstrations are Forge qualifications.

Regenerate the same current report after changing declarations or publications:

```sh
python -m experiments.forge experiments-by-tier \
  --output reports/forge/EXPERIMENTS_BY_TIER.md
```

This updates the report and launches no training. Verify its question, numerical
gates, interpretation, source-bound results and GIF links, including relative
paths and anchors. The freshness test catches stale generated content. Maintain
one current solution leaderboard per goal; an artifact-navigation report does
not introduce another competing ranking.

Lead the PR description with the question and links to the readout, task/view,
compact evidence, GIF and generated guide. State the numerical result, whether
full qualification exists, remaining limits and relevant validation. A passing
code review or merged experiment registration does not promote a solution.

After publication, refresh [source-bound recall](forge-research-memory.md) with
`python -m experiments.forge compile --summaries-only`, then check
`python -m experiments.forge compile --check`. Preserve the existing scientific
tables and telemetry when only updating summaries; changing them requires the
original execution envelopes and an explicitly reviewed reducer update.
