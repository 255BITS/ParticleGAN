# Recipe-owned priors: selected leader rerun

The rerun is complete: BCAP passes 6/6 ordinary Tier 1 questions and 9/21 Tier 2
questions. K3P, KA2, R1/R2 and released GAN each pass 5/6 Tier 1 questions, with
ring density quality blocking ordinary Tier 2; their separate diagnostic scores
are 11/21, 10/21, 9/21 and 9/21. Atlas/E22 remain explicitly blocked with zero
spend. [The completed report](publication/README.md) contains exact metrics and
143 training illustrations from 148 physical attempts costing 7,202.084 seconds.

The [R1/R2 baseline readout](../r1r2-baseline/README.md) freezes the selected
R1/R2 reference recipe and explains its Ring16 and clock failures, acquisition
passes and separate deeper diagnostic limits.

This completed campaign remeasured the seven selected recipe leaders under
`execution.prior_contract=recipe_owned_v1`. It keeps their pre-refactor global
settings, initial prior laws, deterministic seed-0 initialization, declared data-stream laws,
update budgets, observation cadence and numerical gates. Moving learn/freeze
policy and prior penalties from tasks or behavior hosts into `Recipe` is an
explicit training-contract change; the old grades are lineage, not a matched
current-source control or evidence of improvement.

The selected roster was resolved in an isolated actual develop checkout at
`03466efa4de8271b9a2c964406dc6f4f0f260792`. [Exact settings and selections](baseline.json)
bind all 30 original revision-8 task cards by their original SHA256 and byte
length. The originals are retained in [legacy-task-cards](legacy-task-cards/).
The fresh studies use inert legacy aliases as binding controls, without running
or regrading those historical controls.

| Visible family | Selected lineage | Retained `prior_reg` | New prior policy |
| --- | --- | --- | --- |
| Atlas | `atlas` | 0 | learned / VICReg |
| BCAP | `bcap-default-baseline-direction-v1` | 0 | learned / VICReg |
| E22 | `e22` | 0 | learned / VICReg |
| K3P | selected `0b37e98…` configuration | 0 | learned / VICReg |
| KA2 | selected `093c6f2…` configuration | 0 | learned / VICReg |
| R1/R2 | selected `302b6ba…` configuration | 0 | learned / VICReg |
| GAN release 0.7 | selected `1e266b5…` MoG configuration | .05 | learned / VICReg |

VICReg selection with weight zero contributes no VICReg penalty. Every fresh
recipe declares `prior_l2=0`, `prior_reg_target_std=1`, and `prior_reg_eps=1e-4`.
There is one whole configuration per visible family; historical optimizer
variants and diagnostic aliases do not expand the roster.

Five supported families use fresh schema-v3 successor declarations. Atlas and
E22 retain their exact original declarations and explicit reference clouds, with
fresh studies binding the new source/task contract. Fully frozen reference
recipes declare the existing host-adaptation fields so host-owned model and
objective defaults remain owned by the frozen task. This adaptation preserves
actual nonprior settings; the new prior ownership contract keeps `prior_reg`
global even where the legacy host adaptation delegated it.

The revision-9 ordinary view retains six required Tier 1 and 21 required Tier 2
questions, plus the separate 300-second clock diagnostic. Full allowance is
43,020 seconds per leader. The ordinary queue completes runnable current-tier
peers and retains the required-tier veto. E22 and Atlas remain unsupported where
their particle-cloud, policy-control or state-selected serving contracts conflict
with the original ordinary hosts. Their exact blockers stay visible without a
paid attempt or a substitute host.

After ordinary completion, four separate per-family `research_diagnostic`
views measured all 21 previously unattempted Tier 2 questions each. They did not
repeat any ordinary FAIL, INCOMPLETE or INVALID measurement. Checkpoint consumers
need their own passing producer in the diagnostic namespace: only an ordinary
PASS prefix may be duplicated for that purpose, with its original grade and both
costs retained. A failed or incomplete producer leaves its consumer blocked. The
two possible duplicate prefixes add at most 1,020 seconds per leader, giving a
finite campaign ceiling of 44,040 per leader / 308,280 for seven leaders. There
was no Tier 3 spend. Publication retained declared lineages without outcome ranking.

The [explicit ownership migration](ownership-migration.json) binds the exact
[ordinary registration](ordinary-registration.json), original selected card,
seven declarations, and [140 actual task-bound comparisons](binding-compatibility.json).
Publication retains the seven declared lineages without ranking outcomes.
All 20 original pins and their historical selections remain in their exact
original-policy archive; thirteen optimizer alternatives receive archived-only
presentation. Only fresh ordinary receipts can support current measurements.
Diagnostic receipts cannot fill ordinary qualification cells.

The primary archive is
`/mnt/ml7tb/ParticleGAN-forge/recipe-owned-priors-20261010`. Admission requires
20 GiB free; an 18 GiB archive cap and 5 GiB remaining-filesystem floor pause new
allocations. Estimated raw storage for five supported families is about 15 GiB.
This estimate is provisional; raw stdout, traces, checkpoints and tensor dumps
remain outside Git and existing archives are retained.

Use the repository virtualenv and single-thread numerical environment. Preparation
and collection add no training. Root must review the READY plans and commit all
scientific inputs and declarations before either explicit run command.

```sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=.
PY=/home/martyn/dev/ParticleGAN/.venv/bin/python
A=/mnt/ml7tb/ParticleGAN-forge/recipe-owned-priors-20261010

# Stage declarations outside Git for review; does not change configs or admit work.
$PY reports/forge/recipe-prior-refactor/workflow.py prepare --artifacts "$A"
# Explicitly install fresh immutable declarations and resolve admission plans.
$PY reports/forge/recipe-prior-refactor/workflow.py prepare --install --artifacts "$A"
$PY reports/forge/recipe-prior-refactor/verify_bindings.py --registration "$A/ordinary-registration.json" --baseline-repository "$A/compatibility/baseline-develop-03466" --output "$A/compatibility/task-binding-verification"

# After source/declaration review and commit, supply its exact Git SHA.
$PY reports/forge/recipe-prior-refactor/workflow.py run --artifacts "$A" --source-commit FROZEN_SHA --submit
$PY reports/forge/recipe-prior-refactor/workflow.py run --artifacts "$A" --source-commit FROZEN_SHA --drain --gpus 0,1 --workers-per-gpu 2

# After ordinary completion, stage and review only unreached diagnostic questions.
$PY reports/forge/recipe-prior-refactor/workflow.py prepare-diagnostic --artifacts "$A"
$PY reports/forge/recipe-prior-refactor/workflow.py prepare-diagnostic --install --artifacts "$A"
# Commit new diagnostic declarations; scientific source digest must be unchanged.
$PY reports/forge/recipe-prior-refactor/workflow.py run --artifacts "$A" --registration "$A/diagnostic-registration.json" --source-commit DIAGNOSTIC_DECLARATION_SHA --submit
$PY reports/forge/recipe-prior-refactor/workflow.py run --artifacts "$A" --registration "$A/diagnostic-registration.json" --source-commit DIAGNOSTIC_DECLARATION_SHA --drain --gpus 0,1 --workers-per-gpu 2

# Project certificates and exact saved scored views; no sampling, scoring or updates.
$PY reports/forge/recipe-prior-refactor/workflow.py collect --artifacts "$A" --output "$A/publication" --media
$PY -m experiments.forge --root . --queue-root "$A/queue" logs --follow --campaign recipe-prior-refactor-selected-leaders-v1
```

The [completed result report](publication/README.md) links every certified task
result, grading identity, checkpoint and consumed-stream digest. Actual-training
GIFs illustrate retained scored observations; the five mode-hold illustrations
show actual stored training metrics. Ordinary and diagnostic statuses and duplicate
producer costs remain separate. The sole current generated leaderboard is
[the technique inventory](../technique-inventory.md).

The completed rerun contains 148 physical attempts for 7,202.084 paid seconds,
with no retries or Tier 3 executions. All 189 required cells are terminal in
their declared scope: 135 measured and 54 blocked for Atlas/E22. The seven
optional clock cells remain separate. [Exact final metrics and costs](publication/results.json)
and [143 actual-training GIFs with export receipts](publication/MEDIA.md) retain
each attempt's source, recipe, initial prior, checkpoint and grading identity.

| Selected leader | Required ordinary Tier 1 | Ordinary Tier 2 | Separate Tier 2 diagnostics |
| --- | --- | --- | --- |
| BCAP direction | 6/6 | 9/21 | already measured; no diagnostic rerun |
| K3P | 5/6 | vetoed by ring failure | 11/21 |
| KA2 | 5/6 | vetoed by ring failure | 10/21 |
| R1/R2 | 5/6 | vetoed by ring failure | 9/21 |
| GAN release 0.7 | 5/6 | vetoed by ring failure | 9/21 |
| Atlas | blocked, zero spend | blocked | none |
| E22 | blocked, zero spend | blocked | none |

Diagnostic counts exclude their eight separately charged successful prefix
producers. They add measurement coverage without filling ordinary qualification
cells. BCAP alone clears the required Tier 1 gates; none clears all Tier 2 gates.
The same BCAP 9/21 outcome as the archived direction cohort is contextual,
without a causal claim about the ownership refactor.

The four ring failures concern density quality: every recipe acquires 16 modes,
but the four failed leaders finish with high-quality fractions .7639–.8442
against the .85 requirement and component covariance errors 2.066–3.483. BCAP's
mode-hold run ends at 8/8 modes with .9893 high-quality fraction but fails the
required sustained suffix. All fifteen 100-Gaussian layouts fail their complete
coverage, sustained accuracy or independent holdout gates; BCAP's terminal
precision is only .2407–.3017. Final endpoints do not replace the retained curves.

The word smoke gate requires a scheduled full joint generation/inverse pass
and independent confirmation of that same state. It does not promise stable
terminal training. Released GAN passes acquisition and later collapses to one
mode at update 20,001 (mass TV .8, exact reconstruction 0, NLL 9.21); its separate
hold run also fails. R1/R2 likewise loses its acquired solution. BCAP, K3P and
KA2 pass the fixed word continuation measurement. A held-state result and a
smoke acquisition result keep their different original criteria.

The [independent runtime audit](publication/runtime-provenance.json) verifies all 148 completions and their saved
checkpoint/prior policies. Actual ordered Gaussian smoke/stability minibatch
hashes match across all five runnable recipes. Other tasks lack an actual batch
trace alignment proof. Word holds resume each recipe's acquired checkpoint and
continuation stream; different acquisition prefix lengths imply different
held-suffix starting RNG states. The shared declared sampling law, named stream,
common prefix and exact prefix-plus-4,000 budget are verified, without a claim of
matched held-suffix batches. Every scientific prior in this campaign is learned;
frozen prior ownership is tested through the API in software only. An unweighted
VICReg diagnostic statistic may be nonzero when its actual loss weight is zero.

The next research priorities are the ring's within-mode density and spill,
sustained mode/word retention, and 100-Gaussian shape and coverage. K3P's
anisotropic and overlap diagnostic passes are useful hypotheses for a later
declared comparison; its ring failure still prevents ordinary Tier 2 admission.
Atlas/E22 need separately supported policy-aware hosts rather than substituted
results. No further experiment or retuning is launched by this report.

Publication reconstructs exactly the seven ordinary leader rows at the executed
source. Its current presentation additionally retains one exact, zero-attempt
archived BCAP base declaration for taxonomy navigation, with no qualification
credit. The selected BCAP scientific row remains the newly measured DualNorm
recipe. All thirteen excluded optimizer choices and the original twenty-pin
card remain original-policy context.

The explicit source reconstruction and one-time policy advance use the original
ordinary freeze and committed registration:

```sh
$PY reports/forge/regenerate_technique_inventory.py --root . --device cuda \
  --source-commit a0493ca3c213d559523eadc76e244ec9f9c1f81f \
  --advance-policy --ownership-migration reports/forge/recipe-prior-refactor/ownership-migration.json \
  --declaration-registration reports/forge/recipe-prior-refactor/ordinary-registration.json
# Subsequent refresh reads the registered compact summaries; no training or old raw archive needed.
$PY reports/forge/regenerate_technique_inventory.py --root .
```

The policy-advance command requires the original pin state and is recorded once;
it is not a repeated selection command. [Software validation](software-validation.md)
retains the first full-suite failures, exact focused repairs, and separately capped
zero-training publication and saved-media completion stages.
