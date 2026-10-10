# Recipe-owned priors: selected leader rerun

This fresh campaign measures the seven selected recipe leaders under
`execution.prior_contract=recipe_owned_v1`. It keeps their pre-refactor global
settings, initial prior laws, deterministic seed-0 initialization, data streams,
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

After ordinary work is terminal, a separate per-family `research_diagnostic`
view can measure supported Tier 2 questions that were never attempted. It cannot
repeat an ordinary FAIL, INCOMPLETE or INVALID measurement. Checkpoint consumers
need their own passing producer in the diagnostic namespace: only an ordinary
PASS prefix may be duplicated for that purpose, with its original grade and both
costs retained. A failed or incomplete producer leaves its consumer blocked. The
two possible duplicate prefixes add at most 1,020 seconds per leader, giving a
finite campaign ceiling of 44,040 per leader / 308,280 for seven leaders. There
is no Tier 3 spend and no automatic selection or publication refresh.

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

The eventual compact report will link every certified task result, independent
grading receipt, checkpoint and consumed-stream digest. Actual-training GIFs
illustrate retained scored observations. Ordinary and diagnostic statuses and
duplicate producer costs remain separate. The sole current generated leaderboard
continues to be [the technique inventory](../technique-inventory.md); root owns
any later refresh after independent review.
