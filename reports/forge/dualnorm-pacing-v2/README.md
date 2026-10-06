# Finite BCAP dualnorm pacing study

This study tests independent discriminator and sampled-prior pace around the
user-selected full dualnorm starter. It is declared before training. The
[contract](contract.json), [Stage A search](../../../configs/forge/searches/bcap-dualnorm-pacing-v2-a.json),
and [campaign](../../../configs/forge/campaigns/bcap-dualnorm-pacing-v2.json)
set the finite space, conditional rules and budgets. The existing
[technique inventory](../technique-inventory.md) remains the sole generated
leaderboard for this goal. A whole recipe can remain a useful measured
alternative while failing Tier 1; no public default changes follow automatically.

Read [EXPERIMENTATION](../../../EXPERIMENTATION.md),
[compiled memory](../EXPERIMENT_MEMORY.md), and the original
[BCAP optimizer readout](../dualnorm-tier1/README.md) first. The audit on branch
[PR #307](https://github.com/255BITS/ParticleGAN/pull/307) explains the prior-rate omission and
the endpoint tradeoff. Its proposed >=4-pass continuation was unexecuted.
This study explicitly replaces that predicate with >=3 and >=the matched
current-source control before spending: an intermediate rate can improve ring
shape even when pacing alone does not add a complete gate.

| Stage | Global recipes | Conditional admission |
| --- | --- | --- |
| A | 20: full dualnorm mu0, etaG=.01, D/G={.5,.75,1,1.5,2} crossed with absolute prior={.003,.01,.03,.1} | Fixed first stage |
| B | 3: etaG={.012,.016,.022}, mu0; retain A winner's D/G and absolute prior | A fully executed; whole PASS-count/hash winner has >=3 required passes and >=current control |
| C | 2: mu={.5,.9}; retain combined A/B winner's etaG, D/G and absolute prior | All admitted A/B recipes fully executed; whole winner has >=3 required passes and >=current control |

Selection uses Forge's unchanged required-PASS-count objective followed by
ascending configuration hash. Every recipe uses all six required tasks plus
the separate clock diagnostic. All seven actual attempts must finish before
a conditional decision; the selected winner and control must have valid
PASS/FAIL evidence on all seven. Numerical errors remain visible and supply
no passing credit. Missing cells prevent continuation. Task-local maxima and
the five-observation terminal suffix remain unchanged; there is no extra
training to turn a near miss into a pass and no task-specific recipe mixing.

Exactly one Stage A center, etaG=.01/D1.5/prior=.03, is a justified matched
current-source control. Optimizer, public API and observer source changed after
the historical executed commit `15eb7cb0`. Old receipts remain under their original
source and cannot fill this current-source control. The branch does not repeat
Adam recipes, change seeds, or rerun unchanged evidence solely for a merge.
Exact compatible jobs are reused by Forge without new attempts.

All non-optimizer conditions stay fixed: pure BCAP paired logistic loss,
gradient cap coefficient/cap1, no additional interventions or noise, constant
existing schedule multipliers, and each task's architecture, data, prior,
auxiliary objectives, batch sequence, deterministic initializer and named RNG
streams. The explicit direct-coordinate/fixed-initialization task exceptions
remain. Prior rows have no momentum. E shares G's step size. Epsilon fields
are not swept. Two-pole and unused-token have no sampled latent prior, so a
prior-pace setting cannot directly affect their update rule.

The bound is **25 admitted recipes ×2,520=63,000 paid/reserved seconds**.
Each recipe reserves 1,620 CUDA seconds and 900 CPU seconds: Gaussian 120,
two-pole 300, unused-token 300, AE 300, ring 300, words 900, clock 300. There are
at most two exclusive GPU workers plus Forge's automatic CPU worker, with
no speed ranking. The independent monotonic elapsed ceiling is **43,200
seconds (12 hours)**, starting before first admission. Admission requires
the complete original job timeout plus30 seconds of launch headroom to end
before the final30-second cleanup window: timeout+60 seconds must fit the
global remaining time.
Jobs that cannot fit remain unexecuted/unknown. A watchdog begins cancellation
at the cleanup margin; the driver collects supervisors before publishing a
terminal notification receipt. No timeout is shortened to fit the remainder,
and zero automatic retries, edge expansions or Tier 2 jobs are allowed.

The committed numeric master and driver derive every possible branch before
spending: one A search,20 possible B searches and80 possible C searches,
covering240 distinct reachable recipes. Only one B and one C branch can be
admitted, so these240 possibilities do not authorize240 runs. `prepare`
materializes immutable recipe cards; `plan` validates all possible branches
without admission. Conditional specs are passed unchanged to Forge and copied
into normal source-bound search registrations. Configuration cards and report
helpers are outside the scientific source snapshot; no scientific source
changes are needed between stages. `freeze` requires committed source,
driver and master bytes. The same source commit/runtime/protocol/task keys
must remain through execution.

From this worktree, using the existing project environment:

```sh
PYTHON=/home/martyn/dev/ParticleGAN/.venv/bin/python
STUDY_QUEUE=runs/forge/bcap-dualnorm-pacing-v2-queue
mkdir -p "$STUDY_QUEUE"
$PYTHON -u reports/forge/dualnorm-pacing-v2/run.py prepare > "$STUDY_QUEUE/prepare.log" 2>&1
$PYTHON -u reports/forge/dualnorm-pacing-v2/run.py plan > "$STUDY_QUEUE/plan.log" 2>&1
# Review and commit the declarations/driver/cards before freezing or training.
$PYTHON -u reports/forge/dualnorm-pacing-v2/run.py freeze > "$STUDY_QUEUE/freeze.log" 2>&1
$PYTHON -u reports/forge/dualnorm-pacing-v2/run.py run --gpus 0,1 > "$STUDY_QUEUE/driver.log" 2>&1
```

Tail `driver.log` for stage decisions and60-second progress updates containing
completed/pending attempts, running tasks/devices, known gate counts and costs.
`events.jsonl` is the queue's durable secondary stream. Each actual attempt
has its own `run.log`, immutable request and terminal receipt. A process-only
interruption can reattach with the explicit `resume` command, preserving the
same monotonic boot epoch and deadline. It does not retry completed or failed
jobs, reset the clock, alter a branch or admit additional settings.

```sh
tail -F runs/forge/bcap-dualnorm-pacing-v2-queue/driver.log
/home/martyn/dev/ParticleGAN/.venv/bin/python -u reports/forge/dualnorm-pacing-v2/run.py resume --gpus 0,1 >> runs/forge/bcap-dualnorm-pacing-v2-queue/driver.log 2>&1
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/dualnorm-pacing-v2/run.py report
```

Final compact `results.json` records the complete task denominator, actual
source, recipes, branch decisions, costs and whole selection; raw observations
remain ignored in the queue. The local `terminal-summary.json` includes final
status/exit code, zero running workers, finished supervisor/lease checks,
control/winner counts and source identity for the completion notifier. Reports
must distinguish historical and current source cohorts and retain any incomplete
or failed result. Actual-training GIFs and artifact/provenance publication use
the existing Forge exporters after execution. No native7k, R1/R2, statistical
seed claims or width/depth transfer are part of this finite study.
