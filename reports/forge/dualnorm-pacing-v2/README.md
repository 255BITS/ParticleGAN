# BCAP-pure dualnorm pacing: completed Tier 1 search

**Use full dualnorm with G/E step .012, D/G=1.5, sampled-prior step .03,
and network momentum 0 as the new experimental starting recipe.** It passes
four of the six unchanged required Tier 1 tasks, compared with three for the
matched .01 starter. The additional pass is two-pole; unused-token hold, AE
hold and joint words retain their passes. Gaussian and ring acquisition still
fail. This is a complete measured improvement within the provisional screen,
not a qualified public default.

```python
from particlegan import get_recipe

recipe = get_recipe("bcap", optimizer_family="dualnorm", lr=.012,
                    d_lr_mult=1.5, prior_lr_mult=2.5, optimizer_momentum=0.)
```

The actual steps are etaG=etaE=.012, etaD=.018 and etaPrior=.03 before
the unchanged schedule multiplier. Prior momentum remains zero. Preserve each
task's other recipe fields. These are normalized step units; they are not Adam
learning rates. The public `get_recipe("bcap")` continues to return Adam.
The [current technique leaderboard](../technique-inventory.md) is the single
ranking for this goal; its explicit experimental measurement pin is updated
to this complete recipe. The original [Tier 1 screen](../dualnorm-tier1/README.md),
starter receipt and historical search selections retain their original source.

## Results and exact remaining failures

All **25 admitted configurations and 175 actual attempts completed**, without
numerical errors, retries or incomplete cells. The six required tasks account
for 52 PASS and 98 FAIL cells; the separately declared clock diagnostic passes
for all 25 recipes. It is not a seventh requirement. Execution used both GPU
workers and the CPU worker, completing in 10,796.60 elapsed seconds, approximately
three hours. Recorded worker cost is 19,655.90 seconds against the 63,000-second
ceiling; the independent 12-hour allowance was a maximum, not a reason to keep
training after the finite search finished. All supervisors, children and leases
finished, with zero remaining reservation. The completion callback was delivered.

The table audits the selected recipe's gates; it is not another leaderboard.
Every required task has 24 observations and requires at least **five consecutive
passing observations at the end**, with all metric conditions true together.
An endpoint pass alone does not suffice. Every selected endpoint has the required
sample count and finite samples where those conditions apply.

| Required task | Updates | Result / passing terminal suffix | Selected endpoint and threshold |
| --- | ---: | --- | --- |
| Gaussian acquisition | 1,000 | FAIL / 0 | Mean error .12788 sigma <=.2 and std ratio 1.05499 in [.8,1.2] pass; **CDF KS .11428 >.05** fails |
| Two-pole | 80 | PASS / 17 | Mean absolute active coordinate .94355 >=.3; median absolute input gradient .28846 <=1 |
| Unused-token hold | 200 | PASS / 18 | Concept movement .99681 >=.85; unused hold .98366 >=.85 |
| AE hold | 250 | PASS / 22 | Reconstruction error .003065 <=.05; hold error .001007 <=.35 |
| Ring acquisition | 400 | FAIL / 0 | Modes 16 >=16, HQ .94385 >=.85, mass TV .09302 <=.15 and full component minimum eigenvalue ratio .29107 >=.15 pass; **full component covariance error 9.61552 >.85** fails |
| Joint word acquisition | 20,001 | PASS / 6 | Quality 1 >=.95, modes 5 ==5, mass TV .01895 <=.1, reconstruction exact 1 ==1 and minimum reconstruction token probability 1 >=.9 |

The Gaussian's KS error is 2.29 times its bound even though location and width
pass. Relative to the matched control, width improves (1.23113 to 1.05499),
but KS worsens (.08328 to .11428). Matching two moments does not recover the
target CDF. Ring's full covariance error improves from 13.16405 to 9.61552,
yet remains 11.31 times its bound. It is the arithmetic mean of relative
Frobenius covariance errors over all 16 nearest-assigned components, including
assigned tail samples. Core-only covariance error .48050 and overall covariance
error .09988 are different diagnostics and cannot substitute for the failing
gate. There are 230/4,096 samples outside three target standard deviations,
versus 286 for the control. Seven components individually exceed .85; component
1 has full error 60.0491 versus core-only .68394, with spill .33333. Component
10 has full error 32.7079 versus core-only .59589. Better HQ and all modes therefore
do not establish correct local distribution shape.

The matched .01 starter has a two-pole passing terminal suffix of four;
the .012 winner extends it to 17. The winner retains both hold passes and the
word pass, so its fourth required pass does not come from combining different
task-specific configurations. Word mass TV is unchanged at the endpoint.

Stage A's 20 independent D/prior pacing configurations top out at 3/6; the
predeclared PASS-count/hash objective selects the matched starter. Stage B's
three intermediate G rates produce the 4/6 winner at .012, keeping D/G and
absolute prior pace fixed. Stage C's two positive momenta at that same pace
top out at 2/6. Momentum is shared across network parameter groups in this
study and is never applied to particle rows. The promising refinement was
the network rate, rather than evidence for a better independent prior/D pace
or positive shared momentum.

There are useful losing configurations. At G=.016, ring full covariance error
falls to 2.74482 at HQ .94312, but the word minimum reconstruction token
probability .87848 falls below .9. At G=.022, word quality remains 1 while
coverage drops to three modes. At the winning pace with mu=.5, ring HQ rises
to .94897 and covariance error falls to 2.53849, but the full minimum
eigenvalue ratio .08709 fails its .15 floor and the recipe scores only 2/6.
With mu=.9, ring loses one mode and HQ drops to .58521. Higher HQ, a lower
covariance error or valid word samples alone can hide a different failed
condition. These tradeoffs are not reasons to splice task-specific winners.

See [final results](results.json), [verified analysis and diagnostics](analysis.json),
[measurement selection](measurement-selection.json),
[actual-training GIF index](media-index.json), [archive inventory](artifact-inventory.json)
and [software verification](software-verification.json). Compact per-stage
search records preserve all outcomes and prospective continuation decisions;
raw logs, traces and states stay in the separate artifact archive.

Input-gradient validity is task-specific. Scalar and ring hosts switch the
critic to evaluation mode during generator forwards, so their corrected
critic-phase probes are usable. The word fixture leaves it in training mode;
previous generator-phase forwards can still occupy the real/fake input slots.
**Word input-gradient curves are excluded**, as are unverified behavioral-host
input labels. Actual update sizes, weight norms, spectral products and sampled
particle displacements do not depend on that labeling and remain usable. The
matrix spectral product excludes Fourier/input maps and nonlinearities; it is
not a bound on the whole critic. The executed scientific source is unchanged.

The top three complete configurations under the frozen PASS/hash objective
are G=.012, .016 and .01, all at D/G=1.5, prior=.03 and mu=0; the last is
the matched control. Their [Gaussian](diagnostics-gaussian1d_acquisition.svg),
[ring](diagnostics-ring16_acquisition.svg) and
[word](diagnostics-five_word_joint_acquisition.svg) panels show actual per-player
relative update sums, parameter norms, matrix spectral products and sampled-row
movement; valid scalar/ring input-gradient probes are included.
The winner's ring spectral-log total variation is 2.80838 versus the control's
2.93126, while its final spectral product is higher, 12.9549 versus 10.9475.
A slightly smoother curve does not imply a smaller or bounded critic. Final
ring D/G relative update sums rise from .06598/.11129 to .07733/.14586; nominal
normalized steps do not force equal relative parameter speeds. One word G/E
player matrix grows 15.51 times its first logged norm, versus 19.28 for the control.
This is a finite-budget growth flag, not proof of unbounded growth. Every
recorded full-dualnorm sampled-row probe has zero unsampled-row displacement.
This is nonvacuous at the Gaussian/ring checkpoints; word batches sample all
five rows, so their outside-row measurement supplies no unsampled-row control.

All 175 result/certificate/source/runtime/candidate/seed bindings, 150 sustained
grades, 100 saved clock comparisons and 75 diagnostic traces are independently
checked by the reader. All emitted initial component hashes match across the
25 configurations for each task; two-pole's declared fixed fixture is separate
and emits no aggregate initializer manifest. Named stream starts match across
configurations, and all 15 final word stream states match as well. Those final
states are a post-run saved-state audit. No direct consumed-batch digest was
recorded, so matching metadata and streams must not be described as a direct
byte-by-byte batch audit or as identical trained model states.

## Interpretation and next questions

**Magnitude or direction?** The earlier screen recovered Adam's 3/6 aggregate
count with normalized SGD directions; this follow-up improves the selected
dualnorm recipe to 4/6 by changing pace with its direction rule fixed. That
supports investigating normalization and player pace, but it does not isolate
Adam's direction from its magnitude: no matched Adam or graft sweep ran in
this current source cohort. Dualnorm offers one matrix/vector/sampled-row rule
across these tasks. Whether its optimal steps transfer across width and depth
remains untested; this is not evidence of a calibrated optimizer that scales.

Original P1-P6 remain unscored in their native-benchmark/five-seed/scale scope.
This seed-0 screen supplies neither a variance estimate nor a transfer plot.
The shared-momentum result opposes the suggested preference for mu=.5 within
this cohort, but does not falsify the original statistical prediction.

The pacing omission identified in [the audit PR](https://github.com/255BITS/ParticleGAN/pull/307)
is now tested at mu=0 and G=.01. This does not exhaust the available
hyperparameters: momentum was only shared, epsilon was fixed, and the ratio
and prior axes were not repeated at every intermediate G rate. A fine rate
grid could narrow the .01-to-.012 two-pole transition, but would not by itself
explain the remaining Gaussian CDF and ring tail errors. Before another paid
search, inspect the archived per-component tails, saved CDFs and update/critic
diagnostics against those exact failures. Separate network-player momentum
or different matrix/vector pacing would be new optimizer variants requiring
a fresh finite declaration. Neither loss/BCAP changes nor more training are
implied by this result.

One concrete unexecuted refinement is G={.013,.014,.015}, D/G=1.5,
prior=.03 and mu=0, between the selected four-pass rate and the .016
ring-shape/word-confidence tradeoff. These three complete Tier 1 recipes
would require a new 7,560-second maximum reservation and a declared stop
condition. This is a proposal, not an extension of the finished campaign or
a prediction that the Gaussian/ring gates will pass. Split D/G momentum or
softening small singular directions needs its own optimizer implementation
and controls; shared positive momentum did not supply evidence for that change.

## Frozen design and reproduction

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

To inspect the completed local campaign from the final PR checkout:

```sh
PYTHON=/home/martyn/dev/ParticleGAN/.venv/bin/python
STUDY_QUEUE=/home/martyn/dev/ParticleGAN/runs/forge/bcap-dualnorm-pacing-v2-queue
$PYTHON -u reports/forge/dualnorm-pacing-v2/analyze.py --queue-root "$STUDY_QUEUE"
$PYTHON reports/forge/dualnorm-pacing-v2/export_media.py --queue-root "$STUDY_QUEUE" --verify-only --once
$PYTHON reports/forge/dualnorm-pacing-v2/export_media.py --queue-root "$STUDY_QUEUE" --verify-archive
$PYTHON -m experiments.forge compile --check
```

Reporting adds no training, sampling draws or qualification. The analyzer
requires the frozen queue control records and either original attempt files
or the new archive supplied through `--archive`; absolute original clock
artifact locations are preserved in certified receipts. The archive reader
can stage their verified bytes without changing the original receipt.
Use `select_measurements.py --snapshot <source-snapshot.json>` to review a
proposed complete selection card without installing it. Original selections,
numerical source snapshots, results and the raw archive are retained; the
old v1 archive is verified unchanged. The new raw archive is local only,
with exact availability and member hashes in its inventory. Fresh-checkout
leaderboard regeneration uses committed compact numerical snapshots, not
the unavailable raw queue, and must run without a backend filter.
