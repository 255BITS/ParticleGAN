# BCAP optimizer screen

This branch tests optimizer changes on the current **BCAP pure** recipe through Forge's
unchanged public API and full current Tier 1. The user selected this focused
stage before the larger native, regularizer, seed and scale-transfer study.

The finite screen contains **41 global configurations**: five Adam rates, five
plain SGDA rates spanning four decades, four global-normalized rates, four
tensor-normalized rates, four Adam-magnitude graft rates, four dualnorm rates
at each of momentum 0/.5/.9, four D-only dualnorm rates, and three prior-only
row-normalized rates. Every configuration completes all runnable independent
Tier 1 tasks despite scientific failures; none enters Tier 2. The existing
clock diagnostic remains separate from the six required tasks.

The current baseline is the public BCAP preset: relativistic logistic loss,
fixed BCAP coefficient/cap 1, native Adam beta=(0,.999), rate .00425, D multiplier
1, prior multiplier 2, constant rate/moment paths, and no additional training
noise, prior regularizer or EMA selection. Behavioral task-owned objectives
remain unchanged. The historical .0006/delayed-cosine example is a different
cohort. Only optimizer algorithms and their explicit step sizes change here.
Hybrid arms pin the unchanged Adam players at the current baseline rates.

Protocol seed 0, public deterministic initialization, architecture, data law,
named stream bindings, priors, batch sequences, update budgets and evaluation
cadence are matched. All grading uses the task's existing clean/live law.
The screen is provisional and supplies no default adoption or robustness claim.
Existing failing tests are retained; their repairs belong to another PR.

The first-stage campaign reserves at most **103,320 seconds**, with a complete
2,520-second allowance per configuration. This is a worst-case ceiling rather
than expected elapsed time. Two GPU workers and one CPU slot use the existing
queue resource policy. Wall time remains cost evidence, never a speed ranking.
Numerical failures are final; no automated paid retry, seed repetition or edge
extension is authorized by this campaign. A useful edge result can justify a
separately declared finite follow-up.

[Study declarations](../../../configs/forge/searches/bcap-optim-adam-tier1-v1.json)
and sibling `bcap-optim-*-tier1-v1.json` files freeze every grid. [Plan](plan.json)
records the resolved configuration and task identities. Final compact metrics
are written to `results.json`, with per-study selections in the existing
configuration-search reports. The [current family leaderboard](../technique-inventory.md)
remains the single leaderboard for this goal; this study creates no second one.

RNG-free diagnostic observers retain actual update/weight/gradient norms,
matrix spectral products and input-gradient measurements at the declared
observation steps for scalar, ring and joint-word hosts. G/E form one reported
player; the prior is separate. Spectral products exclude Fourier/input maps
and nonlinearities and do not establish a bound on the whole critic.
Unsampled-row measurements use actual row IDs when available; native Adam's
gradient-support measurements are explicitly labeled differently. Raw traces
stay in the ignored queue and archive. Software tests verify exact model,
optimizer and RNG invariance with the observers enabled.

The executed source is `15eb7cb0911905e401bdfcd7e264945a7ea64d97`.
Its optional input-gradient observer can capture generator-phase forwards as
the following critic step's inputs. Those measurements are excluded from this
readout. The correction and its routing regression test are included for future
runs; original receipts and training outcomes retain their executed source.
Update-size, weight-norm and spectral-product observations remain usable.

Original predictions P1–P6 retain their native-benchmark and five-seed scope.
Tier 1 observations can motivate them, but cannot confirm or falsify the stated
five-seed claims. R1/R2, 7k native100, sparse177, confirmation and width/depth
transfer remain explicitly deferred until a promising supported arm exists.

The commands below describe the executed workflow. To reproduce its numerical
cohort, use the recorded commit above and a fresh local queue. A future study
using the observer/API corrections must record its own source identity; these
results do not qualify the final PR's corrected implementation.

Run after committing the scientific source, from the project environment:

```sh
python reports/forge/dualnorm-tier1/run.py prepare
python reports/forge/dualnorm-tier1/run.py plan --queue-root "$PWD/runs/forge/bcap-dualnorm-tier1-v1-queue"
mkdir -p runs/forge/bcap-dualnorm-tier1-v1-queue
python -u reports/forge/dualnorm-tier1/run.py run --queue-root "$PWD/runs/forge/bcap-dualnorm-tier1-v1-queue" > runs/forge/bcap-dualnorm-tier1-v1-queue/driver.log 2>&1
tail -F runs/forge/bcap-dualnorm-tier1-v1-queue/driver.log runs/forge/bcap-dualnorm-tier1-v1-queue/events.jsonl
python reports/forge/dualnorm-tier1/run.py report --queue-root "$PWD/runs/forge/bcap-dualnorm-tier1-v1-queue"
python reports/forge/dualnorm-tier1/run.py media --queue-root "$PWD/runs/forge/bcap-dualnorm-tier1-v1-queue"
python reports/forge/dualnorm-tier1/run.py archive --queue-root "$PWD/runs/forge/bcap-dualnorm-tier1-v1-queue"
```

To export completed attempts while the queue continues, use the resumable reader
below. It records processed IDs locally, verifies existing GIFs, and adds only
newly completed saved observations. It verifies every original certificate,
GIF and input hash before finishing. `run.py media` remains the from-scratch
reproduction command; archiving follows completion of all workers.

```sh
OMP_NUM_THREADS=1 .venv/bin/python -u reports/forge/dualnorm-tier1/export_media.py --queue-root runs/forge/bcap-dualnorm-tier1-v1-queue > runs/forge/bcap-dualnorm-tier1-v1-queue/media-export.log 2>&1
tail -F runs/forge/bcap-dualnorm-tier1-v1-queue/media-export.log
.venv/bin/python reports/forge/dualnorm-tier1/export_media.py --queue-root runs/forge/bcap-dualnorm-tier1-v1-queue --verify-only --once
```

For a compact live tail without the per-observation arrays:

```sh
tail -F runs/forge/bcap-dualnorm-tier1-v1-queue/events.jsonl | jq --unbuffered -r 'select(.event == "claimed" or .event == "completed") | [.timestamp, .event, (.candidate | split("--") | .[0] + ":" + (.[1][0:8])), .task, (.verdicts // {} | tojson)] | @tsv'
```

The driver verifies that all captured scientific bytes exist at the recorded
commit before admission. Actual-training GIFs render existing certified
observations and add no sampling or optimizer updates. The final archive receipt
records exact member hashes and preserves original failures. Compact reports,
provenance, reproduction sources and GIFs may be committed; raw logs, JSONL,
states and checkpoints remain local or in the artifact archive.

To resume already admitted work after an interruption, drain its existing
immutable requests rather than enqueue against changed source:

```sh
python -u reports/forge/dualnorm-tier1/run.py resume --queue-root runs/forge/bcap-dualnorm-tier1-v1-queue --gpus 0,1 > runs/forge/bcap-dualnorm-tier1-v1-queue/resume.log 2>&1
```

The 2026-10-05 environment switch stopped the first coordinator. Forge collected
the workers that had finished, and the restored CUDA environment resumed the
remaining frozen requests. One interrupted CPU worker received an explicit
execution-repair retry after access was restored, within the original budget;
its earlier incomplete receipt and cost remain in the attempt history.
Completed and numerically failed attempts were not rerun. The archive retains
the queue state, original receipts and recovery logs.
