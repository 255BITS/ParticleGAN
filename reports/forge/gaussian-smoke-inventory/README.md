# Post-merge CUDA smoke inventory

**Gaussian acquisition is now passing; no whole recipe unlocks ordinary Tier 2.**
The completed seed-0 CUDA rerun records Gaussian smoke **19 PASS / 3 FAIL /
1 runtime BLOCKED**. The two preselected BCAP and DualNorm configurations each
pass **5/6** required Tier 1 tasks. Both fail ring16. Across the complete submitted
roster, ring16 records **0 PASS / 22 FAIL / 1 runtime BLOCKED**, so no recipe meets
the six-task barrier. All runnable Tier 1 peers finished; **zero ordinary Tier 2
jobs were eligible**, and none ran. The separate shallow and Fourier diagnostic
continuations already ran through 6,000 updates and failed stability.

[Final source-bound readout](final-v4/README.md) ·
[Every candidate's final metrics and blockers](final-v4/readout.json) ·
[Single generated technique inventory](../technique-inventory.md) ·
[Actual-training GIFs and provenance](media/index.json)

The source frozen after [PR328](https://github.com/255BITS/ParticleGAN/pull/328)
is `79fdf16d2ed880a9db1873245f150375e3be31b0`, numerical digest
`d276c5a7344fab6ec5de7b314d3982af5b0ef8027c8b89c01376ae366844a9cb`.
This round admits 23 recipes from 52 declarations and completes 161 attempts:
158 completed numerical/diagnostic runs and three public-API capability errors.
Every actual worker uses CUDA; every completed task retains a certified full
state and consumed RNG streams. The three errors belong to
`five-word-joint-ka2-v1`: its declared `public_components` context cannot own the
scalar trainer needed by Gaussian, ring and the nonrequired clock diagnostic.
Those cells remain BLOCKED, with their original errors and cost, and receive no
numerical verdict. The other 24 preflight blockers and five declaration refusals
remain visible. There are zero scientific retries or seed comparisons.

The successful smoke/stability split, CUDA behavioral hosts, positive shallow
diagnostic and execution/provenance repairs were merged in
[PR322](https://github.com/255BITS/ParticleGAN/pull/322),
[PR324](https://github.com/255BITS/ParticleGAN/pull/324),
[PR325](https://github.com/255BITS/ParticleGAN/pull/325),
[PR326](https://github.com/255BITS/ParticleGAN/pull/326),
[PR327](https://github.com/255BITS/ParticleGAN/pull/327) and
[PR328](https://github.com/255BITS/ParticleGAN/pull/328).
The negative Fourier-removal [PR323](https://github.com/255BITS/ParticleGAN/pull/323)
remains unmerged; its immutable source, metrics and archive are retained in the
[external recall record](../records/gaussian-no-fourier-v1-external-readout.json).

## What the Gaussian result means

The default stays at z dimension 2, width 32, depth 2 and two critic Fourier
frequencies: 1,185 generator and 1,281 critic parameters. The learned prior has
256 uniform MoG components, sigma .1, no standardization and initial scale 1.
Target N(2,.5²), batch 128, initialization and evaluation cadence remain fixed
across recipes. Selected DualNorm uses constant G/D/prior rates .012/.018/.03;
the continuous learner is neither stopped at a passing state nor annealed.

Selected DualNorm passes two of the 24 independently confirmed full-law states.
Its final primary draw at update 1,000 has KS .042974, mean error .013037 target
sigma and width ratio .957388. Selected BCAP passes three paired states while
its final primary KS is .062681, above .05. That is exactly the smoke distinction:
acquisition can succeed before the endpoint without claiming that learning stays
there. All 1,000 updates and all 24 confirmation pairs still execute.

[One hidden layer](../gaussian-shallow/README.md) also acquires, at updates 584 and
792, but fails stationary hold, shifted reacquisition and shifted hold. Removing
Fourier features misses the 1,000-update smoke budget. These results support
keeping the existing small architecture and separating acquisition from
continued stability. They do not identify which moving generator, critic or
prior causes the drift, and provide no ordinary Tier 2 qualification. The
screening profile remains provisional pending calibration.

## What still blocks the families

Selected DualNorm's ring endpoint has all 16 modes, mass TV .057129 and
3-sigma quality .936035. Its full component covariance error is **2.220268**, above
the **.85** gate; component 11 contributes 29.243107. The core-only covariance
diagnostic is .387413, but it cannot replace the full-distribution gate. This
row has **zero full passes among 96 scheduled ring observations**. Merely moving
the suffix/hold requirement would not make this run pass.

The [saved ring binding audit](ring-binding-audit.md) explains why an earlier
1600-update PASS cannot fill this current FAIL: the archived success extended a
400-update checkpoint, while this rerun trains uninterrupted. Recipes, prior,
initialization and the first 24 saved draws through 400 match exactly; the curves
diverge after the restore boundary. The audit establishes that difference, not
its cause. Historical verdicts and source identities remain unchanged. The final
v4 ring endpoint reproduces the audited v3 numerical metrics under its own
complete provenance.

Keep the Gaussian default and the smoke/Tier 2 split. Stop the failed Fourier
removal and depth-only stability arm. Prioritize a declared ring local-spread,
tail/spill and restore-continuation investigation next; visual coverage alone
misses the covariance failure. Separately repair capability declarations for
component-only candidates before attempting scalar tasks. Neither recommendation
changes this round's gates, recipes or qualification results.

## Frozen scope and reproduction

This round reruns every registered idea and the 13 previously selected complete
family configurations against the new Gaussian acquisition smoke and CUDA
behavioral hosts. The Gaussian default keeps depth 2 and two Fourier frequencies:
the original recipe already demonstrated acquisition, the shallow alternative
also acquires but drifts, and removing Fourier features misses the smoke budget.

The frozen roster contains **52 declarations: 39 ideas and 13 configurations**.
Historical tuning grids are not repeated. The selected configurations retain
their exact recipe identities and are registered through one-point searches;
this round does not tune hyperparameters. Protocol seed 0 and the public
deterministic initializer apply throughout. Architecture, prior, target,
sampling law, seen batches, update budget and evaluation cadence are fixed per
task across trainer candidates. Each candidate supplies one global recipe.

All runnable Tier 1 peers finish before a failure blocks higher tiers. A whole
candidate must pass all **6 required Tier 1 tasks** before its eligible **20
required Tier 2 tasks** run. Task-specific checkpoint prerequisites still apply.
Diagnostic clock results do not substitute for required gates. Tier 3 is outside
this round. The Gaussian stability task restores its own candidate's exact
1,000-update smoke state, continues without resetting history and tests the
stationary and shifted goals.

Four old idea contracts and one selected configuration refuse resolution because their control maps do not bind
the new tasks. Their exact declarations and refusal reasons remain in the round;
they receive no attempt or scientific verdict. Draft and unregistered study
blockers are likewise retained. Independent runnable declarations proceed
through the ordinary Forge APIs without weakening those contracts.

Each declaration has a conservative 42,720-second complete allowance, for a
2,221,440-second maximum campaign reservation. These are task timeout ceilings,
not measured runtime. Scientific retries are zero. Software planning occurs
before the merged execution source is frozen and any training starts.

```sh
/usr/bin/python -u reports/forge/prepare_gaussian_smoke_inventory.py plan \
  --queue-root runs/forge/gaussian-smoke-inventory-v4
/usr/bin/python -u reports/forge/prepare_gaussian_smoke_inventory.py enqueue \
  --queue-root runs/forge/gaussian-smoke-inventory-v4
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/usr/bin/python -u - <<'PY' > runs/forge/gaussian-smoke-inventory-v4/drain.log 2>&1
from pathlib import Path
from experiments.forge.queue import Queue, drain
queue = Queue(Path('runs/forge/gaussian-smoke-inventory-v4'),
              report_root=Path('reports/forge'), on_completion=None)
drain(queue, ['0', '1'], campaign='gaussian-smoke-inventory-v4')
PY
tail -F runs/forge/gaussian-smoke-inventory-v4/events.jsonl
```

The original v1 admission stopped before creating any submission, attempt or training update: the host validator conflated the 6,000-update continuation budget with its retained 1,000-update schedule horizon. [The refusal/archive receipt](admission-v1.json) preserves that source and zero cost. The v2 successor changes only this software validation and the registration identity; roster, recipes, task budgets and campaign ceilings remain fixed.

The v2 run then exposed an older generic vector-adapter omission: ring execution
stopped at its 400-update schedule horizon before the declared 1,600-update
allowance. Dispatch stopped, leaving active tasks to finish and retaining every
partial result and cost. The v3 software amendment passes the explicit task
execution allowance to the public trainer, preserving its recipe schedule.
The complete fixed roster runs under the repaired source because ordinary
qualification cannot splice task passes from different sources. This is an
execution repair, with no recipe selection, seed changes or retries of scientific
failures. The combined conservative executable allowance fits inside the
original 2,221,440-second goal ceiling.

The v3 route audit then found missing mandatory checkpoint provenance: generic
vector/image tasks discarded the full context when they were not continuation
parents. Word and behavioral hosts already retained states but did not bind them
to receipt hashes; the direct-particle behavioral state also omitted its standalone
coordinates. Dispatch stopped, both active workers completed, and all 62 results
were retained at 2,897.205946 seconds with zero remaining reservation.
[The exact archive receipt](provenance-interruption-v3.json) preserves this cohort.
The v4 amendment retains certified full states and every consumed named RNG
stream, independently of continuation eligibility. It repeats the fixed roster
under one repaired source; numerical settings and gates remain unchanged.
The previous v2/v3 paid total is 3,708.623331 seconds, and the combined conservative
executable allowance is 986,268.623331 seconds, inside the original goal ceiling.

The preparation interface launches no training. The parent runs the reviewed
campaign with one worker on each physical GPU, preserves raw stdout, traces,
states and original receipts in an artifact archive, and publishes compact
metrics and actual-training GIFs. No raw log is committed.

Publication uses the existing
[regeneration script](../regenerate_technique_inventory.py). New whole-row family
pins select fresh measured evidence; old policy/source outcomes remain archived.
The only generated goal leaderboard is
[technique-inventory](../technique-inventory.md). Whole-family choices retain the
pre-run recipes; no task passes are pooled across configurations or sources.
Publication archives view revision 5 and registers these outcomes under revision
7 with the existing script. No revision 6 measurement is invented.

The first publication attempt stopped before writing the new evidence manifest:
board deduplication preferred `bcap-pure-adam-v1` over its scientifically
equivalent registered canonical `bcap-pure-adam-v2`. Publication now preserves
exact recorded identities first, then prefers canonical declarations among
unmeasured equivalents. The existing explicit unmeasured CUDA display contract
collapses BCAP-pure's declaration-only runtime groups without changing their
scientific rows. Its canonical still requires a registered study before enqueue;
the refused preselected configuration keeps its original control-map blocker and
unbound source. No numerical source, task, recipe, prior, gate or training is
changed by this publication repair. Focused checks verify ordering and restoration
of the declaration resolver after errors.

The [final archive receipt](archive-v4.json) binds 3,981 original files in
`artifacts/gaussian-smoke-inventory-v4-final.tar.gz`, SHA256
`1531a5de3ee93e1bb216060eb25967010a2381c76ff0bc873267580851ba5c21`.
The final run costs 6,136.617596 measured runner-seconds; retained v2/v3 repair
cohorts cost 3,708.623331 seconds, for **9,845.240927 seconds total**, within the
original ceiling. Archive creation verifies every member. Seven GIFs cover all
six measured Tier 1 goals plus the clock diagnostic using existing certified
observations, including the failing ring. Publication performs no training,
model forward or prior draw; independent regrading uses the archived outputs.

After hydrating the byte-exact archive, reproduce the saved readout with:

```sh
/usr/bin/python reports/forge/collect_gaussian_smoke_inventory.py --root . \
  --queue-root runs/forge/gaussian-smoke-inventory-v4 \
  --round configs/forge/rounds/gaussian-smoke-inventory-v4.json \
  --archive-receipt reports/forge/gaussian-smoke-inventory/archive-v4.json \
  --publication-prefix reports/forge/gaussian-smoke-inventory/final-v4 \
  --output runs/software/gaussian-smoke-inventory-v4-readout
```

The publication is registered with:

```sh
/usr/bin/python reports/forge/regenerate_technique_inventory.py --device cuda \
  --source-commit 79fdf16d2ed880a9db1873245f150375e3be31b0 --advance-policy
```

Later cached regeneration uses the same script without `--advance-policy` or a
new source registration. [Final publication verification](publication-verification.json)
records byte-identical cached regeneration of all 26 generated outputs, CURRENT
compiled memory, complete catalog coverage and valid documentation for all 22
current/historical families. Compact interrupted-source search summaries and
display receipts retain their original identities so this memory is reproducible
from the committed checkout. They add no qualification or new-source gate credit.
The source-only repairs already passed 33 checks
(29 CUDA and four metadata checks); the final publication/archive integration
passes another 68 metadata checks, including canonical ordering and destructive
unmeasured-display controls. Saved rendering explicitly prohibits neural
execution. Raw stdout, JSONL, JUnit, checkpoints and tensor dumps remain ignored
or archived, outside Git.
