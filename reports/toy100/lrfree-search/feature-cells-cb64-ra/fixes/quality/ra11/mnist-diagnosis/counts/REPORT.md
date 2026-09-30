# RA11 MNIST recorded-scalar diagnosis

JSON diagnosis PASS; the recorded MNIST result is a failure. No new scoring,
Torch/PT/array import, model forward, action, training update, or seed was used.
Both extraction helpers and their closed input maps were sealed before execution.
The source-preparation path error is preserved under preparation-attempt1; it
occurred before a derived timeline ran.

## Matched reference and timeline

RA11: validation-cb64-ra11/learned/training/mnist/CB64-RA11.
RA4: validation-ra4/learned/training/mnist/CB64-RA4.
The matched E22 is **/ml2/hypergan/gan-attempts/feature-cells-cuda-retest-20260929/learned/training/mnist/E22**.
All three share recorded initial G/D/prior hashes, seed314159, the2000-update
schedule, classifier, and all active39 mask/mean/std/reference hashes. The later
evaluator uses std>1e-6*max(training std) and normalizes only those39dimensions.

The separately preserved first extraction used the older
/ml2/hypergan/gan-attempts/scaling-portability-20260929/validation/runs/mnist/E22.
That evaluator normalizes every embedding dimension with std.clamp_min(1e-4).
Its native FD and54moves are descriptive only; they are not substituted for the
matched E22 active FD or415moves below.

| Update | RA11 active FD / confident classes | RA4 active FD / classes | Matched E22 active FD / classes |
|---:|---:|---:|---:|
|100|109.453 /0|54.789 /2|199.791 /1|
|250|108.793 /0|14.961 /3|29.939 /1|
|500|107.134 /0|2.851 /9|6.010 /8|
|1000|101.021 /0|0.819 /10|1.799 /10|
|2000|40.544 /1|0.393 /10|0.544 /10|

All methods have poor early metrics; the persistent failure to recover is clear
by250–500. RA11 active recall stays0 at every original milestone. Its high
precision at1500/1750 (.937/.973) accompanies recall0 and confident coverage0,
so it does not demonstrate broad recovery.

| Final recorded metric | RA11 | RA4 | Matched E22 |
|---|---:|---:|---:|
| Active FD |40.544410|0.393402|0.544488|
| Active precision / recall |.281250 /0|.895508 /.786133|.869141 /.847168|
| Confident class coverage |1|10|10|
| Pixel clipping fraction |.563290|.407950|.409068|
| Output sigma |1.310714|.029000|.029000|
| Accepted row moves |0|824|415|

## Reaction and serving attribution

RA11 has250reactions, **0mean witness firings,0mean previews,0mean copies**.
Total ordinary/isolation/novel accepted moves are also0. There are1000bounded
novel-cell attempts with no accepted birth,12954categorical discoveries, and
256000isolation flags (250*1024): every reaction flags the full1024-row table.
RA4 performs469ordinary moves by250,782by500, and824by2000. The matched E22
performs0by500,198by1000 and415by2000. These different laws are descriptive
comparators, not interchangeable transport certificates.

At each saved100–2000 milestone, the RA11 mean status is invalid:
missing_or_insufficient_even_group. The source vetoes if **any** fitted learned
group has fewer than2even-real rows, before scalar EB evidence is available.
The stored scalar mean/variance/lower bound are null. An executed output-moment
correction cannot explain this failure. Invalid endpoint reasons do not establish
the reason at every unsaved reaction; cumulative zero fires establish no trigger.

Observed RA11 geometry has requestedK128, actualK64, chart rank8, moment rank8,
512even/512odd real rows, and16–42real topology groups. The output sketch selects
8of784coordinates (1.02%) per observed reaction; the nine reacted milestones
cover31distinct coordinates (3.95%). This is coordinate-count coverage, not
explained variance, semantic coverage, or evidence of a faulty projection.

At all those milestones, FAST supported-parent eligibility is0. The paired lease
is false: all1024rows agree on the learned group, but EMAeligible/coherent rows
are0 against973required. Current leases are refreshed; they fail support, not a
positive lease expiring. RA11 therefore serves FAST at the recorded endpoints.
The cumulative table tester has no stationary decisions or population certificate.

Recorded sigma increases .03294@100, .05841@500, .13878@1000, .45775@1500,
.85615@1750,1.31071@2000; RA4/E22 reach the original .029floor by500.
RA11 nominal G+sigma base rates are .0010625 versus .00425 for RA4/E22, while
prior/D base rates are preserved by their multipliers. This association does not
isolate rate, generator dynamics, noise optimization or support calibration as
the cause; no causal intervention was run.

## Smallest discriminating saved-state proposal — NOT RUN

One fixed CPU support/occupancy audit at the existing RA11 checkpoint0250:
bind source/input/helper before reading PT, restore raw FAST/EMA/current D,
construct exactly one descriptive current chart from saved real FIFO and a
private clone of its saved stream, using existingK/rank/Q/cap. Measure only
even-real counts/scales per learned group, calibrated odd-real support summaries,
and clean FAST/EMA conformity/inside/parent-pool counts against that same chart.
No samples, output-noise draws, actions, birth solver, classifier/scorer or optimizer.

This separates empty/singleton real-chart groups and calibration exclusion from
genuinely unsupported clean anchors. If real support is sound and both clean
views lack eligible parents, it supplies no grounds to relax support or enable
mean copies; training dynamics remains the separate investigation. The ephemeral
historical chart is not saved, so this is a current-view test, not reconstruction
of the historical reaction. No such test has been executed or approved here.
