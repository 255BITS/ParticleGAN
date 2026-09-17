# Component-count and budget follow-up

Exploratory follow-up authorized September 16, 2026. Question: can more components
and/or more training match or beat the original 20k-particle C0, and on which metrics?

Use seed 1 throughout; no new seed-only repetitions. Reuse existing seed-1 C0 and
N=400 references, with historical C0 three-seed mean/range for context. A successful
configuration is a single-run result, not an estimated pass rate or robust winner.
Keep the frozen C0-envelope rule and also compare HQ, absolute width error from 1,
KL, table size, and training cost separately. Passing the envelope does not imply
dominating C0 on every dimension.

## First screen: 13 new 7k runs

- N = 1,600 / 6,400 / 20,000 crossed with r = 0 / 1/40 / 1/16.
  Standardized means, selected small-table optimizer (particle LR multiplier 10,
  beta1=0.5). Matched zero-noise controls at every N.
- At N=20,000 with shipped optimizer (multiplier 1, beta1=0): standardized
  r=0 and r=1/40, unstandardized r=1/40 and r=1/16. Existing C0 supplies the
  unstandardized r=0 reference.
- Full raw-table VICReg applies only at N<=1,024; all larger tables use the
  existing sampled-row recipe. Cross-boundary comparisons include this difference.
- At fixed r, initial median NN distance and therefore absolute sigma change
  with N. Record both; this is not an experiment at fixed absolute sigma.
- Unstandardized arms deliberately allow the scale escape. Inspect raw std,
  sigma/final NN distance and flags before interpreting improved HQ as success.

Time one large-table run first. Provisional screen estimate 5–8 minutes on two
A6000s with two workers each, revised from measured timing before launching the rest.

Measured large-table run: 77 seconds including runner overhead (74.9 seconds in
the trainer). Remaining twelve runs estimated at 4–6 minutes before launch.

## Adaptive optimizer check: four additional 7k runs

The N=20k fast-optimizer atoms control also failed badly, so failures there cannot
be attributed only to adding Gaussian noise. N=1,600 small-noise MoG improved over
its atoms control but remained too narrow and imbalanced. Before selecting longer
runs, add N=1,600 and 6,400 with shipped particle LR/beta at r=0 and r=1/40,
retaining standardized reads. Estimated additional wall time: 90 seconds. These
are configuration comparisons at the same seed, not seed-only repetitions.

## Adaptive longer-budget follow-up

Select useful positive-noise candidates after the screen, using frozen pass status,
then HQ, then width error, while inspecting balance. Include a matched atoms control
and C0 at the same longer budget. Also test whether the previous N=400, r=1/40
setting benefits from more training. Announce concrete configs and timing before
launch. These are fresh same-seed runs with the cosine schedule scaled to budget,
not resumed EMA checkpoints. Preserve all screened results, including failures.

After all 17 screen configurations completed: select the only passing MoG,
N=20,000, r=1/40, shipped optimizer, standardize=False. At 28k steps also run
the same recipe with r=1/16 (tests whether training can recover the wider-noise
setting), C0 (its matched r=0 control), and the previous N=400 r=1/40 winner
(standardize=True, LR multiplier 10, beta1=0.5). Four new runs, seed 1 throughout;
estimated 5–6 minutes before launch. C0 is reused as the matched 20k atoms
control rather than duplicated. The N=400 extension tests its budget response;
there is no matched 28k N=400 atoms control in this batch.

The first screen completed in 4.1 minutes after the initial timed run; the
four-config optimizer follow-up completed in 1.3 minutes. All execution checks
passed. Only the unstandardized 20k r=1/40 setting passes the frozen envelope
at 7k. Standardized intermediate tables recover width with the lower optimizer
LR but still miss balance. All outcomes remain in the results, not only winners.

## Artifacts and logs

Shared trainer and grid runner; no training implementation changes. Final metrics
use 200k EMA samples and matched real samples; traces use 20k every 100 updates.
Configs, sources, final weights, sampled outputs, component diagnostics, provenance,
and runner completion certificates remain in each output directory.

```bash
tail -F results/mog/component_scale.runner.log
tail -F results/mog/component_scale/n20000_r1over40_fast_7k_s1/log.txt
tail -F results/mog/scale_longer.runner.log
```

Do not change `results/mog/results.csv`, the frozen Stage 0 baseline input.
