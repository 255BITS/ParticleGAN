# Fast, stable defaults study

This study follows the original [62-configuration search](../search/README.md):
measure convergence, test schedules, measure sensitivity, validate on the actual
100-Gaussian task, then expose the supported recipe through the public API.
All experiments use seed 0. Live weights determine success; EMA stays separate.

## 1. Measure convergence

The [measurement comparison](measurement/README.md) reruns the original locked
config, R1+R2 reference and best cap. All **27 candidate/toy results match their
historical final metrics exactly** ([parity evidence](measurement/parity.json)).
Each toy now records 24 observations without changing its training RNG stream.

Sustained PASS means the complete expected curve contains a final stretch of
at least five passing observations, with no later measured failure. Ring
convergence requires **8/8 modes and HQ ≥90%**. Other hosts retain their original
29-bound definitions. The existing regression thresholds are unchanged.
An interrupted curve or missing interior observation cannot certify convergence.

The report shows first PASS, start of the final passing stretch, and the fifth
observation that confirms it. Timing includes setup and measurements. These are
observed results through the budget, not guarantees between observations or
beyond the run. A transient pass does not earn a convergence speed.

| Config | Final numerical suite | Sustained toys | Ring sustained PASS |
| --- | --- | ---: | --- |
| Original locked | 7/9 toys; 26/29 bounds | 7/9 | Not reached |
| R1+R2 0.1 | 9/9; 29/29 | 8/9 | Not reached |
| Cap 1.25, coeff 3, no L2, LR ×0.85 | 9/9; 29/29 | 8/9 | Not reached |

## 2. Schedule comparison

The next comparison preserves data, capacity, seed, step budgets and metric
thresholds. The new `lr_schedule: "cosine"` option **replaces** the host's
schedule on every optimizer, using the initial learning rate of each parameter
group. It does not multiply an already decaying schedule. `"host"` preserves
all original schedules. The shared implementation uses ParticleGAN's existing
`learning_rate_scale` function, with zero completed updates before the first
optimizer step.

The initial comparison is the previous cap leader plus seven cosine candidates:
start at 40%, 60% or 80% of the budget, floor 5%, base LR multipliers 0.85, 1 or
1.15. Each candidate uses one policy across all nine hosts. All nine hosts run;
no failed trial is discarded. Phase artifacts preserve the source fingerprint
at execution rather than pretending old measurements came from new code.

Selection prioritizes a complete 29/29 final numerical pass, full final ring
coverage, and a sustained ring PASS. Among eligible candidates, prefer more
sustained toys and earlier confirmation; inspect quality and balance alongside
speed. No schedule or checkpoint is selected independently for different toys.

The [completed eight-config comparison](schedule/README.md) selects
**cap 1.25, coefficient 3, no L2, LR ×0.85, cosine from 60% to a 5% floor**.
It passes **29/29 final bounds and all nine sustained-pass targets**. The ring
holds 8 modes and HQ≥90% from step850 to1200, confirmed at1050 (5.61 seconds
in this run). Final live and EMA ring quality is8/8 at100% HQ. The original
host-schedule control exactly reproduces the measurement phase's final metrics.

Decay beginning at80% also passes every sustained target, but confirms ring
convergence later, at1200. Beginning at40% fails two-pole motion. Increasing the
base learning rate to1 or1.15 does not produce an earlier sustained full-suite
success. All results, including failures, are retained. The
[resolved leading config](leading_config.json) is the starting point for the
next phase's one-variable-at-a-time ±10% sensitivity tests and scaled ring data.
