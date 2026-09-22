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
holds 8 modes and HQ≥90% from step 850 to 1,200, confirmed at 1,050 (5.61 seconds
in this run). Final live and EMA ring quality is 8/8 at 100% HQ. The original
host-schedule control exactly reproduces the measurement phase's final metrics.

Decay beginning at 80% also passes every sustained target, but confirms ring
convergence later, at 1,200. Beginning at 40% fails two-pole motion. Increasing the
base learning rate to 1 or 1.15 does not produce an earlier sustained full-suite
success. All results, including failures, are retained. The
[resolved leading config](leading_config.json) is the starting point for the
next phase's one-variable-at-a-time ±10% sensitivity tests and scaled ring data.

## 3. Sensitivity and data units

The [nine-config sensitivity study](sensitivity/README.md) changes one setting
by ±10% around the scheduled leader, holding everything else fixed. Four of the
eight neighbors pass all 29 final bounds; three of those also sustain every toy.
The cap target tolerates both directions and LR tolerates +10%. Penalty strength
fails in both directions; reducing LR or VICReg also loses ring coverage.
Increasing VICReg passes finally but does not sustain ring quality.

The [physical-unit probe](scales.json) changes both ring radius and data noise,
keeping the quality radius at three sigmas and keeping the model/config fixed:

| Data scale | Live modes | HQ | Sustained ring |
| ---: | ---: | ---: | --- |
| 0.5 | 7/8 | 83.47% | No |
| 1 | 8/8 | 100% | Yes |
| 2 | 6/8 | 59.45% | No |

Scale 1 reproduces the full-suite ring exactly. This exposes a real units
sensitivity, not a changed quality threshold. **The small-host winner is not a
forgiving universal default.** Keep it as a measured regression reference;
validate transfer on the actual 100-Gaussian task before choosing a public preset.
Input units and preprocessing must be documented rather than hidden in a claim
that the same penalty setting works for arbitrary data scales.

## 4. Actual 100-Gaussian transfer

The [matched GPU comparison](grid/README.md) runs the actual existing example,
not the eight-mode host. Each arm gets 7,000 steps, 20,000 particles and batch 256
on one RTX A6000, seed 0. Original metrics and source hashes are recorded.

| Recipe | Live modes | Live HQ | Stable from / confirmed step | Training seconds | Live core-width ratio | SW1 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Stock | 100/100 | 98.21% | 6,000 / 7,000 | 65.4 | 0.934 | 0.144 |
| Full toy transfer | 100/100 | 98.45% | 6,000 / 7,000 | 69.5 | 0.844 | 0.147 |
| Penalty only | 100/100 | 98.25% | 6,000 / 7,000 | 66.7 | 1.009 | 0.132 |

All three have five passing late observations; EMA also passes separately.
The full transfer improves worst-tail HQ but does not reach sustained live
quality earlier. Its cluster cores are narrower. Penalty-only has a core width
closer to the target and lower SW1, with slightly worse mode balance. These are
tradeoffs, not a uniform speed/quality win. Timings are individual measurements,
not repeated throughput estimates. No seed sweep was performed.

Keep the existing stock production defaults. Expose the full transfer as a named
behavioral candidate, and make the common recipe easy to apply correctly in one
training helper. The 100-Gaussian entry point now accepts the cap target and can
record live/EMA metrics without plotting; a test confirms that an observer
cannot perturb the short-run model updates through RNG consumption.
