# Coverage and default selection

**Logistic RP + `b_cap` now passes all nine toys and all 29 numerical bounds
with 8/8 live modes at 100% HQ.** The leading final-step config uses cap target
1.25, penalty coefficient 3, no particle L2, VICReg 0.05, cover 1.5 and a common
learning-rate multiplier of 0.85. Its EMA also has 8/8 modes at 100% HQ.
It outperforms the R1+R2 reference on final coverage while preserving every
original regression bound. All ten independent application checks pass.

This is a promising replacement **candidate**, not yet a stable-default result.
It keeps all eight modes at every measured late checkpoint, and meets the
stricter eight-mode/HQ target at four of five. The one miss is 83.91% HQ.
The [leaderboard](README.md) reports actual coverage, HQ, balance and stability;
the [search ledger](search/README.md) retains every attempted setting and its
failures, including incomplete screening runs. Thresholds and budgets are unchanged.

Rerun the leading candidate using [its resolved config](leading_config.json):

```bash
python -m benchmarks.locked_shared.baseline \
  --configs reports/behavioral_baseline/leading_config.json \
  --reference /path/to/conceptmod --output /tmp/leading_cap_verification
```

## Best cap configurations

| Config | Live toys / bounds | Live ring | Live effective modes | EMA ring | Late full-coverage/HQ checks | Worst late HQ |
| --- | --- | --- | ---: | --- | ---: | ---: |
| Cap 1.25, coeff 3, no L2, LR ×0.85 | 9/9; 29/29 | 8/8; 100% HQ | 7.54 | 8/8; 100% HQ | 4/5 | 83.91% |
| Cap 1.25, coeff 2, no L2, LR ×0.85 | 9/9; 29/29 | 8/8; 100% HQ | 7.51 | 8/8; 100% HQ | 3/5 | 41.89% |
| Cap 1.25, coeff 2, no L2, LR ×1 | 9/9; 29/29 | 8/8; 91.75% HQ | 7.21 | 8/8; 100% HQ | 3/5 | 75.22% |
| R1+R2 0.1, L2 0.02 | 9/9; 29/29 | 7/8; 100% HQ | 6.37 | 5/8; 65.16% HQ | 0/5 | 16.60% |

The coefficient-3 candidate's live observations are:

| Step | HQ-qualified modes | HQ samples |
| ---: | ---: | ---: |
| 1,000 | 8/8 | 90.99% |
| 1,050 | 8/8 | 83.91% |
| 1,100 | 8/8 | 90.99% |
| 1,150 | 8/8 | 92.33% |
| 1,200 | 8/8 | 100.00% |

Every mode still has at least one HQ particle at step 1,050; some additional
particles are outside the quality radius, lowering the overall sample quality.
Final allocation is `[1, 2, 1, 1, 2, 1, 2, 2]`, with all twelve outputs HQ.
This is the most balanced eight-mode allocation possible for twelve equal
particles. Its exact support effective modes is 7.56 and sampled estimate 7.54.
It improves both final coverage and observed late quality over R1+R2; the
remaining HQ fluctuation prevents claiming it has fully settled.

The coefficient-2 ×0.85 candidate finishes with exact particle allocation
`[2, 1, 1, 2, 1, 2, 2, 1]`: all twelve outputs are HQ and the eight-mode
allocation is as even as twelve equally likely particles allow. Exact support
effective modes is about 7.56; the 4,096-sample estimate is 7.51. Its trajectory
MSE is 0.002112 against the 0.02 limit, and all other required bounds pass.

Among the coefficient-2 candidates, ×1 has the stronger worst-checkpoint quality. Its nearest-mode
assignment for each particle is identical at all five late observations, covering
all eight modes throughout. Its HQ dips therefore reflect particles moving
around their discovered clusters, rather than losing a nearest-mode allocation.
At step 1,150 three outputs are 0.232, 0.366 and 0.304 from their centers, outside
the 0.21 HQ radius. At step 1,100 every output is HQ. Two final covered modes
are only narrowly inside that radius. This points toward late update size as a
useful next experiment; it does not prove a single cause.

Reducing the original cap's learning rate to one-half or one-quarter failed
short-budget motion/identity tests. Merely increasing its coefficient to 2 or 5
also failed the ring. The successful combination changes both cap target and
strength. The tested alternative VICReg weights and hinge/least-squares RP
objectives did not improve coverage. These are joint configuration results,
not evidence that one knob independently explains the win.

## Why one mode is missing

The small ring host uses 12 equally likely learned latent particles and a
deterministic generator. Enumerating all 12 outputs gives this final allocation
for R1+R2 coefficient 0.1, with particle L2 0.02:

| Mode index | Angle | HQ particles |
| ---: | ---: | ---: |
| 0 | 0° | 1 |
| 1 | 45° | **0** |
| 2 | 90° | 3 |
| 3 | 135° | 1 |
| 4 | 180° | 1 |
| 5 | 225° | 2 |
| 6 | 270° | 2 |
| 7 | 315° | 2 |

The missing cluster is centered near `(2.1213, 2.1213)`. Its nearest generated
point is **2.2398** units away; HQ requires distance at most **0.21**. This is an
absence from the model's entire discrete support, not random evaluation missing
a rare sample. At step 800, particle index 11 covered mode 1. By step 1,000 no
particle was nearest that mode; at the end, particle 11 maps to mode 2.

All 12 final outputs are close to *some* real cluster, so HQ is 100%. HQ alone
does not penalize an entirely missing cluster. Sampled effective modes is 6.37,
reflecting the imbalance among the represented modes. Exact support effective
modes is 6.45. These differ slightly because the former estimates probabilities
from 4,096 random draws from the 12 particles.

The measurements establish loss of particle allocation and substantial training
oscillation. The toy has no explicit ring-coverage training term; its GAN loss
and latent regularizers have not maintained that allocation. They do not
establish a single causal culprit among the penalty, particle L2, optimizer and
constant learning rate.

## Final-step PASS hid unstable checkpoints

| Step | R1+R2 0.1 live modes | Live HQ | R1+R2 0.1, no L2 live modes | Live HQ |
| ---: | ---: | ---: | ---: | ---: |
| 800 | 8/8 | 100.00% | 3/8 | 33.64% |
| 1,000 | 2/8 | 16.60% | 7/8 | 100.00% |
| 1,050 | 7/8 | 82.45% | 5/8 | 42.94% |
| 1,100 | 3/8 | 23.88% | 8/8 | 100.00% |
| 1,150 | 7/8 | 91.14% | 8/8 | 100.00% |
| 1,200 | 7/8 | 100.00% | 8/8 | 100.00% |

Mode counts here require at least one HQ output per cluster. Some interim
count losses are outputs wandering outside the quality radius; the final
missing 45° mode is also absent under nearest-cluster assignment.

Removing particle L2 from the R1+R2 candidate improves the final ring to **8/8
at 100% HQ**. Its support allocation is `[2, 1, 2, 1, 1, 2, 1, 2]`, the most
even possible eight-mode allocation for 12 equally weighted particles. But
this combined candidate fails shared trajectory: **MSE 0.251673 > 0.02**. It
passes only 8/9 toys and 28/29 bounds. The other seven non-ring toys pass.
Removing L2 is therefore not an unconditional improvement across these hosts.

EMA also reflects the unsettled small-host training: R1+R2's EMA ring is 5/8
at 65.16% HQ. Its averaged model puts a point nearest the missing cluster, but
that point is still 1.1407 units from its center. The measurements are consistent
with averaging weights while particle outputs change modes; they do not isolate
EMA lag from nonlinear parameter averaging as the cause.

## Matched comparison using the stock recipe

To assess the connection to ParticleGAN defaults, run both penalties with the
same stock recipe: 20,000 particles, no particle L2, VICReg 1, 7,000 steps,
batch 256, learning rate 0.0006, prior LR multiplier 10, discriminator multiplier
1.5, Adam betas `(0, 0.999)`, delayed cosine decay, EMA 0.995. Only the penalty
changes. Both runs use seed 0 and the same 8-mode ring host (96-wide MLPs and a
Fourier-3 critic). This is **not** the separate 100-Gaussian benchmark.

| Penalty in stock recipe | Live modes | Live HQ | Live effective modes | EMA modes / HQ | Worst tail live HQ | Tail checks with 8/8 and HQ ≥90% |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Current `b_cap`, coeff 1 | 8/8 | 99.05% | 7.95/8 | 8/8 / 99.66% | 98.51% | 5/5 |
| R1+R2, coeff 0.1 | 8/8 | 99.32% | 7.94/8 | 8/8 / 99.44% | 99.24% | 5/5 |

Tail observations are at steps 6,800, 6,850, 6,900, 6,950 and 7,000. Both keep
all eight modes at each of these observations. R1+R2 has slightly higher live HQ
here; the current cap has slightly higher EMA HQ and sampled mode balance.
These results do not establish a clear default winner. They also provide no
reason to disable EMA globally: both stock-recipe EMA models pass.

The stock rows have more capacity, more steps and different optimization than
the twelve-particle toy rows. They are a matched comparison **with each other**;
do not count them as same-budget wins over the small-host suite.

## Decision and next comparisons

Keep the current ParticleGAN production default while using the new full-coverage
`b_cap` configurations as the leading small-host baselines. R1+R2 0.1 remains
an all-regression-pass reference; its no-L2 combination demonstrates the
coverage/identity tradeoff. For default
selection, target full coverage, high HQ, balanced mode mass and sustained late
quality across the behavioral hosts, then compare candidates on the actual
100-Gaussian benchmark under matched recipe settings and compute. The flat-LR
small host's oscillation makes learning-rate/schedule controls a useful next
experiment. Do not choose a favorable intermediate checkpoint or change
thresholds to make an arm win.

Raw evidence: [completed candidates and checkpoint/support diagnostics](results.json),
[all searched configurations](search/README.md),
[matched stock-recipe runs](stock_ring.json). The diagnostics use a separate
evaluation RNG and preserve the original final metrics exactly for all three
previous candidates. Tests verify that enabling diagnostics does not alter the
training results. No seed sweep was performed.

Reproduce the small-host comparison with the baseline command in [README.md](README.md).
The stock comparison is `python -m benchmarks.locked_shared.default_selection`;
it writes `reports/behavioral_baseline/stock_ring.json` and refuses to overwrite
an existing result. Move that result aside before rerunning. Log every measured
checkpoint with `> /tmp/stock-ring.log 2>&1`, then `tail -f /tmp/stock-ring.log`.
