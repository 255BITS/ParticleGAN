# BISPF quartic fidelity repair: completed diagnostic

**Quartic transport repairs sustained overlap, but does not solve local fidelity.**
The candidate completes **5 PASS / 10 FAIL**, versus the matched control's
**4 PASS / 11 FAIL**, retaining all four controls. Width allocation improves
throughout training while sparse tails remain. Intensity, blobs and grid
quality worsen. Stop this exact global fidelity revision; retain the overlap
result as a source-bound experimental positive.

The [common comparison](../README.md) and [complete numerical readout](../numerical-results.md)
own cross-arm results. The [technique inventory](../../technique-inventory.md)
remains the single goal leaderboard. No ordinary qualification, public default,
original gate, prior/sampling contract or historical result changes.

## Mechanism and frozen evidence

The [schema3 candidate](../../../../configs/forge/ideas/bispf-fidelity-quartic-v1.json)
inherits `bispf-optimized-quintic-v1` and sets only `fidelity_transport_weight=1`
globally. It adds the same-batch sorted projected fourth-power output residual,
divided by squared detached real coordinate variance, once to the G/prior loss.
Dimension 1 uses identity; dimension 2 uses 32 fixed angles; higher dimensions use
up to 32 coordinate axes plus 32 unit trigonometric directions. Exactly constant
real batches use scale 1. No labels, target-component geometry, extra training
draws/forwards, sampler, state or prior-regularizer change enters. The prior
remains learned with SBNP .01.

The direct auxiliary field bypasses the existing critic-logit thermal hook;
adversarial/SBNP cooling and protected-loss geometry keep their original laws.
It is an added uncooled output field, not uniform cooling of the augmented
objective. Quartic cost is unbounded in projected distance, but its finite
projections and global normalization guarantee neither local component moments
nor useful finite parameter motion. The frozen [study](../../../../configs/forge/studies/bispf-fidelity-quartic-v1-three-theme-study-v1.json)
retains the prediction/falsifier and the earlier
[bounded-kernel](../../bcap-physics/round3/README.md),
[allocation/shape](../../bcap-physics/round5/README.md) and
[core/full](../../bcap-three-phase/README.md) negatives.

Execution source: `2a2b662bdfa6249076b27dbea4c136e23d3e7a80`; scientific digest:
`8bfcc01cd7482b1b348edb2162567adfdcdaa7d308f75fc834774302c6ac4acb`.
Control request `728323efa37b4c2345764a9d` and fidelity request
`a0749bea42cc192e60afed81` each complete all 15 unchanged tasks through public
`GANTrainer`. Protocol seed 0, public deterministic initialization, architecture,
target/data sequence, initial prior/capacity, sampling, update allowance and
cadence remain fixed. Named consumed streams are checkpointed.

The [compact diagnostic receipt](analysis.json) binds every task to its result
and request hashes, selected final metrics and complete-curve summaries. The
common publisher independently owns initialization, stream, task-law and cost
proofs. These new matched tensors retain their own evidence identity even when
control endpoints numerically reproduce an older source.

## Sustained overlap repair and preserved controls

Overlap improves from 10/24 complete passing observations, maximum run 3 and
terminal suffix 0 to **22/24**, maximum run 20 and **terminal suffix20**. Updates
250–1200 all pass. Endpoint normalized sliced-W1 is .0537247, mean error .0720676
and global covariance error .119367, against .18/.15/.45 bounds.

All four controls remain PASS: Gaussian acquisition smoke, ring16 acquisition,
mode hold and stripes. Their actual contracts differ. Quartic smoke confirms
acquisition at 125/292/959 despite final KS .208409; acquisition is not settling.
Its own continuation fails: stationary 8/72 versus control 6/72, shifted hold 2/24
for both, no deadline reacquisition. Ring16 suffix improves 12→59; mode hold
remains PASS with suffix 10→8; stripes suffix improves 12→17.

## Width allocation improves; tails prevent sustained density

All24 quartic checks pass mass-TV, sliced-W1 and minimum-eigenvalue bounds.
Control passes none of its mass-TV or sliced-W1 checks. Quartic nevertheless
passes only4/24 full covariance checks and **2/24 complete checks**, never
consecutively: maximum complete run 1, suffix 0, with five required.

| Width endpoint metric | Control | Quartic | Original bound |
| --- | ---: | ---: | ---: |
| Mass TV | .277588 | .0117188 | ≤ .15 |
| Full covariance error | 9.68690 | 2.13772 | ≤ .85 |
| Four-sigma core error, diagnostic | .515564 | .358197 | not the full gate |
| Full minimum eigen ratio | .0934703 | .668359 | ≥ .15 |
| HQ | .981934 | .923828 | ≥ .85 |

The dominant tail moves to the narrowest component. Control's .12-width
component has 28/256 assigned samples beyond 4σ, contributing 88.09% of its full
centered scatter. Quartic's .07-width component has 54/1012 beyond 4σ,
contributing 85.31%; full error 6.43640 coexists with core error .284172.

A separate exhaustive endpoint **G(learned location)** census gives control
counts [96,16,103,41] and quartic [65,64,62,65], with 4σ center-tail counts
[0,2,0,0] and [3,3,1,0]. Tails already exist before kernel perturbation; jitter
alone cannot explain them. This supplementary CPU forward uses exact saved
weights, no optimizer update or additional output-sampling draw. Temporary
constructor RNG is isolated/restored before loading. It is separate from the
original served-MoG gate, which retains jitter and its own assignments. It does
not replace that gate or uniquely attribute the training history.

Unequal mass and anisotropic show the same allocation/shape split. Mass TVs
improve .057422→.026436 and .175212→.024333, but HQ falls .977295→.651367 and
.995605→.829590. Full covariance worsens 1.76764→2.26960 and .313814→1.09301.
Both terminal suffixes remain 0.

## Image fidelity worsens

Intensity keeps exact nearest assignments [16,16], while clean counts fall
**[11,8]→[5,4]**, or19/32→9/32. Quartic never reaches two qualifying levels or
overall HQ in any of 24 checks. Its best HQ is final .28125, below .9.

An exact SSE decomposition separates uniform patch offset, spatial variation
within the 4×4 patch and background. Control shares are 35.59%/64.40%/.015%;
quartic shares 44.11%/55.44%/.452%. Only 5/13 control misses are dominated by
patch-mean offset, falsifying the older investigation's proposed ≥10/13
hypothesis in this new matching cohort. A shared spatial pattern accounts for
73.1%/87.7% of dark/bright control spatial SSE. The problem is not solely uniform
brightness, and quartic fails to correct it.

Blobs has **HQ0 in all 24 observations**, versus control endpoint .8125. Outputs
approach background saturation: maximum pixel .41850 at 25, .00604 at 100,
7.76e-6 at 200; final mean 5.30e-9 and maximum 2.65e-6. Finite tensors exclude a
missing-media/undefined-score explanation. They identify the trajectory, not
the responsible historical gradient. Bars rises 2→3 qualifying modes but loses
HQ .96875→.8125 and still fails its unchanged four-mode gate.

## Native: no full or sustained repair

Independent 100k clean/live holdout precision changes grid **.30497→.22332**,
rotated **.22301→.22796**, staggered **.33617→.41797**, all below .97. Every
candidate has **0/34 passing scheduled coverage checks and 0/5 passing terminal
accuracy checks**; all three independent holdouts also fail. No best-checkpoint
selection is used. Grid holdout shape remains undefined because genuine cell
counts are insufficient, which is a failure rather than zero error.

Staggered centering improves 1.14288→.909601σ, but covariance-trace bias worsens
.541690→.690225 and radial KS remains .369953 versus .04. Its final five 20k
precisions range .32405–.42215. Occupancy, mass TV or one precision endpoint
cannot substitute for density. Native sampling and finite-cloud image
enumeration remain distinct cohorts.

## Decision, reproduction and media

Reject this exact revision as the proposed local-fidelity solution. Preserve
the complete overlap positive and all negatives. No paid successor, coefficient
or seed sweep, continuation or gate relaxation follows. A new mechanism would
need measured output-force/parameter-motion evidence explaining center tails
or image saturation; another global scalar or lower surrogate is insufficient.

```sh
python reports/forge/three-theme-repairs-20261022/fidelity/analyze_saved.py \
  --queue /mnt/ml7tb/ParticleGAN-forge/three-theme-repairs-20261022/queue \
  --output /tmp/bispf-fidelity-analysis.json
```

The reducer verifies frozen source/complete 15-task requests, records result
hash bindings, and matches recomputed complete-curve counts to the independent
grader. It emits scalar ranges/counts and fixed endpoint decompositions;
bulk streams/tensors stay archived. **27 mechanism/public-consumer checks pass**
on the common source, including scalar/image single addition, consumed-stream
parity and complete exact checkpoint continuation.

Actual-training illustrations from the same saved observations:
[overlap](../media/bispf-fidelity-quartic-v1/vector_overlap.gif),
[width](../media/bispf-fidelity-quartic-v1/vector_unequal_width.gif),
[intensity](../media/bispf-fidelity-quartic-v1/img_intensity2.gif),
[blobs](../media/bispf-fidelity-quartic-v1/img_blobs4.gif),
[grid](../media/bispf-fidelity-quartic-v1/grid100.gif).
Numerical gates determine conclusions. Complete media provenance and unique
paid-cost accounting stay in the common publication; shared-GPU timings do not
supply an optimizer-speed ranking.
