# Local exchangeability thermostat: rejected before training

The fixed cheap screen rejects `bispf-local-exchangeability-v1`. It correctly
keeps missing rare/ring support hot and removes the same-law sampling floor,
but cannot distinguish the selected acceptable output laws from severe failures
well enough: AUROC **.75**, below the preregistered **.8** minimum. Its three
non-reference passing states all remain fully hot. **No public API mechanism
was implemented and no scientific training or GPU spend occurred.**

The [compact result/provenance](result.json) and [exact probe source](cheap_controls.py)
retain this negative result. Original task gates and all earlier evidence keep
their own source, prior, initialization, budget and sampling identities. The
[completed three-theme study](../../three-theme-repairs-20261022/README.md)
remains the motivating measured comparison. This report adds no leaderboard,
qualification, default selection or paid successor.

## One mechanism, fixed before computation

The prior raw-nearest-neighbor observer stays hot under independent samples
from the same law. A prior energy-U replacement removes the mean sampling bias
but its global discrepancy can hide missing 2% mass, and its unchanged peak EMA
reheats passing states. A local label scan was proposed to distinguish coherent
support imbalance from finite-sample fluctuations without target component
labels, widths, centres or evaluator metrics in the trainer.

Let real and fake panels each contain `M=2048` examples. Pool them as
`Z=(R,F)`, with `N=4096`. Membership in R/F is the sample label; it is not an
oracle mixture-component label. Fix `alpha=.05`,
`Lmax=ceil(log2(N))+1`, and
`kmin=ceil(log2(2*N*Lmax/alpha**2))`. Use the complete predetermined hierarchy
`kmin, 2*kmin, 4*kmin, ...`, capped at `N/2`, with `N/2` included. At N=4096
the ranks are **26, 52, 104, 208, 416, 832, 1664, 2048**. The hierarchy was
fixed before numerical probing; it is one multiscale test, not selected
bandwidths or a configuration sweep.

For every point i and rank k, take its closed kth-neighbor ball, excluding
itself and including every distance tie at the boundary. Write `s_ik` for its
size and `C_ik` for the number of real neighbours. Under frozen-state equal-law
exchangeability, conditional on the pooled geometry and centre membership,

```text
C_ik ~ Hypergeom(N-1, M - 1{i is real}, s_ik)
p_ik = min(1, 2 min(P(C <= C_ik), P(C >= C_ik)))
p_global = min(1, N * number_of_scales * min_ik p_ik)
m = clip(log(1/p_global) / log(1/alpha), 0, 1), with p_global=0 => m=1
```

Thus every rejected local discrepancy, `p_global <= alpha`, receives **full
mobility**. The same-law conditional-null probability of full mobility is at
most alpha by a union bound, without independent neighborhoods. This is a
finite-sample frozen-state statement, not an anytime theorem for adaptive GAN
training. The probe computes float64 hypergeometric tails, not randomized
permutations. Including all distance ties avoids artificial sample-order
separation: matched constants yield p=1, while separated constants remain hot.

The conditional hypergeometric nearest-neighbor label-count law appears in
[Schilling's original paper, hosted by the author](https://www.csun.edu/~hcmth031/mtstbonn.pdf).
The local maximum, explicit multiscale correction, closed-tie treatment and
mobility mapping above are this proposal's construction; they are not claims
that Schilling published this controller.

The intended public recipe delta was
`bispf_thermal_observer=local_exchangeability` together with the existing
`bispf_thermal_application=displacement`. It would leave critic updates and
their legacy observer unchanged, unscale the G/SBNP gradient hooks, and apply
m to the final normalized/direction-blended G/prior displacement. Learned prior
locations, fixed task kernel widths/weights/capacity, SBNP coefficient .01,
losses and nominal rates would remain as declared. The G/prior cooling policy
and applied update would explicitly change.

The bounded-state plan used one disjoint block of 2048 already-consumed real
examples, plus its existing latent indices and kernel perturbations. Current
G/prior would regenerate the fake panel once per full block without new sampler
draws; the mobility would be held only until the next block. Warmup would remain
at mobility 1. No peak EMA, future-horizon schedule or new temperature threshold
would persist shocks. These buffers, cursor and earned mobility would require
exact checkpoints and conservative refusal of unsupported replay/model hosts.
Adaptive reuse of trained-on examples and latents is a calibration limitation;
the frozen-state null bound alone cannot establish its validity in training.

This is a distinct signal from the failed paired-critic-gap idle rule (#398),
relative tether force cap (#429), global harmonic capture (#435), raw NN
cooling and energy-U observer. Its support guard acts on local sample-label
imbalances rather than a global average whose signal vanishes quadratically
with rare mass.

## Physics interpretation and its limit

The proposed analogy separates equilibrium sampling fluctuations from coherent
density imbalance before choosing a mobility. Actual noisy-gradient thermostats
jointly design dynamics and noise feedback; see
[Leimkuhler and Shang](https://arxiv.org/abs/1505.06889) and
[Leimkuhler, Sachs and Stoltz](https://arxiv.org/abs/1908.09363).
Those papers do not prove a property of this GAN controller. BISPF's game field
is not generally conservative, and normalized optimizer noise is not a declared
Langevin heat bath. No Boltzmann invariant law, fluctuation-dissipation identity,
physical temperature estimate or convergence theorem is claimed here.

## Fixed controls and result

The probe reads the exact previous campaign's control sample artifacts, checks
their SHA256 identities and adds no samples, targets, training or model forwards.
Full raw probe output and the tail-friendly log remain outside Git:

```text
/mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/settling/cheap-controls.json
/mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/settling/cheap-controls.log
```

It uses 20 full 2048-versus-2048 same-law panels, 320 disjoint
128-versus-128 same-law panels, 24 predetermined acquisition/shift/overlap states,
and nine analytical fixtures. Same-law panels split samples from one frozen
generated state. The 24 cross-state comparisons use the first archived passing
generated state as a fixed approximate reference: Gaussian 209, shifted
Gaussian 4709 and overlap 100. Actual real training batches were unavailable.
This is a saved-data diagnostic with approximate references, not a target oracle
or reproduction of historical controller inputs.

The predeclared admission rule required every missing-support/collapse/shifted
constant fixture to have m=1, matched constant/four-atom fixtures to have m=0,
at most 1/20 and 16/320 same-law full-heat cases, PASS-versus-severe AUROC >=.8,
median PASS mobility <=.5, and median severe mobility >=.95. Severe means at
least twice an original task gate bound, fixed independently of the new signal.
No alpha, memory, rank, threshold or seed was tuned after observing the result.

| Fixed control | Observed result |
| --- | --- |
| 20 full same-law panels | 0 full heat; 19 exactly cold; maximum m=.074831 |
| 320 small same-law panels | 3 full heat (.9375%); 284 exactly cold |
| Matched constant and matched four-atom laws | Both exactly cold |
| Shifted constant and two collapsed laws | All full heat |
| Missing one of four/eight/sixteen modes | All full heat |
| Missing nominal 2% population | Full heat; p_global=4.8693e-8 |
| Six selected receipt-PASS states | Three cold, three full heat; median m=.5 |
| Six selected severe states | All full heat; median m=1 |
| PASS-versus-severe AUROC | **.75: admission FAIL** |

The nominal 2% fixture realizes 41 of 2048 real examples in the rare corner
and zero fake examples there. These are explicit zero-width point-mass fixtures,
separate from the original positive-width Gaussian task laws. They validate
sample-label support sensitivity without silently substituting a fixture into
trained task evidence.

The three cold passing states are the approximate reference states themselves.
Every non-reference passing state remains fully hot: Gaussian smoke 500
(`p=1.0349e-11`), overlap 750 (`p=1.7374e-51`) and overlap 1050
(`p=4.6009e-28`). The test detects differences between two acceptable but
nonidentical generated laws. It does not establish practical equivalence to
the actual target. Approximate references limit the inference: this result
does not prove what the observer would do against genuine fresh real panels.
It does fail the exact cheap admission rule chosen for this proposal.

CPU cost is **32.639267 process seconds**, within a hard 120-CPU-second limit.
GPU spend, optimizer updates and new sampling draws are all **zero**.

## Decision and unexecuted implementation plan

**Reject before API implementation and scientific training.** Keep the useful
finite-sample and missing-support results, together with the failed practical
equivalence check. Do not rescue this revision by selecting another reference,
changing alpha/window/ranks, altering admission thresholds or running seeds.

Had the screen passed, the bounded plan was: implement one checkpointed public
observer/replay state; validate exact resume, unsupported-host rejection,
zero-mobility bit preservation and unchanged streams; then submit one frozen
candidate against a fresh control on a common source with unchanged task gates
and full reservations. That plan stops at this failed screen. There is no
controller field in the public API from this branch.

A future substantive question would need to separate distributional equality
from achievable approximation while retaining rare-support resolution. The
original finite-sample floor and support loss controls remain mandatory; a
thermodynamic name does not resolve the approximation problem.

Reproduce the fixed diagnostic without training:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u \
  reports/forge/first-principles-round2-20261022/settling/cheap_controls.py \
  --output /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/settling/cheap-controls-reproduction.json \
  > /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/settling/cheap-controls-reproduction.log 2>&1
tail -F /mnt/ml7tb/ParticleGAN-forge/first-principles-round2-20261022/settling/cheap-controls-reproduction.log
```
