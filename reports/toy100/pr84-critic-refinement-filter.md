# Bounded critic refinement passes the strict saved-state filter

**All 44 checks pass across all three predeclared failure windows.** This is a
local diagnostic result, eligible for the next warm-state test. It is not a
long-hold, cold-acquisition, convergence or shared-production-gate result.
The implementation currently rejects active input noise and the conditional
trajectory host rather than claiming cold-start support.

The previous [fixed-generator diagnosis](pr84-critic-relaxation-diagnosis.md)
found that a bounded fit of the existing penalized critic loss improved its
held-out objective and changed generator guidance inward. The fits were not
stationary. This candidate materializes that same fitting procedure inside
the actual alternating update, without changing its objective or tuning a
gain. It tests critic optimization directly after explicit cross-response and
opponent prediction failed the hold requirement.

| Resumed updates | Original passing checks | Refinement passing checks | Original minimum HQ | Refinement minimum HQ |
| --- | ---: | ---: | ---: | ---: |
| 1324–1335 | 9/12 | 12/12 | .775391 | 1.000000 |
| 1380–1395 | 14/16 | 16/16 | .878662 | 1.000000 |
| 1530–1545 | 1/16 | 16/16 | .788574 | .999512 |

Every candidate check has all eight modes and HQ at least .90, using the
unchanged live 4096-draw evaluator. All three original controls match the
archived supports, update records and first complete accepted-state hash.
Every live G update is scored; no endpoint-only selection or warm-transplant
exception is used. The severe branch begins at the still-passing pre-1530
state, before the original branch loses a mode; this does not demonstrate
recovery from the already-collapsed 1539 state.

## Exact update and scope

[`pr84_critic_refinement.py`](pr84_critic_refinement.py) first performs the
original sharp-D Adam proposal and its existing curvature bound. At accepted
`D*`, with G and prior frozen, it runs **one** L-BFGS attempt on 1024 cached
real/fake pairs: eight 128-pair draws from cloned pre-D data and output-noise
streams. The original penalized D objective is used, including `b_cap`.
The first bank minibatch reproduces the actual phase-zero D gradient bitwise
on every update. No bank draw advances the original training streams.

The fit is the unchanged numerical helper from the prior diagnostic: learning
rate 1, strong-Wolfe line search, history 10, at most 40 iterations and a hard
limit of 80 closure evaluations, with no restart. It selects the lowest finite
**training-bank D loss** among evaluated points. There is no quality, target
center, held-out metric or training-age acceptance rule.

The selected critic `Dhat` becomes the actual critic. G's base and proposal
field evaluations both use it, with the phase-one PR84 stencil width frozen
across the pair. The usual G/prior Adam proposal and own-curvature bound then
materialize. The original D bound applies to the initial Adam proposal only;
the additional explicit refinement is outside that bound. `Dhat` persists
into the next outer step. Current warm-state widths remained exactly .15.

Both Adam moment states advance exactly once per outer update; their ordinary
rates remain G/D .00425 and prior .0085. D Adam moments describe the original
proposal before the extra fit, and are deliberately retained after `Dhat`
moves. The receipt names this mismatch. Noise horizon remains 1200, seed 0,
and the candidate/original training RNG endpoints and full noise receipts
match bitwise. The continuation restores the entire captured model, optimizer,
EMA and RNG state, not only weights.

## Cost and limits

The 44 candidate updates used **2515 fit closures**, 37–80 per update, on
1024-pair banks. Seven fits reached the hard closure budget. Fit-only wall time
was 27.73 seconds, about .63 seconds/update on one CPU thread. There were 44
additional 128-pair gradient checks for exact first-bank parity and 132 normal
128-pair D fields, for **2691 total D gradient calls**; G used 132 normal fields.
Gradient-call counts are not equal-cost operation counts: each fit closure has
eight times the original batch size and computes the gradient penalty.
Bank construction additionally uses eight clean G forwards per outer update;
stencil-forward overhead is also outside the gradient-call totals.

Selected training losses were .61849–.65760; each was below its own initial
bank loss. D refinement parameter norms were .08814–1.11525. At the selected
points, raw maximum gradient components remained .00198–.01613. **No fit is
claimed to be a stationary point or best response.** The meaningful evidence
is the actual quality of the resulting local coupled trajectories, with exact
controls. This does not prove indefinitely stable training or acceptable
compute cost. A longer hold and actual cold acquisition remain required.

The implementation's first active scope is the frozen CPU `mode_hold` host
with discriminator input sigma zero and fixed output scale. Disabled
refinement has exact full-host parity on both existing diagnostic hosts.
Eight focused tests pass, independently rerun: a nonzero quadratic optimum
and exact-zero rest, hard closure cap/best-point restoration, disabled full
state parity, real global/isolated-noise active runs with RNG/moment and
fixed-opponent checks, unsupported-scope rejection, and failure restoration.

## Reproduction

The [evidence manifest](continuous-evidence/pr84-critic-refinement-filter/manifest.json)
archives all six branch runs, source-bound declaration, full summary, log,
and exact adapter, helper, host and generated-source bytes. Adapter SHA256:
`cfdf3d050da538e92e79d5a12d18a715c3a8ca7e59a08515014e8e389c598a11`.
Unchanged fitting-helper SHA256:
`9f5f0d630e818d5259081c333f55e667f8a619ae81b25ebdd1bb62d334d7639b`.

The required input is the **full capture-v2 sidecar**, SHA256
`37aa612bd3b3e1867a2d1508674158849ccf5940c907af76b1f8b061f4eb3e47`.
The compact public subset omits 1380; regenerate the full sidecar with the
frozen stationary-failure capture script before using this filter. Local
noise counters report this resumed segment, not reconstructed prefix history.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' \
  /tmp/pr38-default-env/bin/python -u reports/toy100/pr84_critic_refinement_filter.py \
  --capture /path/to/stationary-failure-replay/capture-v2 \
  --output /tmp/new-critic-refinement-filter
```
