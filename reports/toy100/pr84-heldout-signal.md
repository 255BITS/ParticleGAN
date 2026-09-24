# Fixed-state generator signal at PR84 hold failures

The unchanged PR84 generator receives a **coherent outward field** at the
first dense quality failure. The abrupt update at 1325 is not explained by a
rare G minibatch or a sudden Adam-denominator collapse. This is an attribution
diagnostic on saved states, not a training candidate or a new production gate.

The [exact training replay](continuous-evidence/pr84-stationary-failure-diagnosis/)
captured the post-accepted-D state before each G minibatch and preserved the
original full-state/diagnostic/optimizer/RNG parity. From each such state, this
audit reconstructs D, G, prior and Adam, replays the actual next G batch, then
draws **16 further G minibatches** from the same streams. G/prior weights and
Adam moments return to the captured values before every hypothetical batch;
the critic and original target distribution remain fixed. Each batch computes
the original five-point G loss and its ordinary Adam proposal, replays that
batch for the own-curvature ratio, and applies the original bound. No proposed
state is kept for a subsequent training step. The actual ordinary G proposal
matches the captured weights and moments bit-for-bit at all five states; its
bounded weights differ by at most `2.98e-8` from captured float32 weights.

| Update | Actual HQ before → after G | G factor | Actual clean output RMS, network / prior / joint | Held-out raw network-gradient coherence | Held-out clean-output coherence |
| --- | --- | ---: | --- | ---: | ---: |
| 1324, passing | .99902 → .99438 | 1.000 | .02314 / .00328 / .02540 | .888 | .812 |
| **1325, abrupt failure** | **.99658 → .82422** | **1.000** | **.05344 / .00450 / .05713** | **.943** | **.959** |
| 1389, later dip | .92407 → .89258 | .618 | .01131 / .00241 / .01287 | .740 | .640 |
| 1539, severe episode | .79443 → .79517 | .410 | .02571 / .00494 / .02937 | .863 | .879 |
| 1540, severe episode | .79932 → .78857 | .519 | .03089 / .00592 / .03534 | .828 | .787 |

Coherence is the bias-corrected squared mean divided by mean squared norm
across 16 held-out draws. The raw G-network gradient has coherence `.943` at
1325; the **pre-update Adam-metric-whitened** gradient, `sqrt(P_old) g`, has
coherence `.945`. Those are distinct from the actual bounded output steps in
the last column, which include Adam's just-updated denominator and the
own-curvature interpolation. The output step is dominated by the generator
network; the prior-only change at 1325 is an order of magnitude smaller.

For a diagnostic target direction, the audit differentiates mean squared
clean-particle distance to each particle's *nearest frozen ring center* at
the base state. A positive directional cosine means a proposed descent
direction increases that distance locally. At 1325, the negative **raw**
network game gradient has positive cosine in **16/16** held-out batches
(mean `.680`), and the frozen Adam-metric direction also has positive cosine
in **16/16** (mean `.195`). The finite bounded outputs send each of the four
particles nearest the HQ boundary outward in **16/16** batches. The four
actual radial displacements are `.1013`, `.0474`, `.0478`, and `.1047`. The
frozen critic's input sharpness rises from `.1269` at 1324 to `.1711` at
1325, while its advantage is `.0249` then `.0298`; D's curvature bound stays
inactive. G's median bias-corrected Adam denominator is `.0006677` versus
`.0006673` over the two states. There is no one-step metric collapse.

These facts distinguish the competing explanations for this event. The raw
game gradient itself is consistently outward under this *offline* quality
direction; Adam does not flip an inward mean field outward. Its metric and the
network Jacobian still turn that field into consequential output motion. A
single observed minibatch is not the culprit. The result is conditional on
the saved post-D critic and 16 G batches; it does not establish a population
GAN equilibrium, separate stochastic D updates, or prove that nearest-center
distance is the right training objective. The original ring has twelve
equally sampled supports for eight Gaussian modes, so a GAN objective can
prefer output motion that violates the strict HQ metric even with a coherent
gradient. The prior fixed-epsilon, confidence and slope controllers did not
resolve both cold acquisition and hold, and these measurements do not justify
another damping threshold or noise-averaging candidate.

The [portable receipt](continuous-evidence/pr84-heldout-signal/heldout-v2.json.gz)
contains aggregate raw/scaled-gradient coherence, each hypothetical batch's
bounded clean-output motion and center-radial projection, exact replay
errors, and source/capture hashes. It was reproduced from the committed compact
states and archived stage observations. A second
[full-state parity receipt](continuous-evidence/pr84-heldout-signal/heldout-v2-full-parity.json.gz)
used the original complete local sidecar and gives numerically identical
diagnostic rows, with the additional exact optimizer/parameter comparison.
The [source](pr84_heldout_signal.py) implements the read-only diagnostic; the
compact frozen states are archived with the
[capture report](continuous-evidence/pr84-stationary-failure-diagnosis/).
The older 2400-step diagnostic checked every ten updates after 1200, so it
first *sampled* a failure at 1390. The exact dense capture sees a transient
failure earlier at 1325 and has the same final state as that older run.

Run on the pinned CPU environment with one thread and AVX2:

```bash
python -u reports/toy100/pr84_heldout_signal.py \
  --capture reports/toy100/continuous-evidence/pr84-stationary-failure-diagnosis \
  --output NEW_HELDOUT.json.gz
```
