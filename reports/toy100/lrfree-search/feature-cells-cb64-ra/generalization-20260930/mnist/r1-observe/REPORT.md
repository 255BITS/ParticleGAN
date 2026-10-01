# Frozen RA12 optimizer-surprise observation

Completed exact Toy750→835 (85 updates) and MNIST100→215 (115 updates), original seeds/data cursors/private streams/serial CUDA execution. No scores, samples, source changes or hyperparameter changes. Native restoration was semantically bit-identical; all CPU/CUDA/private/birth RNG states were checked unchanged at every observation. The original decide function ran exactly once per update and its class wrapper was restored without adding any detector instance field.

| Case | Fire step (begin update) | G ratio | Table ratio | Noise ratio | D ratio | Scales before fire G/table/noise/D |
|---|---:|---:|---:|---:|---:|---|
| Toy25 | 820 (821) | 2.586348 | 2.253064 | zero, omitted | 1.580801 | .125 / 1 / 1 / .015625 |
| MNIST | 202 (203) | 17.301340 | 6.937788 | .504148 | 4.164878 | 1 / 1 / 1 / 1 |

## Toy25

KA2 starts its blended anchor at update800/call800. The aggregate is1.05219 before that update and1.05815 at begin801. The new gradients lift the aggregate past1.25 at begin803, past2 at begin810, and sustain12 above-threshold decisions through the fire at begin821. KA2's own ratio is stillNone at that fire; it first becomes1 at begin825 after24 blended calls. The R1 anchor release latch therefore does not activate. Noise is clamped at.029 and omitted for the entire85-update window; no changing group membership caused this fire. Population moves continue every8 updates, including800/808/816, so this trace supports the phase-change hypothesis but does not independently prove it caused the gradient rise.

## MNIST

The aggregate passes1.25 at begin181 and2 at begin192, sustaining12 above-threshold decisions through begin203. The generator and table ratios already exceed2 before the critic ratio rises. All four groups stay active throughout the window, and the learned sigma is.06100 above its.029 floor at fire. No birth/isolating moves occur. KA2 remains pure A with no anchor or ratio. All tester scales remain1; the fire cannot restore a reduced LR, yet it shrinks G/table/D second-moment memory by their squared ratios (about299.34/48.13/17.35). Thus this action increases Adam steps during ordinary early training. The first critic clipping is at update202 after the above-threshold streak began; subsequent clipping increases after the fire. Noise reactivation, population transport and KA2 blend transition do not explain this MNIST event.

## Bounded repair direction

Inspect an actuation rule that reopens only owners whose LR tester has settled below1, and treats explicit loss-phase changes as a new detector measurement epoch. Both are general optimizer/loss lifecycle facts. The moving-distribution positive windows are still required to determine whether the useful R1 response survives this rule. No repair is implemented or numerically validated here. Raw q, old fast/slow, ratios, tester state, KA2 records, birth events and noise state are preserved in trace.jsonl; analysis.json includes selected exact contexts. Original actual-candidate quality failures remain preserved in ra12-auto/CLOSED.json.
