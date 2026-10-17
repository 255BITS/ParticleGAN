# Preserve routed additive code conditioning while neutralizing time

This is a task-derived initialization diagnostic. It changes only fresh additive
FiLM time columns after public deterministic orthogonal initialization, keeping
the additive code columns intact. Training remains pure paired-error RpGAN with
all E22 routed controls. No core policy or public default changes.

PR #231 verified that zeroing the whole additive FiLM branch removes its direct
code-conditioning path. In the real Nova → Qwen task that initialization reduces
first-step encoder gradient energy 107× and finishes 43.2% worse in decoded LPIPS
at update 4,000. Those observations motivate this experiment; they do not prove
this candidate improves the real task. The earlier rate-bypass counterfactual
was negative and is not included here.

For hidden activation `h`, normalized code `c`, and time features `t`, the
conditioned activation is `h*(1+gain(c,t))+shift(c,t)`. This candidate preserves
`W_shift,code` and zeroes only `W_shift,time` and the additive bias. At `h=0`, the
direct additive code derivative remains live. This retains the `Cz` path rather
than erasing it with the host/time initialization.

The fixed campaign compared original, whole-shift zero, and time-only zero at
widths 4 and 16, in that order. Every arm ran 1,200 updates with batch 16 and the
existing spatial fixture, learned critic, BF16 frozen prefix/head, public role
initialization, named caller streams, 128×4 coupled particle-cloud bank, output
sigma 1.3, KA2, native settlement and the settled reopen guard. MSE is clean
evaluation only, never a training or restructuring acceptance loss. The bank is
an explicit particle-cloud exception, not ordinary Forge qualification.

| Final clean live/public-served MSE | Original | Whole shift zero | Time-only zero |
| --- | ---: | ---: | ---: |
| Spatial width 4 | .000143826139 | .000166828046 | .000147896877 |
| Spatial width 16 | .000096847813 | .000074593947 | .000049226761 |

Width 16 improves **49.17% against original** and **34.01% against whole-zero**.
Width 4 is a counterexample: time-only zero improves whole-zero by 11.35% but
finishes **2.83% worse than original**. Restoring the path does not improve every
host width. The prior width-16 antithetic arm still has a lower archived MSE;
this candidate is not the best toy configuration across all changes.

| Initialization measurement | Width 4 original / whole-zero / time-only | Width 16 original / whole-zero / time-only |
| --- | --- | --- |
| Code-Jacobian norm | .00491086 / .00040320 / .00499592 | .00464012 / .00080293 / .00463516 |
| First E gradient squared norm | 7.70004e-11 / 1.61101e-12 / 8.13984e-11 | 8.62724e-9 / 2.74780e-11 / 8.54628e-9 |
| First actual Adam E displacement squared norm | 1.08320e-5 / 4.39631e-6 / 1.09733e-5 | 1.18925e-4 / 4.28667e-5 / 1.18842e-4 |

The restored code Jacobian and first encoder gradient are near the original
values at both widths. Gradient energy and actual optimizer displacement are
reported separately because Adam preconditioning changes their relationship.
Width-16 time-only zero receives one native G cut at update 1,032, with 168
subsequent damped updates at half rate. All other arms have no G cut. The
experiment preserves this native decision and does not attribute the quality
improvement to damping alone.

All six arms finish with finite losses and gradients, live encoder/table
gradients and 128/128 bank gradient rows on every update. Non-G initial owners
and data hashes match, and the caller batch/Gaussian streams match at the fixed
endpoint. All frozen source/package bytes remain unchanged. The 29 CPU contracts
include retained code columns, unchanged unrelated owners, the restored routed
gradient, native controls and exact checkpoint replay. Total paid CPU time was
191.03 seconds. Each width had an external 900-second cap; clean divergence or
nonfinite training stops the run. No seed or parameter sweep was performed.

The [compact receipt](routed_film_code_preserved_20261002.json) binds current
`develop` commit `c4b53495`, source hashes, every fixed clean metric point,
the immutable campaign protocol, complete local readout and JUnit receipt.
Bulk traces and checkpoints are outside Git at
`/ml2/hypergan/routed-film-code-preserved-artifacts-20261002`.

The matched Nova → Qwen GPU pilot is now complete on the same `develop` commit.
It uses the sealed expanded data (10,806 training rows, 176 validation cases),
full native E22 and GAN-only training. Current `develop` first reproduces all
100 original scientific rows, caller/private streams, complete initial/final
checkpoints and clean64 per-row observations exactly. This qualifies reuse of
the immutable original curve; it does not claim a new 4,000-update replay.

| Full176 decoded validation LPIPS | Original native served | Time-only zero native served |
| --- | ---: | ---: |
| 100 updates | .484361237 | .473543106 |
| 300 updates | .389702035 | .367771339 |
| 600 updates | **.300055678** | .310032038 |

The frozen gate stops the candidate at 600: **3.32% worse than control**, with
166/176 cases worse. Both clean live monitors continue improving; at 600 the
guard clean MSE is .248680815 versus original .249099344, a small .17% gain.
Decoded end-to-end quality and clean live MSE therefore tell different stories.
The native policy serves **fast** candidate and **averaged** original at these
checkpoints. This is an identical-policy comparison with different decisions,
not a fixed-serving optimization comparison. The early decoded gain is not
attributed solely to weight convergence. The strongest earlier fresh B16 GAN
at 300 scores .295439204 with fast serving, and remains substantially better.

An independent completed review verifies all 600 updates are finite, all 128
bank rows receive gradients, additive code and encoder gradients remain live,
caller batches/Gaussians match the original, and all 88 protected inputs and
30 package files are unchanged. The public-init recipe remains unchanged.
No reconstruction/VIC loss, evaluation gradient or held-out test is used in
training. The external supervisor exits successfully. Bulk real artifacts are
at `/ml2/hypergan/nova-qwen-code-preserved-20261002` and independent evidence at
`/ml2/hypergan/nova-qwen-code-preserved-independent-review-20261002`.
The [published real-task archive](https://github.com/255BITS/model-glue/tree/441a3e0817833e1bb0fcfcc73c88237b0768d690/experiments/e22_gan_code_preserved_20261002)
contains the frozen driver, protocol, compact scores and independent receipt.

This ablation restores code-conditioning geometry but **does not solve the
real convergence failure**. It supplies a negative transfer check for future
changes. No winner, ordinary Forge qualification or public-default adoption
is claimed. The compact receipt binds the exact task scores and audit hashes.

Reproduce from the repository root, using a new empty artifact directory for
each width:

```sh
timeout --signal=TERM --kill-after=15s 900s python -u \
  -m benchmarks.routed_conditioning.spatial_damping --steps 1200 --width 4 \
  --profile original_native --profile shift_zero_native \
  --profile shift_time_zero_native --output /tmp/film-code-width4
timeout --signal=TERM --kill-after=15s 900s python -u \
  -m benchmarks.routed_conditioning.spatial_damping --steps 1200 --width 16 \
  --profile original_native --profile shift_zero_native \
  --profile shift_time_zero_native --output /tmp/film-code-width16
```
