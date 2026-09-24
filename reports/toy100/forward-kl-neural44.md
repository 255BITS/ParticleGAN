# Finite-GH9 likelihood: native saved44 filter

**PASS44/44, eight modes, minimum HQ .999755859375.** This is an actual neural
G/D and Adam replay, with the new likelihood correction applied on every
update. It is a saved-state stability filter, **not cold acquisition**, an
own-acquired hold, or a production pass. The learner starts fresh at each
saved window; it does not claim to resume earlier likelihood history.

| Consecutive native window | PR84 control | Likelihood correction | Minimum candidate HQ |
| --- | --- | --- | --- |
|1324–1335|9/12; min HQ .775391|12/12|.999756|
|1380–1395|14/16; min HQ .878662|16/16|1|
|1530–1545|1/16; minimum seven modes|16/16|1|

All44 nonlinear fits converge and their **actual realized outputs** strictly
lower the finite GH9 cross-entropy for that update's cumulative data. The
smallest measured decrease is .0006279654. There are87 additional joint-output
Jacobian constructions across44 corrections. Native D/G Adam moments advance
once per update; nominal rates remain G/D .00425 and prior .0085, with the
original noise clock. Correction preserves RNG, D parameters, both optimizer
states and native gradient buffers. The original branch matches archived
states, observations and update records exactly, and both branches finish
with equal RNG/noise state.

The complete44-window/control experiment took about **165 seconds**, measured
between its declaration and final-result file timestamps on one CPU thread.
This is not a throughput measurement for a long cold run: data history starts
over for each12/16-update window. The cumulative estimator grows in both memory
and per-update work, so a full neural rollout is not yet a practical next step
without bounding its cost. The target uses an added density objective, not a
pure correction of the original GAN objective.

[Frozen sources, raw results and manifest](continuous-evidence/round9-forward-kl-neural44/manifest.json)
bind driver commit `be60828`, adapter SHA `44ea3bf3d50e32fa11231d779c6c7888f13dca4246f230f03c797c486b06a9ff`,
and capture SHA `37aa612bd3b3e1867a2d1508674158849ccf5940c907af76b1f8b061f4eb3e47`.
Seven focused adapter tests pass, including an actual pure-operator call and
exact-rest receipt accounting. No test uses mode labels to choose updates.

Reproduce from the archived capture with one-thread CPU/AVX2 settings:

```bash
python -u -m reports.toy100.forward_kl_neural_filter \
  --capture /path/to/capture-v2 \
  --output /new/output/directory
```

Keep as a **neural-filter lead**, below completed neural acquisition/own-hold
evidence. Finite GH9 descent does not certify exact continuous KL, full target
fidelity, parameter boundedness, or indefinite stability.
