The prepared diagnostic tests whether PR227's neutral H/b result transfers to
a reachable target whose input basis must be learned. PR227's teacher copies
the students' initial down factors. This task preserves every fresh student
tensor and teacher up factor, but rotates the teacher down row spaces.

The fixed rotation uses `rho = 0.027625366713810866`, the mean fraction of the
historical ordinary Supra adapter's linear-delta weight energy captured by the
fresh sampled particle down span across 71 sites. Thin QR and a deterministic
canonical complement preserve each down factor's within-rank Gram. There is
no new seed or scalar search. Both adapter families can exactly represent the
new teacher by learning its down/up factors.

This is a specified basis-acquisition regime. The audit covers 70 rank-16 projections with input width 576 and one with
input width 768. Its equal-site isotropic chance overlap is 0.027680, close to
the measured 0.027625 energy fraction. The rank-2, width-16 toy's chance
overlap is 0.125. The transplant is therefore more severe than a dimension-
normalized toy. Canonical directions change activation covariance, target
outputs and the derived fixed coordinate scale; preserved weight Gram does
not preserve task difficulty. The historical trained ordinary span is also
not an estimate of Supra's exact frozen-caption target span.

The [card](e22_routed_convergence_rotated_teacher_v1.json) holds quality
execution pending the independently qualified full Supra final comparison.
The coordinator will enable this single task if neutral particles at 6,400 do
not beat the historical ordinary reference under both required common critics.
H/b gains over the fresh sampled control remain a separate result.

The three native-game arms are ordinary LoRA, original particles, and particles
with only fresh H/b zeroed before policy and EMA construction. They keep the
public initialization, learned conditional critic, native RpGAN/KA2, shared
128×4 R2 bank, editing-only 6,400 budget, and zero feature-harm guards. Output
error is an offline diagnostic. There is no fresh MSE training arm.

```sh
PYTHONPATH=. python -m pytest -q tests/test_e22_routed_convergence_rotated_teacher.py
# After the coordinator freezes and enables the conditional card:
PYTHONPATH=. python -u examples/run_e22_routed_convergence_rotated_teacher.py --out runs/routed-convergence-rotated-v1
```

The runner saves 35 states per arm: zero, every 200 updates, 5,120, and the 802
recovery witness. It scores all 34 fixed curve points with all four baseline
critics trained at 800 and 6,400. It records both 5,120 and 6,400 endpoints, paired panels,
all critic reference paths, signed code contributions, frozen owners, full
native control traces and exact 800-to-802 replay. Offline scoring must leave
the full checkpoint, diagnostics, training streams, data and critics unchanged.

`receipt.json` and `compact-report.json` report execution and scientific gates;
independent qualification remains pending until a reviewer verifies the actual
states, traces, common critic weights, raw scores and replay within the 2,700-
second total budget. Bulk artifacts stay outside Git and progress is JSONL in
`run.log`. A remaining gap is a failure witness for this fixed configuration;
different native rates or noise decisions prevent a unique basis-angle claim.
Absolute scores cannot be subtracted from PR227 because the teacher, fixed
scale and learned judges differ. No quality run has been performed.
