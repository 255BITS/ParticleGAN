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

The [card](e22_routed_convergence_rotated_teacher_v1.json) now authorizes this
single task following the user's explicit request to isolate the problem in
a toy while the full Supra comparison continues. This execution condition
changed before any quality toy updates. The earlier source-only card remains
archived at commit `1cce9072`, SHA
`36e8694e48ea02bf05854dad6dab8d519c58f9d7db5776da6fc0b14327791b8a`.
The native law, task geometry, arms, horizons, gates and budgets are unchanged.
Full Supra qualification is still pending; it is not claimed as launch evidence.
H/b gains over the fresh sampled control remain a separate result.

The three native-game arms are ordinary LoRA, original particles, and particles
with only fresh H/b zeroed before policy and EMA construction. They keep the
public initialization, learned conditional critic, native RpGAN/KA2, shared
128×4 R2 bank, editing-only 6,400 budget, and zero feature-harm guards. Output
error is an offline diagnostic. There is no fresh MSE training arm.

```sh
PYTHONPATH=. python -m pytest -q tests/test_e22_routed_convergence_rotated_teacher.py
# The authorized card is frozen before the single quality run:
PYTHONPATH=. python -u examples/run_e22_routed_convergence_rotated_teacher.py --out runs/routed-convergence-rotated-v1
```

The runner saves 35 states per arm: zero, every 200 updates, 5,120, and the 802
recovery witness. It scores all 34 fixed curve points with all four baseline
critics trained at 800 and 6,400. It records both 5,120 and 6,400 endpoints, paired panels,
all critic reference paths, signed code contributions, frozen owners, full
native control traces and exact 800-to-802 replay. Offline scoring must leave
the full checkpoint, diagnostics, training streams, data and critics unchanged.

`receipt.json` and `compact-report.json` report execution and scientific gates;
the run also preserves exact held source/card/native Python bytes in `source/`
and its frozen `data.pt`, with full-file hashes for independent review.
Independent qualification remains pending until a reviewer verifies the actual
states, traces, common critic weights, raw scores and replay within the 2,700-
second total budget. Bulk artifacts stay outside Git and progress is JSONL in
`run.log`. A remaining gap is a failure witness for this fixed configuration;
different native rates or noise decisions prevent a unique basis-angle claim.
Absolute scores cannot be subtracted from PR227 because the teacher, fixed
scale and learned judges differ.

The single authorized run completed and passed independent review: 569,601
checks, all 105 states, 102 curves, and three exact recovery replays. Training
and scoring took 762.831 seconds; independent review took 54.934 seconds,
817.765 seconds combined within the declared 2,700-second budget.

Final held-out paired games, lower is better:

| Common critic | Ordinary native game | Original particles | Neutral particles |
| --- | ---: | ---: | ---: |
| Ordinary at 800 | 1.818564 | 0.951775 | 1.224921 |
| Ordinary at 6,400 | 2.066303 | 1.943993 | 1.881532 |
| Original particles at 800 | 1.921654 | 1.091335 | 1.286375 |
| Original particles at 6,400 | 2.173814 | 2.315960 | 2.116456 |

Neutral particles beat ordinary native-game LoRA under every common critic
at both 5,120 and 6,400, including each of the six subjects. Both particle
arms retain learned bank/router/code contributions; zeroing codes worsens
all four final scores. Original particles versus ordinary are mixed across
the two final critics, so the original disadvantage is not consistently
reproduced and gap reduction is inapplicable. H/b neutralization also does
not improve original particles under both early critics, so its all-four
support gate fails.

This task does not identify the full-Supra cause. The qualified
[results and provenance](e22_routed_convergence_rotated_teacher_results.json)
retain both endpoints, all subjects, signed contributions, reference rays,
failed gates and their limits. A separate guidance-pair task is being prepared
to test another specific host difference; no angle, seed or scalar scan follows.
