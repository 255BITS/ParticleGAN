# Prior-horizon 4000 diagnostic

This bounded scratch run changes only the particle-prior cosine horizon from
the full 7000-step budget to 4000. The production affine/square generator,
uniform-square particle initialization, G/D horizon 1600, shared optimizer,
noise schedule, seed 1234, and 7000-step budget are unchanged. The archived
script and run mark `shared_gate_eligible: false`; this is not 22-toy evidence.

| Grid100 live measure | Full-budget prior control | Prior horizon 4000 |
|---|---:|---:|
| First accuracy pass | step 6000 | step 3500 |
| First five-check accuracy streak | step 7000 | step 4500 |
| Uninterrupted passing suffix through step 7000 | from step 6000 | from step 5750 |
| Final 20k HQ | 0.9868 | 0.98305 |
| Final 20k mass TV | 0.0435 | 0.04335 |
| 100k holdout HQ | 0.98636 | 0.98224 |
| 100k holdout mass TV | 0.03565 | 0.03583 |
| 100k holdout center RMS / Gaussian σ | 0.13912 | 0.12495 |
| 100k holdout covariance-trace bias | −0.03706 | −0.03036 |
| 100k holdout radial KS | 0.01214 | 0.01064 |

The new and archived runs match exactly before the prior schedules diverge:
step 250 and 500 generator losses and HQ agree bitwise, and step 2250 HQ,
covariance bias, and radial KS agree. After the 4000-step prior decay begins
at step 2400, width and radial shape settle much earlier: at step 4000 HQ is
0.98495 versus 0.9333 in the control. Yet center RMS drifts above its 0.20σ
limit at steps 5000–5500, interrupting the early fidelity streak. The final
uninterrupted suffix improves by only one 250-step evaluation interval.

The saved original coverage gate and accuracy gate both report PASS for
grid100, including the full terminal five checks and independent 100k
holdout. Since the
durable improvement is small and was tested only on grid100, this prior-only
cap is not being advanced to rotated100 or staggered100. The saved run contains
every actual G, D, and prior learning rate, 20k scored draws at all five
terminal checks, the 100k holdout, and its exact source archive:
[`artifacts/toy100-accuracy/prior-horizon4000/grid100`](../../artifacts/toy100-accuracy/prior-horizon4000/grid100).
