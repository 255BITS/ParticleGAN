# Gaussian regression: finite acceptance versus direction

This preregistered three-arm bounded diagnostic compares exact local-v2,
exact PR367 direction blend plus local-v2, and exact PR368 strict finite
projection plus local-v2. The question is whether removing finite acceptance
preserves the rare/broad repair and restores complete Gaussian retention and
reacquisition. No tuning, seed study or ordinary qualification is authorized.

All arms retain the exact saved BCAP winner: nonsaturating loss, full DualNorm
smoothing .001/momentum0/per_offset, G .012, D .018, prior .030, constant floors1,
BCAP cap/coefficient1/every update, zero additive training noise, clean/live
sampling. Local-v2 uses global/local weights1 and32 deterministic projections.
Only `constraint_geometry_mode` changes: none, direction_blend, strict_progress.
Scalar hosts protect the same existing adversarial loss; transport is included
in the total objective but is never another protected loss. The direction module
is byte-exact PR367; strict and transport modules remain byte-exact PR368.

[Saved-state receipt](prior-evidence.json) and [restore source](probe_saved.py)
restore four certified round-four Gaussian states exactly without updates or
sampling. Finite has1,407 conflicts/accepted steps, zero rejections,151 reductions
and minimum accepted scale.25 over6,000 steps. Local-v2 has28/72 stationary and
11/24 shifted-hold passes; finite has8/72 and1/24. Both fail reacquisition.
The archive has endpoint counters and scheduled quality curves, not per-update
conflict/scale events; those counters cannot attribute the entire regression.
Smoke independently certifies an earlier scheduled state but continuation
restores the arm's own completed1,000-update smoke checkpoint, without reset.

Common descent is a local directional construction related to
[Sener and Koltun's multiobjective treatment](https://arxiv.org/abs/1810.04650).
The finite layer uses same-batch sufficient decrease motivated by
[Armijo](https://msp.org/pjm/1966/16-1/pjm-v16-n1-p01-s.pdf).
Neither cited deterministic property establishes distribution convergence in
this changing stochastic game. With one protected scalar, the blend mixes the
nonascent projection and a negative unit protected gradient; direction-only
uses full scale without evaluator calls.

The unchanged tasks are Gaussian smoke1000, own-checkpoint stability+5000,
unequal mass1200, broad1200 and unequal width1200. Full reservations are6120
per arm and18360 total. Queue ceiling21180 plus120 for saved restoration and300
for bounded synthetic software/scorer checks equals the21600 track cap.
No training probe or capacity test is planned. Any retry consumes that same cap.
A failed own smoke blocks stability. All72 stationary checks, the deadline's
five terminal reacquisition checks, and all24 shifted-hold checks determine
Gaussian PASS. EndpointKS<=.05 is only the registered forecast. Vector gates
retain every original mass/quality/covariance/eigen/rare-density bound and five
terminal passing checks; lower covariance alone cannot claim repair.

Protocol seed0, public deterministic initialization, task architecture, target
law, actual batches, MoG implementation/width/weights/learnable locations,
committed update budget, scoring cadence and sampling are fixed across arms.
Gaussian/vector hosts retain their actual one-real-tensor D/G reuse. Constructor,
data, prior/noise and evaluation streams are isolated and checkpointed. All
arms are submitted on one frozen scientific source/runtime before public
Queue/drain execution with shared GPUs and no full-compile callback. Saved
scorer controls and actual-training GIFs will accompany the completed readout.
Parent owns the single current goal leaderboard; this report has a scoped table.

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round5-20261009/gaussian_regression/logs/driver.log
PYTHONPATH=. /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/gaussian_regression/round5/run.py
```
