# Capped Monge prior step: warm gate fails immediately

**Stopped at the dense warm gate. No cold trajectory, ring, or hold.**

The rule is an added sampled-data objective on the frozen PR84 alternating update (G-only five-point stencil, curvature bounds .25/3, constant rates). After that update, farthest-point sampling picks 12 real representatives from the current 128-point minibatch. The Hungarian algorithm builds a minimum-cost bijection with the 12 clean particles. Each prior row then takes at most a 0.2 output step toward its match, pulled back through the row Jacobian in the current Adam metric. A step is kept only when the frozen assignment cost falls. Trajectory would have stayed on unmodified PR84; it was not run.

This is not a rest-damping, slope-gate, or mode-center rule. It targets the valley geometry in `pr84-field-diagnosis.md`: Voronoi means and Sinkhorn barycentric maps send one particle to the average of two modes. A bijection cannot land in that midpoint.

## Why this was the test

Peyré & Cuturi separate the Monge map from the Kantorovich barycentric projection. He et al. 2026 (arXiv:2603.12366) Sinkhorn drifting is barycentric; `nonlocal-signal-filter.md` already rejected one fixed configuration because a distinct eight-mode cloud lost quality. Chamfer (`chamfer-projection-report.md`) passes warm here in the Codex cu126 replay, then fails ring because several ideal targets themselves miss modes. A bijection to real representatives was meant to place surplus mass on an uncovered mode instead of in the valley.

## Same-process warm fork

Python 3.12, PyTorch **2.13.0+cpu**, one thread, AVX2. This is not the cu126 archive machine. The correction-disabled PR84 child therefore does not hash-match `smooth40-warm`. In this process it scores **196/200** (failures 1129–1132, min HQ .86621, final 8 / .99927), which matches PR84's reported warm miss rather than the cu126 200/200 replay. Identity is 200/200. Constant Adam .00425 is 4/200. Those controls show the fork is alive. The Monge arm is judged against them, not against the cu126 hash.

| Arm | Warm checks | Final modes / HQ |
| --- | ---: | --- |
| Scheduled identity | 200/200 | 8 / .99902 |
| Constant Adam .00425 | 4/200 | 8 / .64526 |
| PR84 stencil, correction off | 196/200 | 8 / .99927 |
| PR84 + capped Monge | **0/200** | 8 / .65210 |

Every one of the 200 Monge proposals was accepted at α=1. The first proposal already fails the gate. Seven of twelve particles were matched **2.15–2.33** away (about one ring spacing). The cap then walked them 0.20. Latent displacement norm was only 0.367, and every row Jacobian had rank 2 with a zero linearized residual. This is not the Chamfer latent explosion. The bijection itself orders covered particles onto other modes, because twelve farthest-point representatives are a different occupancy than the twelve particles that already cover eight modes. Decreasing that assignment cost leaves the HQ ball (radius .21).

## Handoff

| Method | Warm | Cold trajectory | Cold ring |
| --- | --- | --- | --- |
| Capped Monge bijection, output cap 0.2, on PR84 | **FAIL 0/200** | not run | not run |

A balanced bijection cannot both rest on a matched 12-to-8 cloud and walk one particle into an empty mode: any target set whose occupancy differs from the particles forces a mode-scale move. Unbalanced nearest-neighbor targets were already tested (coverage, Chamfer). No cap sweep was run. The selected partial candidate remains the original PR84 stencil.
