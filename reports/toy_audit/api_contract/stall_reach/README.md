# Stall-reach critic read: 8-mode ring (public-API toy)

**Question.** With the BCAP direction-blend leader recipe
(`recipe-prior-refactor-bcap-pure-v1` overrides) plus
`generator_critic_read="stall_reach"` (closed PR #107), does the generator still
acquire and hold the 8-mode ring (radius 2, sigma .07)?

**Execution.** `GANTrainer` through the public API, 1200 updates, batch 128,
z_dim 4, 12 learned particles, `SimpleMLPGenerator(4, 96, 3)` /
`SimpleMLPDiscriminator(2, 96, 3, fourier 3)`, `deterministic_orthogonal_`
(G seed 0, D seed 1), trainer seed 0, CPU, one thread.
Source: [`benchmarks/toy_audit/stall_reach_ring8.py`](../../../../benchmarks/toy_audit/stall_reach_ring8.py).

**Gate (declared before training).** Final update: `modes >= 7` and
`hq >= 0.90` on 4096 clean live samples (HQ = within 3 sigma of a center).

| Arm | Final modes | Final HQ | Gate | Widened updates | Stalled updates | Max width |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `generator_critic_read="none"` (same toy) | 7 | 1.000 | PASS (first pass at 650) | - | - | - |
| `generator_critic_read="stall_reach"` | 3 | 0.240 | **FAIL** | 1198/1200 | 964/1200 | 0.5 |

The read stalls from about update 100: BCAP holds the critic's measured input
slope near its cap (`u ≈ .93 ≥ .6`), so the 50-update saturation window fires
almost continuously. The width stays at `REACH/kappa = .5`, about 7 sigma, and
the generator cannot resolve modes. PR #107's original stall test also required
G's own-curvature trust factor to collapse. The current recipe has no trust
factor, so the approved rebuild used slope saturation alone, which is not
selective under BCAP.

![stall-reach ring8 training](stall-reach-ring8.gif)

Reproduce: `python -m benchmarks.toy_audit.stall_reach_ring8 --output runs/stall-reach-ring8`.
`tests/test_generator_critic_read.py::test_ring8_public_api_toy_runs_with_declared_gate`
runs a 100-update execution check.
