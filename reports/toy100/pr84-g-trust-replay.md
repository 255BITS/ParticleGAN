# G trust bet: one step from the AVX512 reach stuck state

Host: neural. Seed 0, one thread, PyTorch 2.14.0+cpu, CPU capability AVX512.
No `ATEN_CPU_CAPABILITY` override. Receipts:
[continuous-evidence/pr84-g-trust](continuous-evidence/pr84-g-trust/).
Replay: [pr84_g_trust_replay.py](pr84_g_trust_replay.py).

## Mechanism

None installed. The named bet was a G own-curvature trust region driven by D's
slope utilisation. Its build rule was one replay: from the saved AVX512
reach-.5 state, at stencil width .5, with the .25 bound on and off. Build only
if the missing mode is approached with the bound off and not with it on. If
neither step approaches, shared-network coupling is the barrier and the line
stops. Both steps approached, so the trust region was not built. Reach width,
the #104 predicates, and both curvature constants are unchanged.

**Purity:** the replay trains the unchanged reach-.5 cold ring. Mode centers
are read only after the step, to grade displacements.

## Why this was the bet

#107 kept reach .5. On AVX512 the cold ring still ends at 7 modes. At that
state the missing center is the critic's peak, and a width of .5 points the
three nearest particles at it, but the run's own width was still on the
`min(s, 1/s)` ramp and G's trust factor was ~.11. The open question was
whether that factor, rather than the critic direction, was what kept the
generator off the hole.

## Replay

The weights were not in the tree. The same reach-.5 constant-rate ring was
run to 1000 completed updates on this AVX512 build. The capture matches the
#107 receipt: missing mode 3, occupancy `1,2,1,0,3,1,1,3`, nearest distances
`2.243 / 2.252 / 2.299`.

That next G step was then run twice at forced width .5. Same weights, Adam
moments, and RNG. `ρ_G = 2.474` on both sides (sharpness 1.75 after the D
half-step, so the unforced reach width would have been .286). Bound on uses
`c = .25` and lands at factor `.101`. Bound off uses a non-binding cap and
lands at factor `1`.

| Replay | Factor | Distances of particles 5, 2, 4 | Radial steps | Closer? |
| --- | --- | --- | --- | --- |
| Bound .25 | .101 | 2.243→2.090, 2.252→2.108, 2.299→2.149 | +.153, +.144, +.151 | yes |
| Bound off | 1 | 2.243→0.841, 2.252→0.890, 2.299→0.925 | +1.509, +1.437, +1.432 | yes |

The two displacements are one Adam direction at two lengths. The radial steps
scale by about 9.9, the same ratio as `1 / .101`. The shortened step is not a
different direction. All three particles are the mode-4 cluster (the left
site). With the bound on they stay on that site. With the bound off they
leave it and stop near distance .84, still outside the .21 ball, so one full
step does not acquire mode 3 either.

## Gates

No warm, cold, or stay run. The build predicate failed before a candidate
existed, and the instructions forbid a second mechanism or a coefficient
sweep after a kill.

| Gate | PR84 pin (AVX2) | #107 reach .5 (AVX2) | This bet |
| --- | --- | --- | --- |
| Warm 1001–1200 | 196/200 | 200/200 | not run |
| Cold trajectory | PASS | PASS | not run |
| Cold ring 1200 | 8 | 8 | not run |
| Own-acquired stay | 53/120, final 6 / .904 | 103/120, final 8 / 1.0 | not run |
| AVX512 cold ring | 7 | 7 | unchanged reach-.5 state, still missing mode 3 at update 1000 |

## Keep / kill / next bet

- **Kill** slope-conditioned G trust. The missing mode is approached with the .25 bound on. The bound scales the step; it does not decide whether the step points at the hole.
- **Keep** #107 reach .5 as the GAN-native reference. This replay does not move that rank.
- **Next bet:** none on this line. A utilisation-dependent trust region would be a longer step along a direction the bound-on update already takes. Shared-network inability is also the wrong kill: the outputs do move toward the hole. The stuck fact left on the table is that those movers are the covered mode-4 cluster, and even the unbound step does not enter the hole.
