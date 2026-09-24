# PR84 follow-up: G critic reach follows D's slope utilisation

Host: neural. Seed 0, one thread, PyTorch 2.14.0+cpu, `ATEN_CPU_CAPABILITY=avx2`
(the repository's parity pin). AVX512 cold rings are reported as a build check,
not a second seed. Receipts: [continuous-evidence/pr84-reach](continuous-evidence/pr84-reach/).
Runner: [gan_followup_probe.py](gan_followup_probe.py). Candidate:
[pr84_reach_candidate.py](pr84_reach_candidate.py).

## Mechanism

G's five-point critic stencil width becomes `max(.15, .5·min(s/κ, κ/s)/κ)`.
Here `s` is D's RMS input slope on the clean particles and `κ = 1` is the host's
b_cap slope limit. PR84's `min(.15, .5/s)` is capped at .15 on every update, so it
never adapts. The new width equals PR84's whenever `s ≤ .3`, which covers the
covered, indistinguishable cloud. It widens to .5 (PR84's own uncapped value at
`s = κ`) only while D is saturating its Lipschitz budget and still separating.
D, both own-curvature bounds, and the losses are unchanged.

**Purity:** GAN dynamics only. There is no coverage, likelihood, anchor,
assignment, mode-count or clip-ladder term. Mode centers are read only by the
offline diagnostic, after training.

## Why this formulation

The pinned PR84 failures share one signature. That signature appears in the
AVX512 stuck seven-mode cold ring and in the own-acquired stay after it
collapses at around update 1800. In both, D sits at its slope limit (sharpness
≈1.0–1.1, advantage .12–.17), while G's own-curvature trust factor collapses to
.06–.08 and its read of the critic stays at .15.

In the covered state, D's slope is ≈.15 and its advantage is ≈.03. D's own
slope utilisation therefore separates "game unresolved" from "game at rest",
without reading coverage.

At the stuck state, the missing mode has the highest critic value (5.25). The
wider read orients all three nearest particles toward it: at width .15 the
projections are −.01/.97/.45, and at width .5 they are .86/.73/1.04. This is the
"idle once covered" critic channel that #97 asked for.

Rejected alternatives:
- R1/R2 zero-centred pulls: already screened, and ruled out by the board.
- Optimism and ExtraAdam: both failed mode-hold.
- The one-step unroll: it does not reverse the outward field.
- Functional output metrics: they fail the trajectory gate.
- Rest-damping on slope: it damps acquisition.
- Barrier probes: killed in #97.

## Gate table (AVX2 pin)

| Gate | PR84 pin | Reach .5 (candidate) | Reach 1.0 (revision 1) | Saturating ramp (revision 2) |
| --- | --- | --- | --- | --- |
| Warm 1001–1200 | FAIL 196/200. Failures 1129–1132, min HQ .866 | **PASS 200/200**, min 8 modes, min HQ .921 | 199/200, fails 1095 (min HQ .900) | 199/200, fails 1095 (min HQ .869) |
| Cold trajectory | PASS | PASS (not a 2-D input, so identical to PR84) | PASS | PASS |
| Cold ring 1200 | PASS 8. Terminal HQ .995/.988/.988/.996/.999. 10/24 observations, suffix 5 | **PASS 8**. Terminal HQ .991/.997/.998/1.0/1.0. 11/24 observations, suffix 10 | FAIL, 5 modes / HQ .438 | FAIL, 6–7 modes, HQ .60–.80 |
| Own-acquired stay 1210–2400 (constant rate) | 53/120, min 0 modes, **final 6 / .904**. Degenerate from 1770 to the end | **103/120**, min 0 modes during 1710–1970, **final 8 / 1.0**. Last failure 2070, 33-check passing suffix | 18/120, final 8 / .921 | 36/120, final 8 / 1.0 |
| Cold ring, AVX512 build | FAIL 7 (1200 HQ .993) | FAIL 7 (1200 HQ 1.0; 4 modes at 1150) | FAIL 1 mode | **PASS 8**, terminal HQ .97–1.0 |

The AVX512 scheduled warm prefix itself reaches only 6 modes at update 1000,
with the identity fork at 0/200, so warm is valid only on AVX2.

Reach 1.0 was one declared revision. The AVX512 reach stuck state points toward
the missing mode only at widths ≥ .5, while the run used ≈.35 there. Reach 1.0
regresses warm and destroys acquisition on both builds. **Killed.**

### Stuck-state fork

[pr84_reach_stuck_fork.py](pr84_reach_stuck_fork.py) forks the identical AVX512
reach state at update 1000, which has 7 modes, and continues it to 1200 three
ways. This is a counterfactual probe of one state, not a candidate.

| Variant | 1050 / 1100 / 1150 / 1200 | Mean G trust factor |
| --- | --- | --- |
| Reach .5 as-is (width ≈ .35) | 7 / 7 / 4 / 7 | .10 |
| Width held at .5, bound on | **8 / 8 / 8 / 8**, HQ .66 / .94 / .98 / .92 | .09 |
| Width .5, G bound removed | 0 / 0 / 2 / 1 | 1.0 |

The blocker is the `min(s, 1/s)` ramp keeping the width under .5. It is not the
trust region: the bound is protective.

Revision 2 was the final declared revision: exact PR84 width up to `s = .3`, a
linear rise to .5 by `s = .6`, and .5 above that. It swaps which build acquires:
AVX512 reaches ring 8, but AVX2 falls to 6–7. It also costs warm and the stay.
Holding .5 through the whole acquisition phase makes the outcome chaotic rather
than robust. **Killed.**

## Rank claim (GAN-native track only)

On the pinned AVX2 harness, reach .5 is the first GAN-native entry with warm
200/200, trajectory PASS, cold ring 8, and an own-acquired constant-rate
continuation that ends on the full ring. PR84 on the same harness ends that
continuation degenerate at 6 modes. That makes it rank 1 over PR84 for
acquire + stay on this build.

It is not a clean stay. There is a 26-check episode with whole-cloud dropouts,
including 0 modes at 1780 and 1970, each recovered within 10 updates. And the
ring is still build-sensitive: AVX512 remains at 7, the same as PR84.

## Keep / kill / next bet

- **Keep** reach .5 as the GAN-native reference. It dominates PR84 on warm and stay at unchanged ring acquisition.
- **Kill** reach 1.0 and the saturating ramp.
- **Not pursued:** #100's "shape D where the cloud is absent" idea. The stuck-state probes show that D already points the nearest particles at the missing mode once G reads it at width ≥ .5, so the missing piece is on G's read side, not in D's field.
- **Next single bet:** widen to .5 on a stall, not on D's slope alone. The trigger is D at its slope limit (`s ≥ .6`) **and** G's trust factor collapsed, for example `≤ .1`, as in both the AVX512 stuck ring and PR84's post-collapse stay. Otherwise keep reach .5 unchanged. Revision 2 shows that D's slope alone is on during healthy acquisition too. The fork shows that a width of .5 from a stalled state recovers 8 modes with the bound kept.
