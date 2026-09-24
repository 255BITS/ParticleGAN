# Alternating Adam with same-sample own-curvature bounds (PR #60 follow-up)

**Cold trajectory is cleared without LR decay; the cold ring is not.** The best
candidate (v8) passes the warm fork 200/200 and cold trajectory at MSE .00094
with a 20-check sustained ≤.02 suffix (the gate is unchanged). Its cold ring
acquires only by update 1150 (terminal HQ .43/.52/.83/.92/.92, final 7 modes),
so it fails hard gate 3 and was not run to 2400 or shift. Nothing here is
eligible for production; no 22/22 claim. All numbers are local to this VM
(the archived traces are not bit-reproducible here), with one shared warm hash,
identity/uninterrupted-control state-hash parity in every warm run, and the
constant-Adam control at 4/200.

## Discriminating finding: the simultaneous-update scaffold

Every earlier recorder-based game update (extragradient, implicit, cross-only,
the #81 error-relative guard, and my v1-v4) evaluates D and G at one joint
point, i.e. a **simultaneous** game. The host alternates: D steps, then G's
gradient sees the new D. With every bound disabled, the simultaneous scaffold
fails cold trajectory (MSE .228, diverging from plain Adam by update 17),
while plain alternating constant Adam passes (MSE .00295, 18/24 checks, from
update ~117). The new alternating adapter with bounds disabled reproduces
plain alternating Adam bit-for-bit on both hosts (trajectory .00295, ring
final HQ .5627). Part of every cold-trajectory near-miss so far (.020, .021,
.069) was paid to simultaneity, not to the stabilizer being tested.

## Mechanism

Own-curvature ratio for player p with Adam metric P=lr/denominator and Adam
step Delta, measured by one same-batch, same-noise replay:

    rho_p = ||sqrt(P) (F_p(after) - F_p(before))|| / ||Delta / sqrt(P)||

This is the effective step times curvature along the step; gradient descent
along it is monotone for rho<1 and diverges for rho>2. The step is scaled by
min(1, c_p/rho_p). With D bounded, each update runs three passes: D's Adam
step; a replay that bounds D, after which G takes its Adam step against the
bounded D; and a replay that bounds G. Moments advance once per player, RNG
replay is verified every pass, and the target is never read.

Signals measured before choosing (plain constant Adam):

| Signal | Acquisition | Matched / failing | Separates? |
| --- | --- | --- | --- |
| G own-curvature rho | trajectory 1-3 | warm ring ~2 | No |
| Successive G-step cosine | trajectory -.44 to -.75 | ring late -.78 | No |
| Critic advantage log2 - L_D | trajectory .14-.44, ring .08-.35 | scheduled rest .023, constant late .09 | Partly; the rejected confidence rule already used it |

## Leaderboard (fail-fast; later columns only after earlier pass)

| Variant | Warm | Cold trajectory | Cold ring |
| --- | --- | --- | --- |
| Plain alternating constant Adam (control) | 4/200 | PASS .00295 | FAIL, final HQ .5627 |
| v3 simultaneous, G+D c=1 | 195/200 | not run | not run |
| v3 simultaneous, c=.25 / .1 | 200/200 / 200/200 | FAIL .250 / .229 | not run |
| v4 simultaneous, c=.25, advantage gate .1 | 200/200 | FAIL .262 | not run |
| v5 alternating, G c=.25, gate .1 | 200/200 (min .9297) | PASS .0012, 9-check suffix | FAIL 5 modes / .19; gate open 1085/1200 |
| v6 alternating, G c=.25 | 200/200 (min .9297) | PASS .00099, 16-check suffix | FAIL: HQ .9995 at 1000, 0 modes by 1100 |
| v7 alternating, G and D c=.25 | 200/200 (min .9653) | FAIL .249 | not run |
| **v8 alternating, G c=.25, D c=2** | **200/200 (min .9297)** | **PASS .00094, 20-check suffix** | **FAIL 7 modes / .9155; HQ .43→.92 over 1000-1200** |
| v9 alternating, G c=.5, D c=2 | 200/200 (min .9331) | FAIL .252 (bad basin) | not run |

Cost: 2 (G bound) or 3 (G+D bounds) gradient evaluations per player per
update; 1 moment update per player. v1/v2 (cross-only with bounds) are in
[cross-curvature-report](cross-curvature-report.md).

## What the failures show

* v6's ring collapse is the critic, not the generator. A measurement-only
  rerun (D bound 1e9, bit-identical to v6) shows rho_D climbing 4.2, 7.8,
  20.4 (peak 71) at updates 1070-1075, followed by critic advantage −80. G's
  bound had held HQ .9995 through update 1050.
* Bounding D at c=2 (the divergence edge) removes that collapse, but slows
  ring acquisition past the terminal window. Bounding D at .25 starves the
  critic and loses trajectory acquisition.
* Cold trajectory has a bad basin at MSE ~.25 (compare Codex's swapped
  identities). G gain .5, the tight D bound and the simultaneous scaffold all
  land there; G gain .25 with a fast D does not. The trajectory pass is real
  under the unchanged gate but sensitive to the gain.
* The advantage gate fails on the ring for a structural reason: constant-rate
  oscillation keeps the critic winning, so the gate stays open and never damps
  the oscillation that keeps it open.

## Follow-on: ring hold (still open)

All rows keep v8's alternating adapter and G bound .25. The D bound never
fires on the warm fork (max rho_D .748), so every D bound ≥ .75 has a warm
result bit-identical to the recorded 200/200.

| D bound | Cold trajectory | Cold ring terminal HQ (1000/1050/1100/1150/1200) | Ring verdict |
| --- | --- | --- | --- |
| none (v6) | PASS .00099 | 1.0 at 1000 but only 6 modes, then 0 modes by 1100 (critic divergence) | FAIL |
| 2 (v8) | PASS .00094, 20 | .43 / .52 / .83 / .92 / .92, 7 modes | FAIL |
| **3 (v10)** | **PASS .00094, 18** | **.916 (7 modes) / .843 (7 modes) / 1.0 / .995 / 1.0 (8 modes)** | **FAIL, 3/5 terminal checks** |
| 4 | PASS .00095, 16 | .39 / .79 / .83 / .51 / .67 | FAIL |

The D-bound response is not monotone, so the ring outcome behaves chaotically
in this parameter. Scanning it further until one value passes would be
selection on noise, the same as a seed sweep, and was stopped. v10 is the
nearest miss: one dip at update 1050.

Signals checked for a G bound that loosens while unmatched and tightens at
rest (per-update clean particle traces, updates binned by 100):

| Signal | Ring acquisition | Ring rest (v6, HQ ≈ 1) | Separates? |
| --- | --- | --- | --- |
| Output directedness (net/path, 10 updates) | .20–.50 | .44 | No |
| Critic advantage | .08–.35 | .17–.37 | No; the 12-vs-8 mismatch keeps the critic winning |
| rho_G / rho_D | ~1.1–1.9 | ~5.8 | Possibly (bin medians only; untested as a control) |

The G bound already tightens as rho_G grows, so it rests naturally; the
unsolved part is that it also throttles ring acquisition (mean factor ≈ .13),
leaving acquisition within about 100 updates of the terminal window.

GIFs (gray = mode centers with the .21 HQ radius, color = the twelve clean
generator particles): `v8_ring_cold.gif`, `v6_ring_cold.gif`,
`v10_ring_cold_dbound3.gif` in the run artifacts. Regenerate them with
`alternating_ring_trace.py` and `ring_trace_gif.py`; traces are under
`continuous-evidence/alternating-curvature/ring-traces/`.

## Final two attempts (both fail; v10 remains the best evidence)

| Variant (G .25 base, D bound 3) | Warm | Cold trajectory | Cold ring terminal HQ |
| --- | --- | --- | --- |
| v10 (reference) | 200/200 (.9297) | PASS .00094, 18 | .916 / .843 / 1.0 / .995 / 1.0, 8 modes |
| v11: G bound .25 x clip(2 / EMA(rho_G/rho_D), 1/1.5, 1.5) | 200/200 (.9111) | FAIL .249 (bad basin) | not run |
| v12: separate network / particle-prior curvature bounds | 200/200 (.9624) | PASS .00095, 20 | .58 / .49 / .36 / .66 / .73, 7 modes |

* v11 agrees with the peer rejection of rho-ratio G-bound scheduling: it
  separates acquisition from rest, but loosening early moves trajectory into
  its bad basin.
* The v10 dip at update 1050 is last-mode acquisition. Particle 5 sits .8-.98
  from any center while mode 4 is empty and travels there during updates
  1033-1061. Meanwhile whole-ring shakes (network steps move every particle)
  pull particle 2 out of a doubled mode. v12 tested the matching fix, freeing
  particle latents from the network's sharper curvature, and it made ring
  acquisition worse.
* v13 isolated the particle-level lever. The network and all non-prior
  parameters keep v10's joint factor, and each particle's latent row gets its
  own same-replay curvature ratio. Warm passed 200/200 (min .9236) and cold
  trajectory passed (.00098, 20-check suffix), but the cold ring got worse:
  .57 / .48 / .43 / .39 / .51 with 5 modes.
* Across seven mechanism variants that all keep trajectory passing, ring
  terminal outcomes scatter widely (v10 3/5 at 8 modes down to 5 modes/.51),
  with no ordering by mechanism. A single cold ring run cannot distinguish a
  small real improvement from this spread. A mechanism that clears the ring
  probably has to finish acquisition well before update 1000, not merely
  change who wins a close finish.
* Mode-coverage triggers need the target's mode count, and first-N-update
  boosts are an elapsed-time schedule, so neither was used.

## Critic-side lever (v14) and acquisition margin

v14 keeps v10 (G .25, D bound 3) but takes G's gradient against an
exponential moving average of the critic's parameters (decay .9); D itself
trains normally. Warm fork passed 200/200 (min .9341). Cold trajectory failed
at MSE .251 (bad basin), so fail-fast stopped it. A diagnostic-only ring run
reached 1 mode (HQ .081) by 1200. The critic-side family is closed here.

Acquisition margin from per-update particle traces (a mode counts as covered
when a clean particle is within .21 of it):

| Run | First 8-mode checkpoint | First update with all 8 covered | Updates with all 8 covered before 1000 |
| --- | ---: | ---: | ---: |
| v10 | 1100 | 1084 | 0 |
| v8 | never (max 7) | never | 0 |
| v6 | never (max 6) | never | 0 |
| v14 | never | never | 0 |
| plain alternating constant Adam | never | never | 0 |

No variant covers all eight modes before update 1084. The actual bottleneck is
reaching the last one or two modes, not holding them. v10 scrapes the gate
by reaching its eighth mode just after the terminal window opens, rather than
holding a pass with margin.

## Spatially smoothed critic for G (v15)

G's critic calls in the two replay passes use E[D(x + sigma eps)] over 8
antithetic Gaussian draws on the data input, from a private generator seeded
by the update index. The host's RNG streams are untouched, and the D update is
unsmoothed. The width is proportional to the critic's own length scale,
sigma = alpha * std(D) / RMS ||grad_x D||, a state quantity with no cap and no
schedule. G stays at .25 and the D bound at 3.

| alpha | Warm | Cold trajectory | Cold ring | First update with all 8 covered |
| --- | --- | --- | --- | --- |
| 1 | FAIL 111/200 (min .5149; ring sigma .25-.83) | .181* | not run | — |
| .25 | **PASS 200/200 (min .9854)** | **PASS .001, 20-check suffix** | FAIL: max 7 modes; terminal 7/.77, 2/.19, 2/.14, 3/.20, 7/.94 | never |
| v10 reference | 200/200 (.9297) | PASS .00094 | 3/5 | 1084 |

\* Unplanned: the alpha=1 cold run was launched in the same command before
its warm failure was read. It has no promotion credit.

The peer run (#84) with sigma effectively fixed at a .15 cap passed the ring
(first 8-mode coverage at 600) but failed warm at 196/200. Here, a
state-proportional width with a similar ring mean (.16, varying .02-.38)
recovers warm with the best margin of any candidate but loses the ring.
Width magnitude alone does not separate the two outcomes. By the stop rule,
this knob does not recover warm without killing the ring.

## Peer stencil critic: warm repair attempt (v16 reproduction, v17)

v16 ports peer #84's ring-passing recipe verbatim (`--stencil-critic`): after
each D step, G's 2D critic scores average a centre plus ±width stencil on
each axis, with width = min(.15, .5 / sharpness) from the critic's RMS input
gradient on clean particles. G .25 and D bound 3 are unchanged. It reproduces
#84 exactly on this VM: warm 196/200 (fails 1129-1132, min HQ .8662),
trajectory PASS .00094 (18), ring PASS 8 modes, final HQ .9988.

Mechanism of the warm dip: at the matched warm state the critic is almost flat
at the particles (sharpness .07-.15). G's own-curvature ratio, measured
through the smoothed critic, falls to .16-.4, so the .25 G bound lets through
full constant-rate Adam steps (factor 1.0 at 1123-1129). Modes 3 and 4 then
bleed from ~660/715 to ~440/460 in-radius samples.

v17 keeps the width formula, cap and smoothed step direction, but measures
G's curvature ratio on the plain critic (two unsmoothed replays, 4 gradient
evaluations per update). The rule is identical on every host.

| Run | Warm | Cold trajectory | Cold ring |
| --- | --- | --- | --- |
| v16 = #84 recipe | 196/200 (.8662) | PASS .00094 | PASS, 8 modes, HQ ~.99 |
| v17 plain-critic curvature | **200/200 (.9763)** | PASS .00094 | FAIL: 7 modes, HQ ~.92 from 800 to 1200 |

The G step factor on the ring falls from mean .156 (v16) to .089 (v17). The
looseness that lets the smoothed recipe reach the eighth mode is the same
looseness that drifts at rest. Stopped by the rule: warm cannot be recovered
here without losing the ring. A warm-only switch or an HQ/mode-count
condition would reach the same outcome only by testing a different method in
the warm fork, or by reading the target, so neither was used.

## Per-particle slope weighting on the stencil recipe (v18)

The critic's slope at the particles separates travel from rest without the
target: RMS input gradient .67-1.07 during cold ring acquisition, .11-.18
once resting, .113 on the warm fork. v18 keeps the #84 recipe and scales each
generated sample's gradient into G by min(1, ||grad_x D_s(x)|| / 1), the
stencil-smoothed critic's slope at that sample. The rule is identical on
every host, and trajectory is unaffected because its critic input is not 2D.

Warm fork: **FAIL 192/200** (fails 1192-1199, min HQ .8342), worse than the
recipe's 196/200. Cold was not run (fail-fast). The weights did shrink
per-sample pushes at rest (mean .09), but G's own-curvature ratio is measured
on the same weighted field. It shrank in proportion, and the G bound relaxed
to match (mean factor .92, versus about .16 unweighted), so Adam's
normalization restored the step. A per-sample weight and a scale-free
curvature bound cancel rather than compose. One clean try; stopped.

## Post-bound slope step scale (v19)

v19 keeps the #84 recipe and scales G's applied step, after the curvature
bound, by min(1, s / 1), where s is the critic's RMS input slope at the clean
particles after D's step. Adam moments and the curvature measurement are
untouched, so neither compensates (unlike v18). Same rule on every host;
trajectory is unaffected because its critic input is not 2D.

| Gate | Result |
| --- | --- |
| Warm | **PASS 200/200** (min HQ .9678; slope scale mean .18) |
| Cold trajectory | PASS .00094, 18-check suffix |
| Cold ring | FAIL: never 8 modes (max 7). Collapses to 0 modes at 800; terminal 2/.20, 3/.27, 4/.34, 6/.58, 6/.66. Slope scale mean .65 |

This is the same warm/ring trade as v17. Damping G by a state signal that is
small at rest also slows ring acquisition enough to lose it (here the ring
also destabilizes). Stopped as agreed; no second knob.

## Recommendations

1. Put future game-update candidates in the alternating adapter
   (`alternating_curvature_scratch.py`, which also gives exact parity
   controls), or re-test simultaneous ones there. Otherwise acquisition
   failures are confounded.
2. The remaining gap is ring acquisition speed under the G bound. The most
   specific next step is a G bound that relaxes while the ring is still far
   from matched, using a signal that separates ring acquisition from ring
   rest. Neither curvature nor step coherence does; the measured critic
   advantage partly does.
3. Keep D at the rho≤2 edge bound; it removes a real critic divergence without
   hurting trajectory.

## Reproduce

```bash
python -u reports/toy100/alternating_curvature_warm_probe.py --curvature-bound .25 --bound-d --d-curvature-bound 2 --output NEW
python -u reports/toy100/alternating_curvature_probe.py --curvature-bound .25 --bound-d --d-curvature-bound 2 --output NEW
# bounds disabled (parity): --curvature-bound 1e9 ; v6: --curvature-bound .25 ; v5: add --advantage-gate .1
```

Evidence, source copies and hashes:
[continuous-evidence/alternating-curvature](continuous-evidence/alternating-curvature/).
Tests: `tests/test_alternating_curvature_scratch.py` (exact plain-Adam
parity, analytic D-then-G bound), `tests/test_cross_curvature_scratch.py`;
the full handoff suite with these ran 168 passed.
