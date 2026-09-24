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
| none (v6) | PASS .00099 | .9995 at 1000, then 0 modes by 1100 (critic divergence) | FAIL |
| 2 (v8) | PASS .00094, 20 | .43 / .52 / .83 / .92 / .92, 7 modes | FAIL |
| **3 (v10)** | **PASS .00094, 18** | **.916 / .843 / 1.0 / .995 / 1.0, 8 modes** | **FAIL, 3/5 terminal checks** |
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


## Rho-ratio G cap (measured, ring still open)

Controller on the v10 adapter (alternating, D bound 3, nominal G cap .25). The G cap interpolates from `--ratio-loosen` when `rho_G/rho_D <= --acq-ratio` to `--ratio-tighten` when `rho_G/rho_D >= --rest-ratio`. G never goes to .5. Same seed 0. No 2400 run, because the ring gate failed.

| Setting | Warm | Cold trajectory | Cold ring terminal |
| --- | --- | --- | --- |
| loosen .35, tighten .15, knots 1.5/4 | 200/200, min HQ .9272 | PASS MSE .000957, suffix 19 | FAIL 7 modes. HQ .910/1/1/.998/1 from 1000-1200. The rest cap held quality; one mode never arrived |
| loosen .25, tighten .12, knots 2.5/4.5 | 200/200, min HQ .9668 | PASS MSE .000984, suffix 18 | FAIL 5 modes, final HQ .468. Tightening through the terminal window stalled coverage and then dropped it |
| loosen .30, tighten .18, knots 1.2/5 | not run | FAIL MSE .253 | not run |

v10 (fixed G cap .25, D bound 3) remains the ring near-miss: 8 modes, and the only terminal miss is update 1050 at HQ .843. Moving the G cap with `rho_G/rho_D` either spends that eighth mode or falls into the trajectory basin. The ratio separates acquisition from rest in bin medians, but using it as a step-size controller did not clear the five terminal checks. That family is closed.

## Two non-ratio attempts (neither beats v10)

Both keep D bound 3 and the nominal G cap .25, and neither changes cold trajectory (MSE .000943, suffix 18) or the warm fork (200/200, min HQ .9297). v10's archived ring is still ahead: modes/HQ at 1000–1200 are 7/.916, 7/.843, 8/1, 8/.995, 8/1.

| Mechanism | Cold ring 1000–1200 |
| --- | --- |
| Loosen G to .40 until 8 modes have been occupied, then latch back to .25 | Never latched. Terminal modes/HQ: 2/.108, 7/.911, 7/.909, 7/.910, 7/1. Mode 1 stays empty; update 900 collapses to 1 mode |
| Ring-only G cap .35 for the first 300 updates, then .25 | Boost ran 300 steps. Terminal modes/HQ: 5/.506, 6/.821, 6/.821, 6/.811, 6/.820 |

No 2400 hold. v10 remains the ring near-miss.
