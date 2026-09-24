# Cross-only replay and bounded cross-only responses (PR #60 follow-up)

**Result: no survivor.** Both new candidates failed the matched warm fork, so
by fail-fast order no cold trajectory, cold mode-hold, hold or shift run was
made for them. Production defaults are unchanged; nothing here is eligible for
the shared gate (scratch adapters).

Platform: this VM reproduces the archived *qualitative* controls (scheduled
PASS, constant FAIL) but not the archived traces; with torch 2.13.0+cu126 and
2.14.0+cpu alike, trajectories diverge from the preserved evidence at update
50. All comparisons below are therefore platform-local: every row shares one
warm hash `57c10b57…`, the identity child matched the uninterrupted control's
final state hash in every run, and the constant control is 4/200 throughout.

## 1. Same-sample replay of the stock cross-only response

`cross_drift_replay.py` returns to each accepted update's base point and
replays the same data/noise at the accepted joint step, the D part only and
the G+prior part only (3 extra evaluations/update), then restores parameters,
buffers and every RNG stream bit-for-bit. A plain cross-only sibling in the
same fork has the identical final state hash (`replay_matches_plain_cross_only`).

The stock method fails here once, at update 1002 (199/200, HQ .8938), rather
than at the archived 1194–1197.

| Group (medians) | alpha | G own-curvature / step | G cross / step | particle RMS motion | particles outside HQ | critic pull toward own mode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Warm failing update 1002 | .5 | 2.02 | .065 | .213 | 0 → 1 | +.96 |
| Warm transient 1001–1010 | .5 | 2.06 | .104 | .166 | 0 | +.85 |
| Warm late passing 1101–1200 | 1 | .28 | .59 | .017 | 0 | −.06 |
| Cold mode-hold acquisition 51–400 | .0078 | .10 | .036 | .0094 | 12 | −.07 |

At the live failure, the target is matched (occupancy [1,1,2,2,2,2,1,1], no
mode switches) and the critic points particles toward their correct centres.
G's own field change caused by its own step is twice the step, i.e. the
effective preconditioned step times curvature is at the gradient-descent
stability edge. The cross-only solve and its cross residual (.39, accepted)
cannot see this block; the full joint residual is 1.89. Cold acquisition is a
different failure: own curvature is small, but backtracking holds alpha at
.008–.03, particles move ~.01/update and the cold mode-hold diagnostic
collapses to one mode (12/12 particles, 0/24 checks; diagnostic only, no
credit).

## 2. Candidates tested (warm fork, updates 1001–1200 every update)

Both keep the cross-only CGD solve on the Adam metric (≤8 GMRES directions,
linear residual ≤ .1, correction bound 2), move Adam moments once per update
and **drop the nonlinear-residual backtracking** that starved acquisition.

* **v1, own-curvature bound**: per player, measure
  rho = ||alpha (F(u_p only) − F0)_p|| / ||u_p|| with one same-sample
  evaluation and scale that player's step by min(1, 1/rho).
* **v2 = v1 + no amplification**: each player's step is also capped at its
  own explicit Adam step length, so the cross response may redirect or
  shorten a step but never lengthen it.

| Method | Warm checks | Failing updates | Min HQ | Grad evals / player / update | Mean / max clean output motion | Bounds active (G) |
| --- | ---: | --- | ---: | ---: | --- | --- |
| Identity (scheduled) | 200/200 | — | .990 | 1 | — | — |
| Constant Adam .00425 | 4/200 | nearly all | 0 | 1 | — | — |
| Stock cross-only | 199/200 | 1002 | .8938 | ~13.6 | .031 / .214 | — |
| v1 own-curvature bound | 199/200 | 1085 | .8013 | 7.77 | .024 / .189 | curvature 52/200 |
| v2 + no amplification | 198/200 | 1134–1135 | .8884 | 7.78 | .019 / .159 | curvature 51, amplification 115 |

All accepted alpha were ≥.5 (v2: 1.0 always). Moment updates: 1 per update.

**v1 failure (1085):** a different mechanism. rho_G was .33 (bound inactive),
but the cross solve made G's step 15.5× its explicit Adam step: at a matched
target G's own field is small, so its anticipation of D's move dominates. The
clean output jumped .189 RMS in one update.

**v2 failure (1134–1135):** no bound was active. rho_G < 1, every G step was at
most its explicit Adam step, and per-update motion was only .01–.03
(the scheduled recipe's late average is .0197). Mode 2's HQ count fell from
~660 to 211 over about ten updates and recovered: one of its two particles
walked out past the .21 radius and came back. In the stock replay, outward-
moving particles had outward critic pull 75% of the time in doubly occupied
modes versus 56% in singly occupied ones, so the 12-particle/8-mode mismatch
contributes but does not explain the walk alone.

## What this adds beyond the rejected table

1. The stock cross-only live failure is own-player G curvature overshoot at a
   matched target, not cross-player rotation or target mismatch; its cold
   failure is step-factor starvation from backtracking.
2. Per-update trust regions (own curvature, cross amplification, and the
   earlier .05 output cap) each remove the large single-update jumps, but the
   next failure is a slow multi-update walk whose individual steps already
   look like the scheduled recipe's. Per-update step-size limits alone do not
   provide "rest at a matched target".
3. The remaining gap is magnitude at rest. At constant rate, Adam's step stays
   near lr even when the matched-target field is small, and bounds relative to
   that explicit step inherit this. A next hypothesis should make step length
   scale with the field relative to a reference magnitude while keeping
   acquisition speed. For example, a metric that does not renormalize the
   field to unit scale at rest, rather than more per-update caps. Plain
   eps/AMSGrad/beta2 changes were already rejected, so any such variant needs
   the matched shift test to prove it can still wake.

## Reproduce

```bash
python -u reports/toy100/cross_competitive_warm_probe.py --output /tmp/x-stock
python -u reports/toy100/cross_drift_replay.py warm --output /tmp/x-replay-warm
python -u reports/toy100/cross_drift_replay.py cold --output /tmp/x-replay-cold
python -u reports/toy100/cross_curvature_warm_probe.py --output /tmp/x-v1
python -u reports/toy100/cross_curvature_warm_probe.py --amplification-bound 1 --output /tmp/x-v2
python reports/toy100/continuous-evidence/cross-curvature/source/replay_analyze.py  # expects /tmp/cgd paths
```

The cold driver `cross_curvature_probe.py` (trajectory, then mode-hold, stop at
first failure) is prepared but was deliberately not run. Evidence and source
copies are in [continuous-evidence/cross-curvature](continuous-evidence/cross-curvature/).
Tests: `tests/test_cross_drift_replay.py`, `tests/test_cross_curvature_scratch.py`
(exact bilinear/own-curvature/amplification cases, zero field, RNG and
moment accounting); the full handoff suite plus these ran 159 passed.
