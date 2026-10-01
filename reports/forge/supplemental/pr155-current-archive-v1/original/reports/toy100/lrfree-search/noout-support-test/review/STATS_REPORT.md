# Independent statistical review of the support test (pkg-E17), returned as a message on 2026-09-29

Reviewer: a general-purpose subagent, CPU only; scripts and outputs are in `stats/` next to this file (`sim*.py`, `out/*.txt`). It also received, mid-run, codex's feature-gauge finding and my native-critic gauge test and answered them (section GAUGE). The report is reproduced as received (HTML escapes restored).

Note on process: the agent finished its report but left a background shell loop of its own running (`until [ -s out/sim5b_drift.txt ] && grep -q "^   9 " out/sim5b_drift.txt; do sleep 15; done`, a condition that could never become true), which kept it on the user's agent list for hours; it was stopped by hand afterwards (nothing else of the agent was running).

---

Everything is under `/ml2/hypergan/gan-attempts/noout-20260928/design/review_E17_stats/` (scripts there, outputs in `out/`). Simulations call the shipped `_isolated`, or a KD replica proven identical to it in `check_kd.txt`.

Two constraint notes. I touched no one else's process. My first `test_E15.py` run reached CUDA by accident (~1 GB for 16 s, exited); I reran it CPU-only. A few short single-thread analyses briefly overlapped my 8 threads.

**Blocking: none.** The rule is respected (Q7, `out/audit7_grep.txt`):
- Raw reals and raw generator outputs are only stored (`birth_death.py` 125-140) or fed to the critic (`_features` 179-207, call at 311).
- `_is_raw` (171-176) is one exact-equality structural check.
- `_isolated` sees head-input features only; `_isolation_pick` sees z only.
- sigma_out enters the fake pool F, not the test.
- Real-only rho, b and `R.mean` are feature-space statistics (grey area; I read them as network internals).

No handoff claim is false; several are overclaimed.

## MAJOR
1. **Churn is self-inflicted by the parent rule (Q6).** Evidence: `churn_leak.txt`, `parent_shares_by_phase.txt` (distances via final G).
   - Of 15,859 re-draws, 0 flagged rows lie within 2σ and 21 (0.13%) within 2-3σ: no false-positive problem.
   - A child of a core parent (<2σ) is re-flagged within 300 steps 4.8% of the time (278/5,790): a small real bulk leak.
   - But 46% of parents (55% in steps ≤3000, 27% after 4000) are unflagged shell rows ≥3σ. Their children are re-flagged within 300 steps 92% of the time (6,358/6,911; median 30-80 steps).
   - On E14s final tables the ball does pick 91-95% bulk parents (`parent_rule_real.txt`), so "reaches the bulk" is state dependent.
   - The ".0004 stray fraction" is the served (averaged) table (4 rows >3σ). The training table keeps 77 (E14s: 285, so 3.7× fewer, not 35×), none flagged offline.
   - On grid100 the mechanism acts only 2-3 times per run (final-state counters), so "3/3" rests on rotated/staggered.
2. **Offline recall in the handoff is not the shipped statistic.** `real_state_check.txt` (shipped code, clean centres, real critic, 3 reservoirs each): strays flagged rotated 261-267/285, staggered 107-126/192, grid 0/66 (handoff: 264, 146, 33). There were 0 false flags in 9 draws (the noisy-scored variant gives 5-23 per draw). Synthetic power (`sim3_local.txt`, `sim2_fdr.txt`): the threshold is ~5-6σ (5-8 local radii). Recall at 5/6/8σ is .80/.99/1.0 with 300 strays and .07/.59/.99 with 60. 4σ needs ≥1000 strays (.30; .04 at 300).
3. **The guard does not cover the window that matters (Q5).** Evidence: `sim5_missing.txt`, `sim5b_drift.txt`, `sim3b_rare.txt`.
   - Flagged fraction equals missing mass mu (recall of absent rows 1.000) for mu ≥ 0.24%. The guard blocks only mu > 5%.
   - For mu in [0.24%, 5%], every row of the absent classes (49-919) is re-drawn at every evaluation. There is no persistence requirement (E14 needs n≥2).
   - 1-3% of modes shifted ≥6σ between reservoir and table: 94-100% of their rows are re-drawn, 0 false flags elsewhere.
   - Sorted 10-class stream: mu .50-.60, guard never acts (0/10).
   - Rare components, with 300 strays: 4- and 10-row components lose 29-32% and 4-6% of rows per evaluation (20 rows: 0). With 1000 strays: 95%, 50-56%, 3%.
4. **Duplicates break exchangeability (Q4).** `sim4_dup.txt`, reservoir drawn from M unique points:
   - P(p≤1e-3)/1e-3 = 21 (M=1000), 2.7 (2000), 1.8-2.4 (3000-10000).
   - At M=1000, 5,532 fresh rows are flagged per evaluation (72-16,115) and the guard acts in 17% of evaluations.
   - M≤300: everything is flagged and the guard skips (graceful).
   - Twins land in both parity halves; the floor=1.0 fallback is dimensional. Fix: group-split or dedupe by feature identity.
5. **Ball is not local at large z_dim; mass bias (Q6).** `sim6_action.txt`:
   - The ball holds 23/90/100% of the table at z_dim 16/32/64 (iid z): uniform parents = E15 (failed 0/3).
   - At z_dim 2 with 1:10 masses, inflow/mass-share is .31-3.4, corr -.72 with mass (uniform rule .80-1.27). Mass returns by catchment, not by mass.
   - Between a 10× heavier and a lighter mode, the heavy one takes 91-99% of strays for x in [.4,.6].
6. **Feature gauge:** see below.

## CONFIRMED / MINOR
- **Null (Q1).** Clean-centre p-values are super-uniform (grid100 P(p≤.01)=0, KS .33; real critic median p .74 vs .50 noisy, P(p≤1e-3) 0 vs .0012). Noisy-scored and iid rows are uniform (.94-1.05).
  - BH false flags: 0/700 evaluations (95% upper bound .53%; expected falsely re-drawn rows 0.000 per evaluation, ≤0.2 at 95%) in grid100 and the unequal/mixed-width mixture.
  - 0/150 at 8-D and 32-D, 0/40 at 128-D (`sim1_*.txt`).
- **FDR (Q2).** .028-.054 (SE .0004-.005) for 39-4000 far strays (`sim2_fdr_se.txt`), consistent with q.
  - PRDS: Benjamini-Yekutieli 2001; conformal p-values with a shared calibration set, Bates et al. 2023. Both assume iid rows.
  - Comonotone duplicate groups give P(any flag) .033-.050 for ≥50 rows per group, flagging 50-1,056 rows at once (`sim2b_lumpy.txt`). It is safe here only because clean atoms are central.
  - Under 40 strays: no flags (0/150 fresh null; one event with 20 strays flagged 25 legit rows). At 39 strays, legit tail rows complete the quota in 56% of evaluations.
- **Local scale (Q3).** Marginally exact, conditionally tail-heavy.
  - Rows at r≥4σ have p≤1e-3 50% of the time (×498); 3-4σ ×65.
  - With 300 strays BH removes 15.4 legit rows per evaluation (FDR .048), all ≥2σ.
  - Core-halo (density ratio 1000): halo within .03 of the core ×48. Thin-ribbon ends ×17; ring |dr|≥3 ×246. The spiral's 8× density gradient shows no bias along the curve.
  - Tail trimming matters only when rows carry the spread (sigma_out=0).
- **High D (Q4).**
  - 0 false flags at 2/8/32/128-D.
  - Recall .5 needs ~3.5/2.3/1.7/1.3 noise radii; null CV .30/.14/.078/.055.
  - Nuisance noise ≥2σ in 126 unused dims makes the test blind up to 12σ, with no flags (`sim4_highdim_*.txt`, `sim4_manifold.txt`).
- **Constants (Q8), flag:**
  - Ball factor 2, picked among E15/E16/E17 on the three acceptance tasks.
  - Q reused as the guard (5% of N per ~10 steps). It sets the onset: first action at step 1100 with 997≤1000 flagged.
  - Floor fallback 1.0 (dimensional).
  - Derived/structural: k, the 50/50 split (2/Q = 40-row floor for any N; inert for N<800), the 1e-3 floor, the lower median.
- **Tests.** `test_E15.py` passes CPU-only (`test_E15_on_E17_cpu_only.txt`) but never asserts the ball semantics (my `test_ball_property.py`: 8/8). It tolerates 59 legit flags (BH implies ~15), uses one draw, and its duplicates check asserts only shape.

## GAUGE (`sim7_gauge_d*.txt`, `sim7b_exactness.txt`, `spectrum_check.txt`)
(a) 0 false flags in every variant and gauge (iid and clean rows). Displacement for recall .5 with 300 strays, in noise radii (2-D: σ), at natural / ln4 / ln16 gauge:

| features | raw | diagonal std (identical in every gauge) | whitening |
|---|---|---|---|
| d=8 | 2.4 / 2.5 / 3.9 | 2.5 | 2.4 |
| d=32 | 1.8 / 2.6 / 4.1 | 2.9 | 1.8 |
| d=128 | 1.3 / 2.3 / 4.0 | 2.6 | 1.3-1.7 |
| 2-D manifold in 128-D | 4.7 / 5.1 / 6.4 | 6.8 | 8.0 |

(b) The test is exact iff the statistic is a function of R1 only. P(p≤t)/t at t=.001/.01/.05:

| statistic estimated from | std | whitening |
|---|---|---|
| R1 only | .93/.98/1.01 | .90/1.02/1.01 |
| R1+R2 | ~1 | 1.36/1.74/1.43 |
| R2 only | ~1 | 1.58/2.30/1.77 |

(n/h=8; diagonal std stays ~1 in every case.) E17's `R.mean(0)` centring is harmless.
(c) Whitening is valid but fragile:
- It amplifies low-variance directions (real critic: 95/128 eigenvalues >1e-4·max, participation ratio 6.9; synthetic 2-D manifold: 32/128).
- The pseudo-inverse floor makes it gauge dependent (103-110 of ~250 dims kept at ln16).
- It costs h³ and is noisy when h > n/10.

(d) Default: per-feature standardisation by the R1 std (`R[0::2]`, floor for dead units). Apply it once in `_features` so ordinary birth-death gets it too and concatenated heads are equalised. Not full whitening.

## NEXT 3 EXPERIMENTS
1. **Finite pool:** native100 with reals from M=1000/2000/5000 unique points (shuffled epochs). Expect flags on fresh rows and precision loss at M≤2000.
2. **Transient absence:** withhold 3 (then 10) modes for one reservoir turnover at step 3000. Expect E17 to erase ~600 rows and need a persistence rule, while E14s recovers.
3. **Unequal masses and high z_dim:** masses 1:10 with 0.02-0.1% modes, and z_dim 32. Expect the ball to be about uniform, rare modes lost while >300 strays exist, and worse mass TV than E14s.

---

What was done about it (this archive): major 1 -> E20 (p-weighted parents; churn -40%) not adopted because ring_shift fails; major 2 -> corrected in RESULTS.md; major 3 -> persistence tried in E20/E21 (not adopted), the iid-reservoir assumption stated, probe run (RESULTS 00.5: E14s never recovers a component withheld for 100 steps, E22 does); major 4 -> duplicate guard in E22; major 5 -> E19 rank cap; gauge -> E18 per-feature standardisation (diagonal std, as recommended). The three suggested experiments: the transient-absence one was run in a simpler form (`harness-absence`, 100 steps, the 2% component); the other two are in the recommended list (RESULTS section 6).
