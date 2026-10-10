## 00. Current candidate: E22 (`pkg-E22` + `overrides-E22.json`; its evidence is E19a's), 2026-09-29 — E14s plus a support test that re-draws table rows with no real support, plus a parametrisation-invariant feature scale

*The tables in this section are generated from the run directories by `analysis/build_results00.py` (template `analysis/results00.template.md`); re-run it to refresh cells marked n/a (runs that were still going when it was last built).*

**Why this exists.** The user's answer to the E14s report (2026-09-29): stay on one deterministic seed, first solve the thin rotated100 precision margin (E14s: +.0028 over the .97 limit, holdout +.0023), then revisit seeds; toy gates are advisory when the test is bad or the result is right another way. The margin is decided by *strays*: table rows 3-16 sigma from every mode (2.5-2.9% of the table at 7k, median 12 sigma, 89% inside the lattice). The critic cannot pull them back (a stray has weight 1/batch in its loss, the far field is flat, the mean critic force on a stray has no direction: cos +.01 with the direction to the nearest centre) and the row-evidence gate recognises only a third to a half of them. A per-row test against the real reservoir *in the critic's feature space* (allowed by the user's rule) flags 261-267 of the 285 rotated strays of the E14s final state (every stray beyond 6 sigma) and 0 bulk rows in nine draws (the statistical review's numbers for the shipped statistic; my first offline numbers used noisy fakes: 264/146/33 for rotated/staggered/grid).

**How it is built (each step a copy of the previous package plus one change; `diff -ruN` between neighbours shows exactly what changed):** E14s -> E15 (`birth_death_isolation`: the support test, parents drawn uniformly) -> E16 (parents = one of the k nearest unflagged rows) -> E17 (parents uniform among the unflagged rows within 2x the distance to the nearest one) -> E18 (`birth_death_feature_scale: "std"`, after codex's audit) -> E19 (rank cap k^2 on the parents, chunked stale-site check: code review) -> **E22 = E19 + a duplicate guard** (the test is off when more than Q of the reservoir's feature rows are exact copies of another row; bit-identical to E19 whenever it does not fire, `tests/test_E22.py`; every run reported for E19a is therefore E22's evidence). Side branch: E20/E21 (parents drawn with probability proportional to their own p-value, persistence of two evaluations, duplicate guard), built for the statistical review's findings; they lower the churn by about 40% and pass the natives but regress the flaky ring_shift gate (section 00.4), so they are not the candidate. The mechanism is described in README.md (M5, M6).

@@NATIVE@@

The E22 rows are confirmation runs of the duplicate-guard package: all three reproduce E19a to the printed digits (and every live check, 26-28 per task, is identical to full precision), as the CPU test predicts (`tests/test_E22.py`).

Reading (7k, last-5 worst live precision; limit .97): the margin on rotated100 goes from +.0028 (E14s) to **+.0151 (E19a .9851)** (E17a .9853, E20a .9850), staggered from .9758 to .981-.983, grid stays at .983-.985 (the support test acts only 2-8 times on grid: the grid pass does not rest on it). Centre errors stay inside the .20 limit (last-5 worst .145-.169; holdout .117-.139). The S2 margin class (REQUIREMENTS section 4: x_hat <= .16 on every task and precision >= .973 on rotated/staggered) is FULL for E19a (x_hat = sqrt(holdout^2 - .0435^2) = .121 / .121 / .109), THIN for E14s (rotated .9728 < .973). The worst mode's covariance eigenvalue ratio (limit 1.70) is 1.41-1.64 and is now the thinnest gate: it flickers above 1.70 in mid-run checks and decided E17b rotated (1.83), E17blr075 staggered (1.76), E15a grid (1.74) and, before this pass, E14s lr x1.33 grid (1.83).

### 00.1 What each variant taught (attribution; ablations are single runs, 7k)
- **E15a (uniform parents) fails 0/3 although its precision is .978-.982**: accuracy fails (rotated holdout centre .217, staggered .351, grid coverage streak 4). Mass returned to random modes shuffles mode masses (mode-count sd/mean about .10 vs .085) and, more importantly, the critic loses its confinement of the bulk: the table tester's mean block cosine at the long scale is ~0 (E14s: -.126) and turns positive at the short scale (+.064, t +4.5 at step 3448), the sign-only ladder never halves (rotated: 1250, 2000, then 6750 instead of 3500), the jitter stays 2x. Measured with the true centres (`analysis/critic_radial.py`): around a mode the median outward critic gradient is negative at every radius in E14s (-.004 at 1 sigma, -.0126 at 5, -.0148 at 6-8; 69-85% of random points feel an inward force out to 14 sigma: a funnel), weaker in E17a/E18a (-.003..-.009; 54-72% inward) and gone in E15a (~0 beyond 6 sigma, 50% of the points outward). The strays, which are fakes in the far field, are the negatives that hold the funnel up (the funnel measurement is direct; the causal reading is an interpretation).
- **E16a/E17a/E18a/E19a all pass 3/3**: returning a stray to its own neighbourhood restores the mass balance and the ladders (E17a rotated: the same four halvings as E14s). E16's parents were rim rows (47% of the 17,756 parents at >= 3 sigma), E17's ball reached the bulk on final states (92-95% bulk parents) but during the run 46% of its parents were still unflagged shell rows, and the statistical review found their clones were flagged again within 300 steps 92% of the time (6,358 of 6,911).
- **The row-evidence gate is still needed**: E16b (gate off, k-nearest parents) 3/3 but E17b (gate off, ball parents) 2/3 (rotated fails the shape limit: 1.53-1.92 in the last 10 checks vs 1.38-1.56 with the gate) and lr x.75 2/3 (staggered 1.72-1.76); E17a with the gate is 3/3 and 6/6 with lr x.75/x1.33.
- **The feature scale (M6) is what makes the toys pass** (00.4): E17a (raw features) 10/13 -> E19a (standardised features) 13/13; the support test is inert on all tiny tables (N <= 256), so the difference is the feature scale alone. Measured on the img_bars4 critic at the end of a run: the head-input features' std spans 3e-3..2.2e-1 (70x) and the 8 largest of 128 features carry 42% of the mean squared distance between real rows in the raw metric.
- **Churn**: the support test re-draws 14,876 rows (E19a) on rotated per 7k steps (E17: 15,859 of 5,100 distinct rows, one row up to 38 times); flagged rows sit 4-6 sigma (53%), 6-10 (28%), 10+ (11%), 3-4 (8%); the critic's force on rows beyond 3 sigma points outward in these runs (cos -.71), so the bulk leaks into the shell at ~2 rows/step and is recycled. E20's p-weighted parents cut this to 9,106 (5,642 distinct rows, at most 8 times).

@@S4@@

@@HORIZON@@

@@SUITE@@

@@VARIANTS@@

### 00.5 Probes on copies of the harness (the frozen harness is untouched; `harness-bigN/` and `harness-absence/` are scratch copies with edited task specs; `launch_bigN.sh`, `launch_absence.sh`)
@@PROBES@@

### 00.6 Independent reviews of E17 and codex's audit (reports were returned as messages; evidence, scripts and predictions are in `design/review_E17_stats/` and `design/review_E17_code/`; handoff `design/E17_HANDOFF.md`)
| finding (source; severity) | what was done |
|---|---|
| no blocking finding from either review; the user's rule is respected (stats) | - |
| churn is self-inflicted by the parent rule (stats; major) | E20 (p-weighted parents): -40% re-draws; not adopted because of ring_shift (00.4) |
| my offline recall was for a different statistic than the shipped one (stats; major) | corrected: rotated 261-267/285, staggered 107-126/192, grid 0/66 |
| the guard covers only flagged > 5%; a missing mass of .24-5% is re-drawn every evaluation, no persistence (stats; major) | E20/E21 add persistence of two evaluations (not adopted); the iid-reservoir assumption is stated; probe 00.5 quantifies it |
| exact copies in the reservoir break exchangeability (finite pools: P(p <= 1e-3) is 21x nominal at 1,000 distinct rows) (stats; major) | **E22: the test is off when more than Q of the feature rows are copies** |
| the distance ball is not local at z_dim >= 16 (23/90/100% of the table at 16/32/64); mass returns by catchment, not by mass (stats; major) | **E19: rank cap k^2**; the catchment bias is inherent to returning a stray to its own neighbourhood |
| ball factor 2 was picked among E15/E16/E17 on the acceptance tasks (stats; A2) | still open in E22 (E20 dropped the ball and lost ring_shift; E21 restored it) |
| the feature gauge (stats + codex 517c5e6f) | **E18: per-feature standardisation by the reference half** (recall .90-.91 in every gauge on a real critic; raw drops to .61 at 16x; whitening is valid but fragile: it amplifies low-variance directions and its pseudo-inverse floor is gauge dependent); side effect: toys 10/13 -> 13/13 |
| stale-site `cdist` is O(N x moves): 0.5 N^2 bytes at the 5% guard (code; high for N >= 100k) | **E19: chunked** |
| BH cannot flag below ~40 rows at once; inert below N = 800 (code, stats, codex a15dd2a6) | stated; on the tiny toy tables the mechanism is inert by design |
| float32 shortlist in the kNN degrades once (feature scale)/(local spread) >~ 1e3 (code; medium, shared with the ordinary path) | not fixed; standardised features keep the ratio small |
| test coverage: 23 of 57 injected mutants killed by tests/test_E15.py; the reviewer's added tests kill 57 of 57 with it | adopted (`tests/test_E17_extra.py`; `test_E22_extra.py` for the duplicate guard) |
| tie sensitivity on integer-lattice features (CPU vs GPU flags), harness `interval_moved` diagnostic ignores isolation moves (code; low) | documented |
codex's second commit after E14s (a15dd2a6, "small-batch critic-memory preflight", `reports/toy100/lrfree-search/streaming-smallbatch-toy/`): a toy with N = 16-64 rows and batch 2 shows that rolling critic-payoff reactions (cloning the lowest-payoff row into the highest) are worse than the ordinary GAN, and derives the same small-N limit of the support test; nothing there changes E22 (the mechanism is inert at those sizes) but it argues against any small-table reaction of that kind.

@@FORECAST@@

### 00.8 Not clean, and what would fix it (ranked recommendations are in section 6)
- Single deterministic seed everywhere. Rotated is now +.0151 in precision, but the shape gate (worst mode's covariance eigenvalue ratio, limit 1.70) is at 1.41-1.66 in the natives and stress cells, E19lr133 grid's centre error is .198 (limit .20), and ring_shift is flaky (below).
- ring_shift (the shift gate) shows abrupt collapse episodes in every variant (hq .9 -> .1-.2 within 50 steps, recovery over 400-600 steps: E14s at ~3600, E19a at ~3440, E21a at ~4140); a pass needs the last five checks clean, so the outcome depends on when the episode falls (E19a PASS, E21a FAIL); the cause is not identified.
- The support test acts only where the table has at least ~800 rows and only on strays beyond ~5-6 sigma (synthetic recall .80/.99/1.0 at 5/6/8 sigma with 300 strays; .07/.59/.99 with 60); 3-4 sigma rows are almost never flagged.
- iid reservoir: a component absent from the reservoir for many evaluations loses its rows (probe 00.5; the ordinary birth-death does the same); a finite pool with exact copies is guarded (off), a near-duplicate pool (augmentations) is not.
- Chosen constants: Q reused three more times (BH level, act-only-if guard, duplicate guard), the ball factor 2, persistence absent, the 1e-3 and 1e-8 numerical floors; older constants unchanged (gate window 50, hold budget, m = 4, anchor factor 2).
- The custom hosts (the 8 side experiments of the 22-check suite) are still not run: the harness refuses trainers with methods or settings it does not know.
