# noout-20260928: a portable native100 candidate without any data-space statistic (7,000 updates, QR init, noisy scoring; single deterministic seed)

The user's rule (2026-09-28, `REQUIREMENTS.md` section 0.0): the latent z, the particles, the generator, the discriminator and the trainer may be used and changed; **data space may not be used** (no distances, kNN/density/support statistics, scales or geometry
computed on raw data or raw generator outputs); fakes-versus-reals comparisons in the critic's feature space are allowed; an averaged model of record is allowed; the goal is one config that ports to other datasets and data types; native 100-Gaussian gates count more than the other toys.
Base for every diff: `seqtest-20260928/pkg-seqC` (sha256 70e7b5f5...). Columns: `last5` = worst of the last five 250-step terminal checks; `hold` = the 100k holdout; x_hat = sqrt(hold^2 - .0435^2).

<!-- sec00:begin -->
## 00. Current candidate: E22 (`pkg-E22` + `overrides-E22.json`; its evidence is E19a's), 2026-09-29 — E14s plus a support test that re-draws table rows with no real support, plus a parametrisation-invariant feature scale

*The tables in this section are generated from the run directories by `analysis/build_results00.py` (template `analysis/results00.template.md`); re-run it to refresh cells marked n/a (runs that were still going when it was last built).*

**Why this exists.** The user's answer to the E14s report (2026-09-29): stay on one deterministic seed, first solve the thin rotated100 precision margin (E14s: +.0028 over the .97 limit, holdout +.0023), then revisit seeds; toy gates are advisory when the test is bad or the result is right another way. The margin is decided by *strays*: table rows 3-16 sigma from every mode (2.5-2.9% of the table at 7k, median 12 sigma, 89% inside the lattice). The critic cannot pull them back (a stray has weight 1/batch in its loss, the far field is flat, the mean critic force on a stray has no direction: cos +.01 with the direction to the nearest centre) and the row-evidence gate recognises only a third to a half of them. A per-row test against the real reservoir *in the critic's feature space* (allowed by the user's rule) flags 261-267 of the 285 rotated strays of the E14s final state (every stray beyond 6 sigma) and 0 bulk rows in nine draws (the statistical review's numbers for the shipped statistic; my first offline numbers used noisy fakes: 264/146/33 for rotated/staggered/grid).

**How it is built (each step a copy of the previous package plus one change; `diff -ruN` between neighbours shows exactly what changed):** E14s -> E15 (`birth_death_isolation`: the support test, parents drawn uniformly) -> E16 (parents = one of the k nearest unflagged rows) -> E17 (parents uniform among the unflagged rows within 2x the distance to the nearest one) -> E18 (`birth_death_feature_scale: "std"`, after codex's audit) -> E19 (rank cap k^2 on the parents, chunked stale-site check: code review) -> **E22 = E19 + a duplicate guard** (the test is off when more than Q of the reservoir's feature rows are exact copies of another row; bit-identical to E19 whenever it does not fire, `tests/test_E22.py`; every run reported for E19a is therefore E22's evidence). Side branch: E20/E21 (parents drawn with probability proportional to their own p-value, persistence of two evaluations, duplicate guard), built for the statistical review's findings; they lower the churn by about 40% and pass the natives but regress the flaky ring_shift gate (section 00.4), so they are not the candidate. The mechanism is described in README.md (M5, M6).

### 00.0 Native gates, 7k (single deterministic seed per cell; cell = verdict, last-5 worst live precision, last-5 worst centre RMS in sigma, (holdout precision / holdout centre), worst mode covariance eigenvalue ratio of the last 5 checks (limit 1.70); frozen limits: live precision >= .97, mode masses in [.005, .02], eigenvalue ratios in [.4, 1.7], radial median ratios in [.65, 1.4], mass TV <= .10; accuracy on the final five 20k clouds and the 100k holdout: mass TV <= .06, centre RMS <= .20, |cov trace bias| <= .10, radial KS <= .04)
| run | grid100 | rotated100 | staggered100 | live |
|---|---|---|---|---:|
| **E22** (confirmation of E19a) | **P** prec 0.9834 ctr 0.157 (hold 0.9851 / 0.129) eig 1.64 | **P** prec 0.9851 ctr 0.152 (hold 0.9863 / 0.129) eig 1.41 | **P** prec 0.9822 ctr 0.150 (hold 0.9835 / 0.117) eig 1.58 | 3/3 |
| E19a (rank cap; the evidence base of E22) | **P** prec 0.9834 ctr 0.157 (hold 0.9851 / 0.129) eig 1.64 | **P** prec 0.9851 ctr 0.152 (hold 0.9863 / 0.129) eig 1.41 | **P** prec 0.9822 ctr 0.150 (hold 0.9835 / 0.117) eig 1.58 | 3/3 |
| E20a (p-weighted parents, persistence, duplicate guard) | **P** prec 0.9829 ctr 0.145 (hold 0.9826 / 0.120) eig 1.46 | **P** prec 0.9850 ctr 0.146 (hold 0.9854 / 0.118) eig 1.46 | **P** prec 0.9809 ctr 0.169 (hold 0.9823 / 0.139) eig 1.61 | 3/3 |
| E21a (E20 + distance limit on parents) | **P** prec 0.9849 ctr 0.142 (hold 0.9837 / 0.106) eig 1.45 | **P** prec 0.9847 ctr 0.142 (hold 0.9853 / 0.108) eig 1.46 | **P** prec 0.9810 ctr 0.156 (hold 0.9819 / 0.134) eig 1.41 | 3/3 |
| E18a (+ feature scale) | **P** prec 0.9847 ctr 0.155 (hold 0.9853 / 0.126) eig 1.53 | **P** prec 0.9848 ctr 0.156 (hold 0.9854 / 0.109) eig 1.45 | **P** prec 0.9827 ctr 0.155 (hold 0.9831 / 0.137) eig 1.65 | 3/3 |
| E17a (ball parents, raw features) | **P** prec 0.9822 ctr 0.149 (hold 0.9852 / 0.118) eig 1.52 | **P** prec 0.9853 ctr 0.137 (hold 0.9867 / 0.097) eig 1.56 | **P** prec 0.9808 ctr 0.161 (hold 0.9825 / 0.123) eig 1.53 | 3/3 |
| E17b = E17a without the row-evidence gate | **P** prec 0.9839 ctr 0.152 (hold 0.9847 / 0.124) eig 1.45 | **F** prec 0.9808 ctr 0.161 (hold 0.9835 / 0.120) eig 1.83 | **P** prec 0.9799 ctr 0.152 (hold 0.9808 / 0.112) eig 1.42 | 2/3 |
| E16a (k-nearest parents) | **P** prec 0.9847 ctr 0.174 (hold 0.9843 / 0.142) eig 1.54 | **P** prec 0.9811 ctr 0.141 (hold 0.9819 / 0.106) eig 1.62 | **P** prec 0.9812 ctr 0.161 (hold 0.9830 / 0.127) eig 1.47 | 3/3 |
| E16b = E16a without the gate | **P** prec 0.9839 ctr 0.176 (hold 0.9847 / 0.143) eig 1.45 | **P** prec 0.9788 ctr 0.147 (hold 0.9816 / 0.104) eig 1.54 | **P** prec 0.9791 ctr 0.160 (hold 0.9806 / 0.124) eig 1.44 | 3/3 |
| E15a (uniform parents) | **F** prec 0.9818 ctr 0.162 (hold 0.9827 / 0.145) eig 1.74 | **F** prec 0.9799 ctr 0.245 (hold 0.9810 / 0.217) eig 1.58 | **F** prec 0.9780 ctr 0.385 (hold 0.9804 / 0.351) eig 1.48 | 0/3 |
| E14s (before this pass) | **P** prec 0.9836 ctr 0.138 (hold 0.9842 / 0.106) eig 1.58 | **P** prec 0.9728 ctr 0.130 (hold 0.9723 / 0.091) eig 1.30 | **P** prec 0.9758 ctr 0.155 (hold 0.9777 / 0.128) eig 1.56 | 3/3 |

The E22 rows are confirmation runs of the duplicate-guard package: all three reproduce E19a to the printed digits (and every live check, 26-28 per task, is identical to full precision), as the CPU test predicts (`tests/test_E22.py`).

Reading (7k, last-5 worst live precision; limit .97): the margin on rotated100 goes from +.0028 (E14s) to **+.0151 (E19a .9851)** (E17a .9853, E20a .9850), staggered from .9758 to .981-.983, grid stays at .983-.985 (the support test acts only 2-8 times on grid: the grid pass does not rest on it). Centre errors stay inside the .20 limit (last-5 worst .145-.169; holdout .117-.139). The S2 margin class (REQUIREMENTS section 4: x_hat <= .16 on every task and precision >= .973 on rotated/staggered) is FULL for E19a (x_hat = sqrt(holdout^2 - .0435^2) = .121 / .121 / .109), THIN for E14s (rotated .9728 < .973). The worst mode's covariance eigenvalue ratio (limit 1.70) is 1.41-1.64 and is now the thinnest gate: it flickers above 1.70 in mid-run checks and decided E17b rotated (1.83), E17blr075 staggered (1.76), E15a grid (1.74) and, before this pass, E14s lr x1.33 grid (1.83).

### 00.1 What each variant taught (attribution; ablations are single runs, 7k)
- **E15a (uniform parents) fails 0/3 although its precision is .978-.982**: accuracy fails (rotated holdout centre .217, staggered .351, grid coverage streak 4). Mass returned to random modes shuffles mode masses (mode-count sd/mean about .10 vs .085) and, more importantly, the critic loses its confinement of the bulk: the table tester's mean block cosine at the long scale is ~0 (E14s: -.126) and turns positive at the short scale (+.064, t +4.5 at step 3448), the sign-only ladder never halves (rotated: 1250, 2000, then 6750 instead of 3500), the jitter stays 2x. Measured with the true centres (`analysis/critic_radial.py`): around a mode the median outward critic gradient is negative at every radius in E14s (-.004 at 1 sigma, -.0126 at 5, -.0148 at 6-8; 69-85% of random points feel an inward force out to 14 sigma: a funnel), weaker in E17a/E18a (-.003..-.009; 54-72% inward) and gone in E15a (~0 beyond 6 sigma, 50% of the points outward). The strays, which are fakes in the far field, are the negatives that hold the funnel up (the funnel measurement is direct; the causal reading is an interpretation).
- **E16a/E17a/E18a/E19a all pass 3/3**: returning a stray to its own neighbourhood restores the mass balance and the ladders (E17a rotated: the same four halvings as E14s). E16's parents were rim rows (47% of the 17,756 parents at >= 3 sigma), E17's ball reached the bulk on final states (92-95% bulk parents) but during the run 46% of its parents were still unflagged shell rows, and the statistical review found their clones were flagged again within 300 steps 92% of the time (6,358 of 6,911).
- **The row-evidence gate is still needed**: E16b (gate off, k-nearest parents) 3/3 but E17b (gate off, ball parents) 2/3 (rotated fails the shape limit: 1.53-1.92 in the last 10 checks vs 1.38-1.56 with the gate) and lr x.75 2/3 (staggered 1.72-1.76); E17a with the gate is 3/3 and 6/6 with lr x.75/x1.33.
- **The feature scale (M6) is what makes the toys pass** (00.4): E17a (raw features) 10/13 -> E19a (standardised features) 13/13; the support test is inert on all tiny tables (N <= 256), so the difference is the feature scale alone. Measured on the img_bars4 critic at the end of a run: the head-input features' std spans 3e-3..2.2e-1 (70x) and the 8 largest of 128 features carry 42% of the mean squared distance between real rows in the raw metric.
- **Churn**: the support test re-draws 14,876 rows (E19a) on rotated per 7k steps (E17: 15,859 of 5,100 distinct rows, one row up to 38 times); flagged rows sit 4-6 sigma (53%), 6-10 (28%), 10+ (11%), 3-4 (8%); the critic's force on rows beyond 3 sigma points outward in these runs (cos -.71), so the bulk leaks into the shell at ~2 rows/step and is recycled. E20's p-weighted parents cut this to 9,106 (5,642 distinct rows, at most 8 times).

### 00.2 Robustness to the base learning rate (S4), same seed, only `overrides["lr"]` changes (x.75 = .0031875, x1.33 = .0056525; the E14s rows are the E13s runs, byte-identical)
| run | grid100 | rotated100 | staggered100 | live |
|---|---|---|---|---:|
| E14s lr x.75 | **P** prec 0.9843 ctr 0.166 (hold 0.9844 / 0.136) eig 1.49 | **P** prec 0.9729 ctr 0.122 (hold 0.9742 / 0.097) eig 1.46 | **P** prec 0.9766 ctr 0.147 (hold 0.9764 / 0.120) eig 1.38 | 3/3 |
| E17a lr x.75 | **P** prec 0.9843 ctr 0.142 (hold 0.9858 / 0.114) eig 1.39 | **P** prec 0.9855 ctr 0.123 (hold 0.9862 / 0.102) eig 1.49 | **P** prec 0.9807 ctr 0.156 (hold 0.9831 / 0.119) eig 1.67 | 3/3 |
| **E19a lr x.75** | **P** prec 0.9832 ctr 0.151 (hold 0.9836 / 0.112) eig 1.55 | **P** prec 0.9833 ctr 0.148 (hold 0.9844 / 0.087) eig 1.47 | **P** prec 0.9822 ctr 0.144 (hold 0.9851 / 0.100) eig 1.63 | 3/3 |
| E20a lr x.75 | **P** prec 0.9840 ctr 0.149 (hold 0.9849 / 0.124) eig 1.51 | **P** prec 0.9849 ctr 0.143 (hold 0.9846 / 0.103) eig 1.38 | **P** prec 0.9851 ctr 0.151 (hold 0.9857 / 0.109) eig 1.37 | 3/3 |
| E17b (no gate) lr x.75 | **P** prec 0.9851 ctr 0.127 (hold 0.9861 / 0.103) eig 1.56 | **P** prec 0.9788 ctr 0.129 (hold 0.9811 / 0.096) eig 1.61 | **F** prec 0.9806 ctr 0.148 (hold 0.9820 / 0.118) eig 1.76 | 2/3 |
| E14s lr x1.33 | **F** prec 0.9821 ctr 0.154 (hold 0.9830 / 0.124) eig 1.83 | **P** prec 0.9739 ctr 0.140 (hold 0.9756 / 0.118) eig 1.36 | **P** prec 0.9770 ctr 0.145 (hold 0.9779 / 0.110) eig 1.44 | 2/3 |
| E17a lr x1.33 | **P** prec 0.9813 ctr 0.162 (hold 0.9822 / 0.130) eig 1.47 | **P** prec 0.9846 ctr 0.171 (hold 0.9854 / 0.142) eig 1.65 | **P** prec 0.9837 ctr 0.142 (hold 0.9844 / 0.124) eig 1.54 | 3/3 |
| **E19a lr x1.33** | **P** prec 0.9827 ctr 0.198 (hold 0.9842 / 0.159) eig 1.57 | **P** prec 0.9860 ctr 0.169 (hold 0.9867 / 0.122) eig 1.45 | **P** prec 0.9794 ctr 0.168 (hold 0.9821 / 0.122) eig 1.48 | 3/3 |
| E20a lr x1.33 | **P** prec 0.9815 ctr 0.158 (hold 0.9818 / 0.134) eig 1.66 | **P** prec 0.9862 ctr 0.161 (hold 0.9859 / 0.128) eig 1.42 | **P** prec 0.9829 ctr 0.161 (hold 0.9827 / 0.114) eig 1.48 | 3/3 |
E14s 5/6 (x1.33 grid fails the shape limit, 1.83); E17a, E19a, E20a 6/6; E17b (no gate) 5/6. The thinnest stress cells: E19a lr x1.33 grid (centre .198 against .20), E20a lr x1.33 grid (eigenvalue ratio 1.66), E17a lr x.75 staggered (1.67).


### 00.3 Sustainability (the recipe carries no horizon: the 14k run is a prefix of the 28k run)
| run | grid100 | rotated100 | staggered100 | live |
|---|---|---|---|---:|
| E14s 14k | **P** prec 0.9836 ctr 0.147 (hold 0.9838 / 0.123) eig 1.57 | **P** prec 0.9789 ctr 0.111 (hold 0.9792 / 0.071) eig 1.27 | **P** prec 0.9778 ctr 0.148 (hold 0.9793 / 0.119) eig 1.35 | 3/3 |
| E14s 28k | **P** prec 0.9845 ctr 0.168 (hold 0.9843 / 0.141) eig 1.45 | **P** prec 0.9747 ctr 0.116 (hold 0.9738 / 0.082) eig 1.36 | **P** prec 0.9775 ctr 0.175 (hold 0.9784 / 0.143) eig 1.31 | 3/3 |
| **E19a 14k** | **P** prec 0.9843 ctr 0.134 (hold 0.9854 / 0.109) eig 1.38 | **P** prec 0.9872 ctr 0.114 (hold 0.9871 / 0.076) eig 1.49 | **P** prec 0.9867 ctr 0.149 (hold 0.9860 / 0.124) eig 1.29 | 3/3 |
| **E19a 28k** | **P** prec 0.9832 ctr 0.138 (hold 0.9830 / 0.117) eig 1.38 | **P** prec 0.9888 ctr 0.113 (hold 0.9874 / 0.070) eig 1.31 | **P** prec 0.9866 ctr 0.127 (hold 0.9857 / 0.088) eig 1.34 | 3/3 |
| E20a 14k | **P** prec 0.9850 ctr 0.132 (hold 0.9849 / 0.115) eig 1.50 | **P** prec 0.9871 ctr 0.120 (hold 0.9879 / 0.077) eig 1.40 | **P** prec 0.9845 ctr 0.181 (hold 0.9848 / 0.150) eig 1.48 | 3/3 |
E14s rotated precision drifted .9728 (7k) -> .9789 (14k) -> .9747 (28k) and its grid/staggered centres grew (.138 -> .168, .155 -> .175); E19a holds precision .983-.989 with centres .113-.149 at 14k and 28k (all six runs pass; rotated 28k .9888 / .113, holdout .070); the E20a 14k run passes too (staggered centre .181, holdout .150: closest to the limit).


### 00.4 Portability suite (13 gates through the shared pool, same config unchanged; `suite_table.py`)
| task | E14s | E17b | E17a | E19a | E20a |
| --- | --- | --- | --- | --- | --- |
| mode_hold | PASS 9/24 | PASS 9/24 | PASS 9/24 | PASS 9/24 | PASS 9/24 |
| img_bars4 | FAIL 0/24 | FAIL 0/24 | FAIL 0/24 | PASS 17/24 | PASS 17/24 |
| img_blobs4 | PASS 18/24 | PASS 18/24 | PASS 18/24 | PASS 18/24 | PASS 18/24 |
| img_intensity2 | FAIL 5/24 | FAIL 5/24 | FAIL 5/24 | PASS 16/24 | PASS 16/24 |
| img_stripes2 | PASS 21/24 | PASS 21/24 | PASS 21/24 | PASS 22/24 | PASS 22/24 |
| vector_anisotropic | PASS 22/24 | PASS 22/24 | PASS 22/24 | PASS 22/24 | PASS 22/24 |
| vector_overlap | PASS 20/24 | PASS 20/24 | PASS 20/24 | PASS 20/24 | PASS 20/24 |
| vector_spiral | PASS 24/24 | PASS 24/24 | PASS 24/24 | PASS 24/24 | PASS 24/24 |
| vector_two_broad | PASS 23/24 | PASS 23/24 | PASS 23/24 | PASS 23/24 | PASS 23/24 |
| vector_unequal_mass | FAIL 17/24 | FAIL 17/24 | FAIL 17/24 | PASS 20/24 | PASS 20/24 |
| vector_unequal_width | PASS 20/24 | PASS 20/24 | PASS 20/24 | PASS 21/24 | PASS 21/24 |
| ring_shift | PASS 270/460 | PASS 328/460 | PASS 328/460 | PASS 269/460 | FAIL 239/460 |
| stationary | PASS 700/750 | PASS 703/750 | PASS 703/750 | PASS 706/750 | PASS 706/750 |
| **passed** | 10/13 | 10/13 | 10/13 | 13/13 | 12/13 |
E19a passes all 13: img_bars4 (0/24 for every earlier candidate, including the base), img_intensity2 and vector_unequal_mass pass for the first time in this lineage, with the feature scale as the only relevant change (the support test never acted on the N <= 256 tables: `iso_acted` 0, e.g. vector_unequal_mass 0 of 600 evaluations); ring_shift and stationary (N = 20,000) are the two tasks on which the support test acts (E19a: 11,484 and 919 re-draws). The gates are advisory (the user's rule); the earlier toy failures were real failures, not bad tests (an ideal 256-row table passes vector_unequal_mass 99.8% of the time, `scratchpad/ideal_table_unequal_mass.py`), and are fixed here.


#### E20/E21 on ring_shift (the reason they are not the candidate)
E20a passes the natives (table above) and 12 of the 13 gates but **fails ring_shift 239/460** (three of its suite tasks first died of a GPU out-of-memory at start, another job had filled the card, and were rerun: PASS) (final streak 3: the noisy hq hovers at .89-.92 against the .90 limit; E19a .96) with 58% more ordinary birth-death moves (46,981 vs 29,706) and a lower clean hq (.936 vs .983). E21a (E20 + the distance limit on the parents, built on the hypothesis that a rank cap alone draws parents from other modes after the shift) passes the natives with the best centre errors of the pass (.142 / .142 / .156; holdout .106 / .108 / .134) and stationary (706/750) but also fails ring_shift (292/460, streak 0), differently: an abrupt collapse (hq .95 -> .2 within 50 steps at step ~4140, partial recovery at the end); E19a and E14s have the same kind of collapse earlier (~3440 and ~3600) and recover in time. So ring_shift is flaky in this lineage (the timing of the episode decides the streak-of-5 verdict) and I cannot say whether persistence, the p-weights or chance moved the outcome; with one seed this is where a multi-seed run is needed.


### 00.5 Probes on copies of the harness (the frozen harness is untouched; `harness-bigN/` and `harness-absence/` are scratch copies with edited task specs; `launch_bigN.sh`, `launch_absence.sh`)
- **Unequal masses at native table size** (vector_unequal_mass with 20,000 rows, batch 2048, 3,000 steps; target masses .55/.30/.13/.02; `harness-bigN/`): E14s passes the (loose) gate but its masses are off and get worse at the end (final .629/.231/.128/.012, mass TV .079, min mass ratio .586, component covariance error .27); **E22** .527/.325/.124/.0232 (mass TV .0286, min ratio .956, covariance error .16); E20a .546/.312/.121/.0215 (TV .0135, min ratio .928). Returning a stray to its own neighbourhood does not bias the masses here (the statistical review's fear: mass returns by catchment, not by mass, when a stray lies between a heavy and a light mode); the feature scale and the support test improve them.
- **Transient absence** (the 2% component withheld from the training batches during steps 1200-1300, evaluation untouched; `harness-absence/`): the component's mass ratio falls at once in E14s (.73 -> .134 at 1250), in E20a (.75 -> .098) and in E22 (.75 -> .098): the ordinary birth-death does the same as the support test (both assume the reservoir is an iid sample of the current data law); persistence of two evaluations does not help against an absence of ten reservoir turnovers. Recovery differs: **E14s never recovers** (min ratio .47 at 3000, final masses .690/.161/.139/.0095, mass TV .150 at the limit), **E22 recovers by step ~2250** (.946; .98 at 3000; final .546/.294/.135/.026, mass TV .0104), E20a likewise (.946 at 2250, final TV .0205).

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

### 00.7 Forecast scoreboard for this pass (pass/fail forecasts written to `forecast-E15..E22*.json` before the runs they concern; Brier = mean squared error of the stated probability, .25 = coin flip)
| forecast file | run | P(pass) grid / rotated / staggered | outcome | Brier |
|---|---|---|---|---:|
| forecast-E15.json | E15a | [0.85, 0.93, 0.88] | [0, 0, 0] | 0.787 |
| forecast-E16.json | E16a | [0.75, 0.55, 0.65] | [1, 1, 1] | 0.129 |
| forecast-E16-batch.json | E16b (gate off) | [0.8, 0.55, 0.7] | [1, 1, 1] | 0.111 |
| forecast-E17.json | E17a | [0.8, 0.8, 0.8] | [1, 1, 1] | 0.040 |
| forecast-E17b.json | E17b (gate off) | [0.85, 0.85, 0.85] | [1, 0, 1] | 0.256 |
| forecast-E17b.json | E17blr075 | [0.8, 0.8, 0.75] | [1, 1, 0] | 0.214 |
| forecast-E17-S4.json | E17lr075 | [0.8, 0.85, 0.8] | [1, 1, 1] | 0.034 |
| forecast-E17-S4.json | E17lr133 | [0.55, 0.75, 0.7] | [1, 1, 1] | 0.118 |
| forecast-E18.json | E18a | [0.8, 0.8, 0.8] | [1, 1, 1] | 0.040 |
| forecast-E19.json | E19a | [0.85, 0.85, 0.85] | [1, 1, 1] | 0.023 |
| forecast-E19.json | E19lr075 | [0.8, 0.85, 0.8] | [1, 1, 1] | 0.034 |
| forecast-E19.json | E19lr133 | [0.7, 0.8, 0.75] | [1, 1, 1] | 0.064 |
| forecast-E19-horizon.json | E19a14k | [0.8, 0.85, 0.8] | [1, 1, 1] | 0.034 |
| forecast-E19-horizon.json | E19a28k | [0.65, 0.75, 0.65] | [1, 1, 1] | 0.102 |
| forecast-E20.json | E20a | [0.85, 0.85, 0.85] | [1, 1, 1] | 0.023 |
| forecast-E20-S4.json | E20lr075 | [0.85, 0.85, 0.8] | [1, 1, 1] | 0.028 |
| forecast-E20-S4.json | E20lr133 | [0.7, 0.8, 0.75] | [1, 1, 1] | 0.064 |
| forecast-E20-horizon.json | E20a14k | [0.8, 0.85, 0.8] | [1, 1, 1] | 0.034 |

all scored cells: 54, Brier 0.119 (coin flip .250)
Value forecasts: E15a rotated precision .982 [.976, .986] -> .9799 inside; E15a rotated centre .13 [.10, .17] -> .245/.217 outside (the failure the forecast missed); E16a mass TV .038 [.030, .052] -> live .026-.043 inside; E17 rows re-drawn on rotated 6,000 [1,500, 13,000] -> 15,859 outside (the churn was underestimated); E20 rotated re-draws 4,500 [1,500, 11,000] -> 9,106 inside. The forecasts were over-confident about E15 and E17b, under-confident about the S4 cells.


### 00.8 Not clean, and what would fix it (ranked recommendations are in section 6)
- Single deterministic seed everywhere. Rotated is now +.0151 in precision, but the shape gate (worst mode's covariance eigenvalue ratio, limit 1.70) is at 1.41-1.66 in the natives and stress cells, E19lr133 grid's centre error is .198 (limit .20), and ring_shift is flaky (below).
- ring_shift (the shift gate) shows abrupt collapse episodes in every variant (hq .9 -> .1-.2 within 50 steps, recovery over 400-600 steps: E14s at ~3600, E19a at ~3440, E21a at ~4140); a pass needs the last five checks clean, so the outcome depends on when the episode falls (E19a PASS, E21a FAIL); the cause is not identified.
- The support test acts only where the table has at least ~800 rows and only on strays beyond ~5-6 sigma (synthetic recall .80/.99/1.0 at 5/6/8 sigma with 300 strays; .07/.59/.99 with 60); 3-4 sigma rows are almost never flagged.
- iid reservoir: a component absent from the reservoir for many evaluations loses its rows (probe 00.5; the ordinary birth-death does the same); a finite pool with exact copies is guarded (off), a near-duplicate pool (augmentations) is not.
- Chosen constants: Q reused three more times (BH level, act-only-if guard, duplicate guard), the ball factor 2, persistence absent, the 1e-3 and 1e-8 numerical floors; older constants unchanged (gate window 50, hold budget, m = 4, anchor factor 2).
- The custom hosts (the 8 side experiments of the 22-check suite) are still not run: the harness refuses trainers with methods or settings it does not know.
<!-- sec00:end -->

## 0. Previous candidate: E14s (`pkg-E14` + `overrides-E13-scaled.json`), 2026-09-29 — what it is, what it scores, what the independent review found

**How E14s is built (each step is a copy of the previous package plus one flag or fix; `diff -ruN` between neighbouring `pkg-*` directories shows exactly what changed).**
E11 (section 0b) -> **E12** (`reopen_signal: "none"`: no statistic of the real batch is read anywhere) -> **E13** (three fixes the reviewers found: the stale-reset bug in the critic-space birth-death, a robust feature layer, a safe `load_state_dict`; and a calibrated null for the row-evidence gate, `row_evidence_null: "scaled"`) -> **E14** (the critic's feature space is built from every learned scalar score head, so critics with a raw linear skip work).
On the three native gates E14 is byte-identical to E13 (single-head critic; verdict files compared), E12 is byte-identical to E11 wherever the data are stationary (the drift statistic never fired in any of the 15 E11 native runs: reopen counter 0).

**What E14s does (network-internal mechanisms only; the rule is in `REQUIREMENTS.md` 0.0):**
1. *Table stationarity tester* (inherited): halves the table step scale `s` when the bulk of the rows has stopped moving coherently (sign of block-displacement cosines at two scales). In intrinsic time it behaves like a fixed `1/t` ladder: it can only delay a halving, never speed it up (reviewer 1 measured for E11 that every halving came at the earliest step the test allows: rotated 441 + 768 * 2^j; the tester never released after settling).
2. *Anchored release*: a release (the table step rising again) is accepted only when its evidence scale is at least twice the scale where the bulk last settled; otherwise it is treated as inconclusive. On the native runs it blocked at most one verdict; the ladders are monotone.
3. *Row-evidence gate*: every table row keeps exponentially weighted statistics (window 50 touches) of its own gradients and is tested against "no persistent push"; Benjamini-Hochberg (BH) at q = .05 over the rows. Flagged ("hot") rows keep the full step (the tester's scale is undone for them), do not vote in the tester, and descent is held while more than q of the rows are flagged. **E13's change:** the p-values use the same law applied to `t2 / c`, where `c >= 1` is the smallest scale that puts the median p-value of the tested rows at .5, i.e. the typical row is the null. That absorbs whatever inflation the bulk has (correlated gradients, a critic that slows down after each halving, another noise level) and flags only rows that are extreme relative to the bulk.
4. *Birth-death teleports in the critic's feature space*: the base's density-ratio evidence (kNN, sequential test) computed on the inputs of the critic's learned score heads; locality (anchors, radii, stale resets) in the table's own latent space. **E13's fix:** the stale-reset sites were read after the moves had overwritten the table rows (a view instead of a copy), so the death sites were lost; **E14's change:** all learned scalar heads are used (a critic with `score = f(x) + w.x` no longer silently falls back to raw samples), a critic whose heads read the raw sample is refused with an error.
5. *Served averaged model*: an exponential average of the training iterate (window = 4 tester blocks) is swapped into the live parameters while the tester's last decisive verdict is STATIONARY; the fast iterate keeps training and is what `state_dict` saves.
No statistic of raw data or of raw generator outputs is computed anywhere in the executed path (audit in `README.md`). Declared inputs: `output_noise_std` .029 (the data's noise level), host critic and table size.

### 0.1 Native gates (single deterministic seed per cell; last-5 worst terminal check, holdout in brackets; frozen limits: live precision >= .97, mode masses in [.005, .02], mode covariance eigenvalue ratios in [.4, 1.7], radial median ratios in [.65, 1.4], mass TV <= .10; accuracy gate on the final five 20k clouds and the 100k holdout: mass TV <= .06, centre RMS <= .20 sigma, |covariance trace bias| <= .10, radial KS <= .04)
| run | grid100 | rotated100 | staggered100 | live |
|---|---|---|---|---:|
| **E14s 7k** (= E13s) | **P** prec .9836 ctr .138 (hold .106, x_hat .097) | **P** prec .9728 (hold .9723) ctr .130 (hold .091, x_hat .080) | **P** prec .9758 (hold .9777) ctr .155 (hold .128, x_hat .120) | **3/3** |
| E13t 7k (theory null, else E13) | P .9835 / .139 (.110) | P .9725 / .133 (.087) | P .9752 / .154 (.129) | 3/3 |
| E12 = E11 7k (before the stale-reset fix) | P .9841 / .172 (.142) | P .9735 / .131 (.097) | P .9722 / .148 (.116) | 3/3 |
| **E14s 14k** | **P** prec .9836 ctr .147 (hold .123, x_hat .115) | **P** prec .9789 (hold .9792) ctr .111 (hold .071, x_hat .057) | **P** prec .9778 (hold .9793) ctr .148 (hold .119, x_hat .111) | **3/3** |
| **E14s 28k** | **P** prec .9845 ctr .168 (hold .141, x_hat .135) | **P** prec .9747 (hold .9738) ctr .116 (hold .082, x_hat .070) | **P** prec .9775 (hold .9784) ctr .175 (hold .143, x_hat .136) | **3/3** |
The 14k and 28k runs are the same trajectory as the 7k one (no horizon enters the recipe; the 14k run is a prefix of the 28k run and their ladders are identical); nothing degrades: rotated precision .9728 (7k) -> .9789 (14k) -> .9747 (28k), grid centre .138 -> .147 -> .168, staggered centre .155 -> .148 -> .175 (the accuracy limit is .20; last-5 worst). The table's ladder (the tester of the particle-table group; the earlier draft of this paragraph read the generator's group by mistake) is slow and monotone: grid 1 until step ~4,600, then .5 (10,750: .25; 25,250: .125), rotated halvings at 1,250 / 2,000 / 3,500 / 6,750 / 12,750 (scale .031 at 28k), staggered at 2,250 / 5,000 / 12,250 / 20,250 (scale .0625 at 28k), all read at the 250-step checkpoints; each halving takes about twice as long as the one before it (an implicit 1/t schedule in intrinsic time). The first STATIONARY verdict, from which the average is served, comes at step 4,600 on grid (E11: 5,112), 1,144 on rotated (1,208) and 2,040 on staggered (3,064): the grid gate result therefore depends on the averaged model over the last 2,400 steps, as reviewer 2 found for E11 (unaveraged, E7's grid failed at centre .219).
Forecast (`forecast-E14s-horizon.json`): 14k .85/.70/.85 (Brier .045), 28k .70/.55/.70 (Brier .128); all six passed.
The stale-reset fix moved the grid centre error from .172 to .138; the null (scaled vs theory) moved nothing on these tasks (all metrics within .0007 / .004), so the accuracy comes from the birth-death repair and the averaged model, and the calibrated null is a portability repair, not an accuracy repair. Rotated keeps a thin precision margin (+.0028 on the last live check, +.0023 on the holdout, against the .97 limit).

### 0.2 What each piece contributes (ablations, 7k, one deterministic run per cell; F = fails the frozen gate)
| variant | grid | rotated | staggered | reading |
|---|---|---|---|---|
| E14s (everything) | P .9836 | P .9728 | P .9758 | |
| E13 with the gate off (`E13ng`) | P .9833 | **F** .9687 | P .9741 | gate = +.004 precision on rotated |
| E11 with the gate off | P | **F** .9625 | P | same reading before the fixes |
| E12 without the hold (`nohold`) | P | **F** .9665 | P | |
| E12 gate = hot rows only (`hotonly`: no hold, no exclusion) | P | **F** .9690 | P | exclusion changed nothing on grid/staggered (bit-identical to `nohold`) |
| E4 lineage without birth-death | F | P | F | (history) birth-death is what balances the mode masses |
| E7 = E4 + feature-space birth-death, no served average | F ctr .219 | P | P | the averaged model is what brings grid inside .20 |
Every part of the gate matters on rotated (.9625 without the gate -> .9690 hot rows -> .9665 hot + exclusion -> .9735 with the hold); the forecasts for the two ablations run under E12 were too optimistic on rotated (.70 and .65 for cells that failed): rotated is the discriminating gate and it is decided by strays (rows more than 3 sigma from every mode centre, 2.5-2.9% of the table at 7k).
Gate calibration (final `row_evidence` counters): mean flagged fraction over the run 3.3-4.9% with the theory null (E11/E12/E13t: the hold budget is 5%, i.e. the gate lives at the edge of its own hold threshold), **.12% / 2.26% / .42%** with the scaled null (grid / rotated / staggered), mean scale `c` 1.68 / 1.24 / 1.42. The hold (descent deferred while more than q of the rows are flagged) was engaged in 21.3 / 14.9 / 10.2% of the steps with the theory null (E11) and in **0 / 10.5 / 0%** with the scaled null; hot rows per step 92 / 298 / 159 (E11) against 22 / 298 / 75 (E13s): the gate now stays quiet where nothing is out of equilibrium and works on rotated, where more than 5% of the rows really are. On the saved E11 final states the scaled null flags 129 instead of 642 rows on rotated with stray precision .71 instead of .25 (recall .33 instead of .58) and 0.16% instead of 2.3% of the bulk.

Gate calibration against the noise level (rotated100, 7k; only `output_noise_std` changes, .015 = half and .060 = double the declared .029; the declared value is a dataset input, so every one of these runs fails the gate for the trivial reason that the model's noise no longer matches the data's: precision .943-.948 at half noise, .666 at double noise; what is measured is the gate, not the score):
| null | sigma_out .015 | nominal .029 | sigma_out .060 |
|---|---|---|---|
| theory (E14 with `row_evidence_null: theory`) | mean flagged **7.99%**, hold on **58.4%** of steps | 4.26%, hold 14.7% (E13t) | 1.31%, hold 7.6% |
| **scaled** (E14s) | mean flagged **2.59%**, hold 7.6% of steps, mean c 1.86 | 2.26%, hold 10.5%, c 1.24 | .77%, hold 5.3%, c 1.08 |
The theory null doubles its flag rate and keeps the hold on for most of the run when the noise is halved (the reviewers' E4 runs showed up to 29% flagged); the scaled null moves by .3 percentage points. The forecast (`forecast-E13s-confirm.json`) was: theory > 8% / < 2%, scaled < 4% / < 2%: observed 7.99 / 1.31 / 2.59 / .77%.

### 0.3 Robustness to the base learning rate (the "no learning-rate adjustment" test), same seed, only `overrides["lr"]` changes
| candidate | lr x.75 | nominal | lr x1.33 |
|---|---|---|---|
| **E14s** (= E13s) | P .166 (.136) / P .122 (.097) / P .147 (.120) = **3/3** | P / P / P = **3/3** | **F**: the largest mode covariance eigenvalue ratio is 1.72-1.83 on the last four checks (limit 1.70; it was 1.55-1.64 before); precision .982, centre .154, mass TV .030 are all inside / P .140 (.118) / P .145 (.110) = **2/3** |
| E11 (before the fixes) | 3/3 | 3/3 | 3/3 (P .167 / P .131 / P .156) |
| E4 | 3/3 | 3/3 | 0/3 |
| seq-C (base) | 2/3 | 2/3 | 2/3 |
The x1.33 grid failure is one mode that elongates at the end (covariance eigenvalue ratio 1.72-1.83 against the live limit 1.70), not a precision, centre or mass failure; E11's 6/6 came from code with the defects listed in 0.5, so 5/6 is the honest number for the corrected candidate. Forecast (`forecast-E13s-confirm.json`): .80/.65/.80 and .70/.60/.70 (the x1.33 grid was forecast .70 and failed); Brier .157.

### 0.4 Portability suite (13 gates through the shared pool, same config unchanged; `suite_table.py`)
| task | seq-C (base) | E4 (codex run) | E10 | E11 | E12nr | E13s | E14s |
|---|---|---|---|---|---|---|---|
| mode_hold | PASS 9/24 | PASS 9/24 | FAIL 4/24 | PASS 9/24 | PASS 9/24 | PASS 9/24 | PASS 9/24 |
| img_bars4 | FAIL 0/24 | FAIL 0/24 | FAIL 0/24 | FAIL 0/24 | FAIL 0/24 | FAIL 0/24 | FAIL 0/24 |
| img_blobs4 | PASS 18/24 | PASS 18/24 | PASS 17/24 | PASS 17/24 | PASS 17/24 | PASS 18/24 | PASS 18/24 |
| img_intensity2 | FAIL 2/24 | FAIL 2/24 | PASS 16/24 | PASS 16/24 | PASS 16/24 | FAIL 5/24 | FAIL 5/24 |
| img_stripes2 | PASS 22/24 | PASS 22/24 | PASS 22/24 | PASS 22/24 | PASS 22/24 | PASS 21/24 | PASS 21/24 |
| vector_anisotropic | PASS 22/24 | PASS 22/24 | PASS 22/24 | PASS 22/24 | PASS 22/24 | FAIL None/0 | PASS 22/24 |
| vector_overlap | PASS 21/24 | PASS 21/24 | PASS 24/24 | PASS 20/24 | PASS 20/24 | PASS 20/24 | PASS 20/24 |
| vector_spiral | - | PASS 24/24 | PASS 24/24 | PASS 24/24 | PASS 24/24 | PASS 24/24 | PASS 24/24 |
| vector_two_broad | PASS 23/24 | PASS 23/24 | PASS 23/24 | PASS 23/24 | PASS 23/24 | PASS 23/24 | PASS 23/24 |
| vector_unequal_mass | PASS 14/24 | PASS 14/24 | FAIL 12/24 | PASS 17/24 | PASS 17/24 | FAIL 17/24 | FAIL 17/24 |
| vector_unequal_width | PASS 20/24 | PASS 20/24 | PASS 20/24 | PASS 21/24 | PASS 21/24 | PASS 20/24 | PASS 20/24 |
| ring_shift | - | PASS 355/460 | PASS 332/460 | PASS 345/460 | PASS 314/460 | PASS 270/460 | PASS 270/460 |
| stationary | - | PASS 709/750 | PASS 688/750 | PASS 704/750 | PASS 704/750 | PASS 700/750 | PASS 700/750 |
| **passed** | 8/10 | 11/13 | 10/13 | 12/13 | 12/13 | 9/13 | 10/13 |
E14s passes **10 of 13**: img_bars4 fails as for every candidate of this lineage and the archived one; **img_intensity2** and **vector_unequal_mass** fail where E11/E12 passed. Both are tasks whose tables are tiny and never settle: in img_intensity2 the table tester never declares STATIONARY (nothing is averaged; E11's tester declared it at steps 312 and 504, so E11 served an average), and vector_unequal_mass passes 17 consecutive checks (steps 200-1000) and fails the last four (hq .936-.950); the birth-death repair changes how often stale evidence is dropped (img_intensity2: 273 moves and 659 stale resets against E11's 168 and 262), so these are chaotic single-seed flips at the pass boundary, not a measured mechanism effect (E13t, the theory null, reproduces the img_intensity2 failure exactly: 5/24). `vector_anisotropic` (critic with a raw linear skip) **passes with the learned-head feature space (22/24)**; in E11-E13 it ran birth-death on raw samples (E11-E12) or errored on the raw-head guard (E13), see 0.5.
Averaging was served in only two suite tasks under E11 (img_intensity2, vector_overlap; reviewer 2), so this table mostly tests the base's un-annealed dynamics plus the feature-space birth-death; the native gates are where the new mechanisms operate.
The 8 custom22 hosts were not run: `harness/components.py` refuses any package whose `GANTrainer` defines methods or non-default recipe fields outside `KNOWN_TRAINER_METHODS` / `KNOWN_RECIPE_FIELDS` (this lineage adds `_stray_gate`, `_table_tester`, ... and `row_evidence_gate`, `table_release_rule`, `birth_death_space`, `serve_average`, ...); the base ERRORs them too.


### 0.5 What the two independent reviews of E11 found and what was done (reports were returned as messages; evidence, scripts and predictions are in `design/review_E11_stats/` and `design/review_E11_code/`)
| finding | severity | action |
|---|---|---|
| the data-drift statistic (real-batch random features, standardised by the first batch) is live: it re-opens every tester (`data_score > 3`), and also feeds the KA2 anchor pin and `game_trust`; it fires on ring_shift (296 tester reopens) and on any non-iid ordering of a stationary stream (6,999 of 7,000 steps when the same data arrive class-sorted); I had called it inert | high | **removed from the executed path** (E12, `reopen_signal: "none"`, `observe_blind`): ring_shift still passes (314/460, second segment arrives after 240 steps instead of 430, with 8 departures), stationary and every other gate identical, natives byte-identical; the anchor pin stays on (drive 0) exactly as in every native run |
| the gate is not an error-controlled test: bulk p < .01 in 7-11% of rows (nominal 1%), mean flagged fraction 3.3-4.9% against a hold budget of 5%, flagged rows are 93% bulk on grid (13 of 1,294 are strays), false flags grow after each ladder halving (the critic's step is slaved to the table's), depends on the noise level (up to 29% flagged at half noise, ~0% at double), synthetic nulls with correlation >= .4, a slow shared component or low-rank d=32 covariance break it; the statistic is a pooled-variance t^2, not Hotelling's T^2, so the "exact F" law is not exact | high | E13 `row_evidence_null: "scaled"` (0.2); measured effect above; **not fixed:** low-rank covariance in high d (the pooled statistic assumes isotropic noise), lag-1 deflation of `n_eff`; "d=32 never runs" was wrong: the test starts after about 2,000 steps at d=32 (window 50, n_eff >= 3d) |
| critic-space birth-death picked "the last nn.Linear in module order" as the feature layer: for critics with a raw linear skip (two of the eight frozen host critics, one of the 13 gates: vector_anisotropic) that is `Linear(2, 1)` fed the raw samples, i.e. E11's suite result on that gate silently used data-space birth-death | high | E13 refuses raw heads (error), E14 uses every learned scalar head (96 = 64 + 32 features on the additive critic); **vector_anisotropic passes with learned features (22/24)** |
| stale-reset bug (`q_loc = z` is a view; the moves overwrite it; the child's old site is lost, the data branch reads pre-move outputs) | medium | fixed (`z.clone()`); at N=200 over 400 toy steps the stale resets go 490 -> 596; grid centre .172 -> .138; the suite tasks whose tables are tiny move differently (img_intensity2, vector_unequal_mass flip to FAIL, see 0.4) |
| the grid pass comes from the served average alone, which starts at the first STATIONARY verdict (E11 grid: step 5,113); in the 13-gate suite the average was served in only 2 tasks (12/13 says nothing about averaging); serving ignores the hold | high | reported, not changed: the averaged model is a declared, allowed part; at 14k and 28k (0.1) the dependence is measured; serving-with-hold left as is because forbidding it would switch the grid average off |
| the ladder is an implicit 1/t schedule with no floor; after settling a release needs >= 24 b_anchor / s steps (6-12k); the serving window 4b/s reaches ~2,000 steps at 28k | medium | measured at 14k and 28k (0.1); with the data-drift reopen removed, a genuine late distribution shift on a settled table is not followed quickly (the ring hosts never settle their table, so the suite does not test this) |
| constants: anchor factor 2, m = 4, the 3d rule, q reused as the hold budget, window 50 are chosen, not derived (window and m were looked at on S1 logs); the tester's Bonferroni comment says .05 per decision but the code path uses a .9875 quantile so the level can reach .075 (inherited) | medium | declared (`README.md` ledger); no sensitivity sweep was run (working rule: no constant sweeps) |
| the feature-space evidence assumes one intrinsic dimension for all points (calibrated on 2-D/8-D mixtures and conv images; sd of the standardised evidence 1.78 on mixed 1-D/3-D structure) | medium | open (a dimension-free two-sample kNN count would remove it) |
| `load_state_dict` released the served average before validating (a rejected checkpoint left the trainer released); 12 of 27 injected bugs were caught by no shipped test | medium | fixed; the reviewer's serve-symmetry, BH-reference, hook-leak, stale-site and failed-load tests are adopted in `tests/review2/`, `tests/test_E13.py`, `tests/test_E14.py` (they catch the serving, feature-layer, guard, teleport-reset and hook mutants; still uncaught: "early look ignores the hold", "locality radius = nearest neighbour") |
| custom hosts refuse this package: `components.py` allows only the base's trainer methods and recipe fields; adding row-evidence, hot-undo and serving to `Engine.update` and `state_dict` is the smallest correct change (allow-listing the names would silently run base dynamics) | info | harness is frozen; a decision for its owners |
Found sound by the reviewers: flags-off parity with seq-C (parameters and every RNG stream), the serve swap symmetry on `state_dict`/load/failed step/`sample`, BH (600/600 sets equal to a float64 reference), the float32 shortlist with exact recompute (0 of 60,000 neighbours missed), no hook leak, no crash on empty flagged sets, tiny N or NaN rows, QR/table initialisation without data, the T^2 statistic conservative in d=2 under iid noise, the critic-space evidence calibrated on native-style critics.

### 0.6 Forecast scoreboard (all forecasts were written to `forecast-*.json` before the runs they concern; Brier score = mean squared error of the stated probability, 0.25 = coin flip)
| forecast file | cells | Brier |
|---|---|---|
| `forecast-E11-S4.json` (E11 at lr x.75 / x1.33) | 6 | .15 (observed 6/6 pass) |
| `forecast-E11-14k-nogate.json` | 6 | .18 (14k all pass; gate-off: rotated F only) |
| `forecast-E12-noreopen.json` (ring hosts without the drift statistic) | 2 | .03 (both pass) |
| `forecast-E12nr-confirm.json` (identity with E11) | 5 | .01 (all identical) |
| `forecast-E12-ablate.json` (nohold / hotonly) | 6 | .19 (rotated failed both times) |
| `forecast-E13.json` (E13 scaled / theory) | 6 | .14 |
| `forecast-E13-nogate.json` | 3 | .09 (rotated F, forecast .45) |
| `forecast-E13s-confirm.json` (S4, suite, gate calibration) | 6+ | S4 .157 |
| `forecast-E14.json` (vector_anisotropic .65, natives identical .97) | 4 | .06 (both as forecast) |
| `forecast-E14s-horizon.json` (14k, 28k) | 6 | .09 (all pass) |
| gate calibration vs noise (in `forecast-E13s-confirm.json`) | 4 | 3 of 4 thresholds met (theory at half noise 7.99% against "> 8%") |
Systematic error: I over-predict rotated passes for every ablation (that task lives at +.0008); I under-predict the 14k sustainability.

## 0b. The previous candidate E11 (pkg-E11 + overrides-E11.json), 2026-09-29 - superseded by E14s (its 12/13 suite and 6/6 S4 came from code with the defects listed in 0.5)
E1-E4 below inherit the base's birth-death, which does a nearest-neighbour test of the table's samples against a real reservoir **in data space**: that is forbidden by the rule, so they are history, not solutions. E11 replaces it and adds a served averaged model. Parts, each behind one flag:
1. `row_evidence_gate` (E1): each table row keeps a statistic of its own gradients; rows with a persistent push keep the full step size ("hot rows"), do not vote in the table tester, and descent is held while many rows are flagged (network-internal).
2. `table_release_rule: anchor` (E4): a release (the table rate rising again) is accepted only if its evidence scale is at least twice the scale where the bulk last settled (network-internal).
3. `birth_death_space: critic` (E7): the base's birth-death evidence (kNN density ratio of the model's samples against real samples) computed on the **critic's penultimate features** instead of on raw samples; locality stays in the table's own latent space; float32 shortlist with exact float64 recompute (E10).
4. `serve_average: 4` (E9/E11): the model of record is an exponential average of the training iterate with a window of 4 table-tester blocks (steps per block = b/s of the tester, controller state); the trainer swaps the average into the live parameters between steps so the unchanged harness scores it; the fast iterate keeps training (training is identical to E7's); it is served only while the tester's last decisive verdict is STATIONARY (E11), because a position average of rows that still migrate between modes blurs them.

| 7k live | grid | rotated | staggered | live |
|---|---|---|---|---:|
| **E11** | **P** prec .9841 ctr .172 (hold .142, x_hat .135) | **P** prec .9735 (hold .9729) ctr .131 (hold .097, x_hat .086) | **P** prec .9722 (hold .9747) ctr .148 (hold .116, x_hat .107) | **3/3** |
| E7 (same without the served average) | F ctr .219 (hold .193) | P prec .9749 ctr .180 | P prec .9725 ctr .195 | 2/3 |
| E4 (data-space birth-death; history) | P ctr .197 (hold .165) | P prec .9727 ctr .158 | P ctr .182 | 3/3 |
| E4 with birth-death removed | F ctr .265, 11 streaked modes | P prec .9768 ctr .196 | F ctr .216 | 1/3 |
Centre margins are .03-.07 (E4: .003-.04); precision margins on rotated and staggered are still thin (+.002 to +.004).

**Robustness to the base learning rate (the "no learning-rate adjustment" test), same seed, only `overrides["lr"]` changes (grid / rotated / staggered, last-5 worst centre; holdout in brackets):**
| candidate | lr x.75 | nominal | lr x1.33 |
|---|---|---|---|
| **E11** | P .195 (.121) / P .125 (.094) / P .143 (.103) = **3/3** | P .172 / P .131 / P .148 = **3/3** | P .167 (.125) / P .131 (.095) / P .156 (.106) = **3/3** |
| E4 | 3/3 | 3/3 | F / F / F = 0/3 |
| seq-C (base) | P / F / P = 2/3 | P / F / P = 2/3 | F / P / P = 2/3 |
| seq-G (kNN gate, data space; history) | 3/3 | 3/3 | F / P / F = 1/3 |
E11 passes all six cells; the S4 forecast (`forecast-E11-S4.json`, written before the runs) was .75/.65/.75 at x.75 and .50/.55/.55 at x1.33. Averaging the served model makes the result nearly independent of the table's step size, as Polyak-Ruppert averaging predicts.

**Portability suite (13 gates through the shared pool, same config unchanged; `suite_table.py`):**
| task | seq-C (base) | E4 (codex run) | E10 | E11 |
|---|---|---|---|---|
| mode_hold | PASS 9/24 | PASS 9/24 | FAIL 4/24 | PASS 9/24 |
| img_bars4 | FAIL 0/24 | FAIL 0/24 | FAIL 0/24 | FAIL 0/24 |
| img_blobs4 | PASS 18/24 | PASS 18/24 | PASS 17/24 | PASS 17/24 |
| img_intensity2 | FAIL 2/24 | FAIL 2/24 | PASS 16/24 | PASS 16/24 |
| img_stripes2 | PASS 22/24 | PASS 22/24 | PASS 22/24 | PASS 22/24 |
| vector_anisotropic | PASS 22/24 | PASS 22/24 | PASS 22/24 | PASS 22/24 |
| vector_overlap | PASS 21/24 | PASS 21/24 | PASS 24/24 | PASS 20/24 |
| vector_spiral | - | PASS 24/24 | PASS 24/24 | PASS 24/24 |
| vector_two_broad | PASS 23/24 | PASS 23/24 | PASS 23/24 | PASS 23/24 |
| vector_unequal_mass | PASS 14/24 | PASS 14/24 | FAIL 12/24 | PASS 17/24 |
| vector_unequal_width | PASS 20/24 | PASS 20/24 | PASS 20/24 | PASS 21/24 |
| ring_shift | - | PASS 355/460 | PASS 332/460 | PASS 345/460 |
| stationary | - | PASS 709/750 | PASS 688/750 | PASS 704/750 |
| **passed** | 8/10 | 11/13 | 10/13 | 12/13 |
E11 passes 12 of 13: only `img_bars4` fails, as it does for every candidate of this lineage (base, E4, E10, and the archived lineage); `img_intensity2`, which the base and E4 fail, now passes; `mode_hold` and `vector_unequal_mass` pass after the settled-only gate (E10, which served the average unconditionally, failed both: the regression came from averaging the small tables' migrating rows, attributed by E7 (no averaging: both pass) and E9data (data-space birth-death plus averaging: both fail)).
The 8 custom22 hosts were not run (the base ERRORs them on critic input noise: a harness refusal).

**What is still not rule-clean (declared, README section "audit"):** the base's real-batch drift statistic (`DataDriftController`, data space; never fired in any native run: no tester reopen, drive 0) is still in the package; `output_noise_std` .029 (the data's own noise level) is a declared dataset input (learned freely it collapses: .029 -> .004 in 3,000 steps from every start; wrong by 2x it fails);
the host critic's Fourier frequencies are tuned to sigma .03 (host-provided); lr .00425, batch 2048, the gate's window (50 touches) and hold (Q), the averaging constant m = 4 are constants whose portability is what the suite tests. Pending: 14k (S1b), the ablation without the row-evidence gate, an independent review of E11.

## 1. History: E1-E4 (data-space birth-death inherited from the base; not rule-clean) - leaderboard (7k live unless stated)
| rank | candidate | grid | rotated | staggered | live | notes |
|---|---|---|---|---|---:|---|
| 1 | **E4** = gate (hot rows + exclusion + hold) + ANCHOR release | P ctr .197 (hold .165, x_hat .159) | P prec .9727 (hold .9722) ctr .158 (.139) | P ctr .182 (.166, x_hat .160) prec .9747 | **3/3** | THIN on rotated precision (+.0022 on the holdout) and grid centre (+.003); 14k: 3/3 PASS; RULES: violates A2 (window W, hold budget) -> DIAGNOSTIC |
| 1 | E2a = gate + release `never`, E2b = gate + release `both` | identical to E4 (bit-identical runs) | identical | identical | **3/3** | E2b 14k: all three PASS, monotone ladders, no release after 7k (below) |
| 3 | E4nh = E4 without the hold (recommended by the audit: Q keeps one role) | **F ctr .2012** (hold .177) | P prec .9766 (.9764) ctr .177 (.124) strays .0118 | P (= E4) | 2/3 | grid misses by .0012 on the terminal check; ladder identical to E4, chaos scale is .03 |
| 3 | E4nhx = E4nh without exclusion (hot rows only + ANCHOR, no hold) | F ctr .2012 (= E4nh) | P prec .9727 (hold .9720) ctr .158 (.141), identical to E4 | P (= E4) | 2/3 | the smallest mechanism that still passes rotated |
| 5 | E1 = gate with the unmodified table tester | P (= E4) | **F** prec .9568 ctr .272 tr .120 KS .058, releases 4921/5305/5401 | P (= E4) | 2/3 | the gate alone does not stop the release cascade |
| 6 | seq-C (base) | P ctr .197 | F prec .9545, releases 4793/5177/5273/6689 | P ctr .173 | 2/3 | |
| ref | seq-G (kNN support gate; **I2**, inadmissible) | P | P prec .9755 | P | 3/3 | upper bound with a real-reservoir flag |

## 2. Attribution on rotated100 (the discriminating task; one-factor changes of E4, 7k live)
| arm | gate parts | release rule | last5 prec | strays | ladder | verdict |
|---|---|---|---:|---:|---|---|
| seq-C | none | any | .9545 | .0206 | releases 4793,5177,5273,6689 | F |
| E2c | none | never | .9670 | .0198 | monotone 1081/1849/3257 | F |
| E1 | hot+excl+hold | any | .9568 | .0175 | releases 4921,5305,5401 | F |
| E3noHot | excl+hold | both | .9664 | .0234 | monotone (4153:.0625) | F |
| E3noExcl | hot+hold | both | .9727 | .0155 | monotone | P |
| E2b / E4 | hot+excl+hold | both / anchor | .9727 | .0154 | monotone 1081/1849/3385/6969 | P |
| E4nh (= E3noHold) | hot+excl | anchor | .9766 | .0118 | monotone 825/1593/3129/6713 | P |
| E4nhx | hot | anchor | .9727 | .0155 | monotone 1081/1849/3385/6969 (= E4) | P |
Reading: (a) hold and exclusion act only through the timing of the first descents (with the hold off and exclusion on the tester descends at 825 instead of 1081; with the hold off and exclusion off, or the hold on, the ladders coincide with E4). (b) without a release fix the gate fails (E1); without the gate the release fix fails by .003 (E2c); both are needed. (c) hot rows are the operative part of the gate (removing them is worse than no gate),
exclusion is redundant once the release is fixed, and deleting the hold helps rotated (strays .0154 -> .0118) but moves grid across .20 by .0012. (d) the three release rules (`never`, `both`, `anchor`) are bit-identical on all
six runs: exactly one DRIFT verdict (step 4920, tested scale 8, mean cosine +.083 at b, -.043 at 2b, last STATIONARY evidence scale 16-32) was converted to INCONCLUSIVE and the ladder never released again.

## 3. Sustainability (S1b) and robustness (S4)
| run | grid | rotated | staggered |
|---|---|---|---|
| E2b at 14,000 (official 5-check verdict at the last multiple) | P ctr .174 (hold .148) prec .9825; ladder 4089:.5 10233:.25 | P prec .9723 ctr .160 (hold .126); ladder unchanged after 6969 | P ctr .161 (hold .132) prec .9776; ladder 3065:.5 6137:.25 10233:.125 |
| E4 at 14,000 (bit-identical to E2b at 14k) | P ctr .174 (hold .148); ladder 4089:.5 10233:.25 | P prec .9723 ctr .160 (hold .126); ladder unchanged after 6969 | P ctr .161 (hold .132); ladder 3065 6137 10233 |
| E4nh (hold removed) at 14,000 | P ctr .200 (hold .171), streak 9: at the edge | **F** ctr .269 (spike right after a late descent 12857:.03125), prec .9684, eig 1.76 | P ctr .183 |
| E4 with base lr x .75 (S4) | **P** ctr .170 prec .9842 | **P** prec .9727 ctr .165 | **P** ctr .181 prec .9731 |
| E4 with base lr x 1.33 (S4) | **F** ctr .209 tr .046 KS .026 | **F** ctr .219 (prec .9760) | **F** ctr .205 |
S4 with controls (same harness, same seed, only `overrides["lr"]` changes; grid / rotated / staggered):
| candidate | lr x.75 | nominal | lr x1.33 |
|---|---|---|---|
| seq-C (base) | P ctr .163 / **F** prec .9688 / P ctr .183 = 2/3 | P / F / P = 2/3 | **F** ctr .227 / P prec .9708 ctr .190 / P ctr .184 = 2/3 |
| seq-G (kNN gate, I2) | P ctr .163 / P prec .9759 ctr .199 (at the edge) / P ctr .178 = 3/3 | 3/3 | **F** ctr .213 / P prec .9796 ctr .180 / **F** ctr .261 (releases 3449, 6137) = 1/3 |
| **E4** | P ctr .170 / P prec .9727 ctr .165 / P ctr .181 = 3/3 | 3/3 | **F** ctr .209 / **F** ctr .219 / **F** ctr .205 = 0/3 |
Reading: at -25% E4 matches the I2 reference and beats the base; at +33% nobody is 3/3 (the fragility is inherited from the sign-only ladder) and E4 is the weakest in this single-seed comparison (rotated centre .219 vs .180-.190), a difference at the chaos scale (.03) but not a robustness gain.
S4 forecast (written before the runs, `forecast-E4-S4.json`): P(pass) .55/.5/.55 at x.75 and .35/.45/.45 at x1.33; observed 3/3 and 0/3 (Brier .20 on 6 cells). No table release occurred in any S4 run. The ladders take the same number of halvings at x1.33
(rotated 761/1529/2297/6393), so the jitter at the same scale is larger and the sign-only tester does not compensate: it is amplitude-blind (A7 stays open for the inherited statistic).

## 4. What the diagnostics established (all evaluation-side; scripts in `analysis/`, facts in `design/FACTS.md`)
1. The final strays are not leftovers of the initial transport: only 22-28% were never captured; 72-78% left a mode after step 2000 (evaporation). In seq-C rotated 36 of 58 late departures fall in the tester releases (steps 5500-6750);
   displacements are gradient-limited (no teleports). The critic gives strays only a weak inward force (mean +.0016..+.0046 vs |F| .005-.010) and no far-field ramp; it cannot see an individual stray (one point, weight 1/B).
2. The release cascade is driven by the bulk, not the strays: replaying the tester statistic on the seq-C probe (`tester_replay.py`), the row-mean block cosine at the short scale turns positive as the rate falls (+.03..+.05, t up to +10 over 20,000 rows)
   while the long scale is only marginally negative (t -1.8..-2.7 against the strict -2.6); excluding oracle strays changes nothing. Mean, median and trimmed aggregates behave alike (strays are not outliers of a bounded cosine).
3. Jitter is critic-field noise on the mode-mean force divided by a weak restoring stiffness (.15 sigma offset = .13-.33 of the noise per 125 steps); the annealing ladder is the only lever, so per-row or per-mode significance rules cannot replace it for the bulk.
4. A row's own gradient history (Hotelling T^2 of the weighted mean of its last ~50 touches) separates rows currently >3 sigma from the bulk with AUC .91-.95 at every time and is near its null on the bulk; BH at Q=.05 flags ~150 of 6000 rows with recall .5-.75, precision .65-.80. It has no predictive power for rows that will leave (AUC .45-.64).
5. Refuted on logs by the design panel (`design/`): critic-value (WFR) row weights (D at strays AUC .47-.55), Rprop-type sign rules, kNN-pooled row evidence, robust row aggregation in the tester, a per-row dead-zone gain on the whole table (freezes mode offsets at .2-.4), a ladder-free table.

## 5. Admissibility status (REQUIREMENTS section 10; the full audit is `README.md`)
RULES: two mechanisms behind two flags (`row_evidence_gate`, `table_release_rule`), both Controller layer, both I0. Flagged A2 (window W=50 has no fixture calibration before the first native log and was chosen after offline looks at 25/50/100/200 on the rotated probe; Q has a second role as hold budget in E4),
A7 (the inherited sign-only statistic remains; the anchor bounds the release rate but the S4 x1.33 result shows the amplitude-blindness), A5 (ablations of hot/exclusion/hold exist; hold and exclusion are not needed). Not done: S3 (13-gate regression; needs go-ahead), 28k, A9 x.5/x2 sensitivity of W (an offline synthetic fixture says W=100 breaks: `calibration/window_fixture.py`), the review verdict (running).
Tests: parity flag-off == seq-C (digest ecb590cf049cb962), checkpoint replay and step-offset identical, decoy test identical (digest d9bb330716fe2b1c with the hot branch executing 4,323 row-steps), lint clean except the declared literals and `moved_rows` bookkeeping.

## 6. Recommended next experiments (ranked; updated 2026-09-29 for E22; the E14s list is in `RESULTS.E14s-only.bak`, older ones in `RESULTS.E11-only.bak`)
1. **Three seeds of the three native gates for E22, with E14s as the control (18 runs, about one GPU-hour on a quiet machine).** The user's condition (solve the thin rotated margin on one seed first) is met: rotated precision +.0151, S2 FULL. The seeds differ only in sample order (the QR initialisation is deterministic), which is exactly what the single-seed numbers cannot bound. Needs the standing no-seed-sweeps rule lifted once (a user decision). Falsifier: any seed of E22 below .973 precision on rotated or a shape-limit failure on grid.
2. **Evidence accumulation for the support test and a longer memory** (REQUIREMENTS 9.9): the probe shows a 2% component withheld for 100 steps is erased within 50 steps (E14s and E19/E20 alike) and only the support-test variants rebuild it. Replace the single-evaluation flag by a sequential test over the last evaluations and test on the absence probe (`launch_absence.sh`, ABSENT / ABSENT_START / ABSENT_END) and on a class-sorted stream. Falsifier: the mass ratio of the withheld component stays above .5 during the absence with no loss on the natives.
3. **Identify the ring_shift collapse episodes** (REQUIREMENTS 9.12): E14s, E19a and E21a each show an abrupt hq collapse (.9 -> .1-.2 in 50 steps) at different times; a pass needs the last five checks clean. Log the network group's tester events, the served/fast switch and the critic loss around the episode in a rerun of E19a (deterministic, ~15 minutes) before judging any parent rule (E20 p-weights, E21 distance limit, persistence) by this gate.
4. **A network-internal trigger for a distribution shift on a settled table** (REQUIREMENTS 9.4): the flagged fraction of the support test is already such a signal (all rows unsupported after a shift): reopen the tester ladders when it stays above one half for a few evaluations; fixture: native100 with a data shift at step 4,000 (a local script around the frozen native100 modules); falsifier: no re-arrival within 2,000 steps or one false trigger on the stationary native runs.
5. **Critic-side negatives instead of teleports** (REQUIREMENTS 9.10): replay the culled rows as extra fakes for the critic; hypothesis: the funnel around each mode (69-85% of random points feel an inward force in E14s, 54-72% in E19a, ~50% in E15a) is held up by far-field negatives, so the churn (14.9k re-draws per rotated run) falls without losing precision.
6. **A calibration reservoir larger than the table** (REQUIREMENTS 9.11) so the support test has power below N = 800, then the small hosts; and **finite pools with near-duplicates** (augmentations): group-split or dedupe by feature identity (the exact-copy guard covers only exact copies).
7. **Unequal masses and high z_dim with the support test** on the harness copies (`harness-bigN`): masses 1:10 with 0.02-0.1% components, z_dim 32/64 (the rank cap is tested only on synthetic tables).
8. **Run the lineage on the 8 custom hosts** (the harness engine refuses it: `components.py` `KNOWN_TRAINER_METHODS` / `KNOWN_RECIPE_FIELDS`); a decision for the harness owners, low priority now.
9. Derive the gate window and hold budget (A2); a dimension-free birth-death evidence; serve earlier by a state rule; amplitude-aware annealing (REQUIREMENTS 9.2, 9.7): unchanged from the E14s list.
