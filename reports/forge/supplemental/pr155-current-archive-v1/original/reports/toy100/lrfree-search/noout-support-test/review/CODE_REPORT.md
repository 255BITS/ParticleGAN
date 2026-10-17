# Independent code and integration review of the support test (pkg-E17), returned as a message on 2026-09-29

Reviewer: a general-purpose subagent working read-only on `pkg-E17`, CPU only; evidence, mutants and added tests are in `code/` next to this file
(`mutation_table.txt`, `survivor_to_test.txt`, `test_E17_extra.py`, `adversarial.out`, `fuzz_E17.out`, `replay_*.out`, `probes/`; the bulky `mut/logs` were left out of the archive).
The report is reproduced as received (HTML escapes restored).

---

No `blocking` finding. Nothing in the handoff is false, no result looks wrong, and by code read only critic features of G(z_i), the real reservoir, the table's own z and the private RNG stream are used. R = `/ml2/hypergan/gan-attempts/noout-20260928/design/review_E17_code`. `tests/test_E15.py` passes on pkg-E17 (ALLOK, 14 s, CPU; its CUDA part is skipped). I touched no GPU and nothing outside R.

## Findings by severity
1. **HIGH: the "cannot allocate O(N²)" check fails, for N ≳ 100k only.** The ball rule and `_isolated` are chunked (largest block 16,777,000 entries, about 0.34 GB). The stale-site `cdist(anchor, sites)` at `birth_death.py:446` is not chunked and costs 5·N·S bytes, with S = 2(ordinary + isolation moves).
   - Measured 196 / 769 / 3,057 MB at (N,S) = (20k,2k) / (40k,4k) / (80k,8k) (`probes/stale_mem.out`).
   - At the 5% guard this is 0.5·N² bytes: about 0.2 GB at N=20k and about 20 GB at N=200k (12 GB at 3% strays).
   - E14 had the pattern, but isolation adds up to 0.1·N sites.
2. **MEDIUM: the ball rule is not dimension-free.** For a table z~N(0,I_d) at N=20k, the median share of unflagged rows inside "2× nearest" is 0.02% (d=2), 0.06% (4), 0.75% (8), 21% (16) and 100% (64) (`adversarial.out`). At d≥16 the parent is a uniform table row, which is E15's rule that failed. The native z_dim=4 is fine, and clustered ring tables at z_dim 4 and 64 behave: 60/60 strays re-drawn onto modes.
3. **MEDIUM: the float32 shortlist in `_knn` silently degrades.** This is shared with the ordinary path. Once (centred feature scale)/(local spread) ≳ 1e3, the k-th distance is wrong for 68% of queries (d=2, 1e3), 53% (d=8, 3e3) and 100% (1e4) (`probes/p2_shortlist.out`).
   - Nothing logs it.
   - It relies on TF32 being off, which the harness sets and the package does not enforce.
   - Calling `_isolated` uncentred with offset 1e4 changes 761 of 838 flags.
4. **MEDIUM: detectability floor.** BH needs at least 2/Q = 40 rows tied at the smallest p. More exactly it needs 40·(1+c), where c is the number of calibration points scoring above the strays. The guard allows at most 0.05·N rows.
   - The mechanism is inert below N=800 (N=799: 39 allowed, 40 needed). N=20/32/200 run 120 steps identical to flag-off.
   - At N=20k only strays beyond all but at most 24 of the 10,000 calibration points can ever be acted on.
   - The boundary is tested: 16 tied rows are flagged, 15 give none (m=2400, |R2|=3000).
5. **LOW: constants and clock (A9).** The floor 1e-3×median radius is in neither the handoff nor the docstring. The ball factor 2 is whitelisted by the lint as "trivial". Q is reused for two new roles (the BH level in `_isolated` and the guard level). `admissibility_lint.py` on `E14_to_E17.patch` shows one clock hit (`iso_log` uses `completed_steps`, diagnostic only) and the literals 1e-3, 30 and 2^24. L3–L5 are clean.
6. **LOW: harness diagnostics ignore isolation teleports.** `harness/screen.py:1054` reads `counters['moves']` (ordinary only) for `interval_moved`, so isolation re-draws count as continuous motion in `affine_motion_v1`. It is diagnostic only. Also `last['moves']` includes isolation moves while `counters['moves']` does not.
7. **LOW: tie sensitivity, so CPU and GPU flags can differ on tie-rich features.**
   - 9 of 75 integer-lattice trials differ from a brute-force reference, including one cliff of 182 vs 0 flagged. The k-th-neighbour set is ambiguous under exact ties, which moves one calibration score to 1000.
   - 0 of 225 Gaussian, duplicated and heavy-tail trials differ (`fuzz_E17.out`, `probes/p5_lattice145.out`).
   - With small jitter the residual lattice differences (d=1 trials) are the finding-3 shortlist limit.
8. **INFO:**
   - The explicit `S/W/n[moved]=0` is dead code, because the stale reset already covers each moved row's own old site. It matters only if the radius is NaN (mutant M43).
   - The stale set differs from a float64 oracle for 6 of 1,940 rows (float32 boundary flips).
   - The `recipes.py` comment says "exact null", while the docstring says conservative.

## 1. Correctness read: all confirmed
- `searchsorted` is side=left, so `len − idx` = #(null ≥ s) and ties are conservative.
- The BH tie rule `p ≤ ps[last pass]` is correct.
- Odd reservoirs work: R1 gets the extra row, and mask == reference at |R|=2401.
- The guard `0 < n ≤ QN` acts at exactly 1000 of 20,000 and not at 1001.
- Ordinary children are excluded from both dead and keep.
- Parents are never flagged or moved: 0 violations in 289 fuzz trials plus crafted cases.
- `tester.rebase` and `row_evidence.reset` receive exactly the moved rows.
- The flag-off path draws no random numbers. The flag-on path draws only when acting, after the ordinary draws, from the private stream.
- Cross-flag checkpoint loads are refused (recipe mismatch). `iso_log` is diagnostic-only and not checkpointed.

## 2. Mutation testing (`mutation_table.txt`, `survivor_to_test.txt`)
I tested 57 mutants covering every item on your list plus a few more.
- `test_E15.py` catches 23 of 57. It lets 34 survive, all of them my mutants of these kinds:
  - BH rank shifts (M01, M02), calibration set or split (M06, M18), tie side (M07), p without the +1 (M09), median (M10, M11), non-leave-one-out radius (M12), k off by one (M13, M14, M22, M55), floor (M16, M17, M56) and float32 recompute (M15).
  - Every ball, exclusion and chunk mutant (M28–M31, M33–M38).
  - Guard `<` vs `<=` (M24), uncentred features (M20), stale-site omissions (M41, M42), history copy (M46) and global RNG (M52).
- **New tests:** `test_E17_extra.py` (31 checks, 18 s), run in addition to `test_E15.py`, catches 52 of 57 alone, and 57 of 57 together with `test_E15.py`. The five it misses are dry run, counter restore, flag-off RNG and the two recipe checks, all of which `test_E15.py` catches. It contains:
  - A differential test of `_isolated` against an independent float64 reference, plus the BH boundary (16/15 tied rows), scale invariance, a dtype contract and a floor case.
  - Crafted `_isolation_pick` tables at N=20,000.
  - Trainer-level checks including a float64 stale oracle, hand-over to the tester, and history/EMA/optimizer copies.
- The last survivor was M43, dead code (finding 8). It is only killed by the contrived NaN-radius test.

## 3. Determinism, memory and time
- **Determinism:** No new op needs a deterministic-mode exemption. In this torch (2.13.0) only `median` with indices, `cumsum` on float, `histc` and `bincount(weights)` throw on CUDA, and the code uses none of them (`positive.median()` has no indices output; the local median uses `sort`). The harness sets deterministic mode and TF32 off. I could not run CUDA, so this is a docs and library-strings audit.
- **Precision:** Features are float64 with a float32 shortlist, and `searchsorted` gets float64 on both sides.
- **Memory:** `_knn` chunks are 2048×n_pts×4 B: 0.16 GB at N=20k, 1.6 GB at N=200k, half of that in `_isolated`. The pick block is about 0.34 GB at any N. The stale matrix is the outlier (finding 1). Measured RSS peak at N=20k with 600 moves: +381 MB flag off, +420 MB flag on.
- **Time:**
  - Existing passes per evaluation: 6 N² (4 in feature space, 2 in latent space).
  - Added: 3 calls totalling 1.0 N², plus a tiny jitter search, so +25% of feature-space cells and +17% of all.
  - Measured CPU wall time 1.13→1.35 s, 4.39→5.08 s and 16.6→19.4 s at N=5k, 10k and 20k.
  - GPU time not measured; the logged native run times are dominated by GPU sharing.

## 4. Integration
- With the flag on, `serve_average`=4, gate True and False, natural and forced-served average, the full `state_dict` was bit-exact against an uninterrupted run for checkpoints at steps 40, 43 and 72. That is 12 of 12 (`replay_natural.out`, `replay_forced_serve.out`), with 3 acting evaluations and 179 isolation moves.
- Flag-off pkg-E17 equals pkg-E14 bit for bit in both replay settings (3 ordinary moves) and in an 800-step N=200 run with 115 ordinary moves.
- A flag-on trainer whose isolation never acts equals flag-off. This ran on the ring table with one evaluation at zero flagged and four guard-blocked; a stricter case is in `test_E17_extra.py`.

## 5. Adversarial (`adversarial.out`)
- **Tiny N (N=5):** refused by the existing "at least 6 particles" check. N=6 to 32 are inert and identical to flag-off.
- **Duplicate rows:** duplicates are flagged together and handled without error.
- **Non-finite inputs:** a NaN or inf reservoir row skips the evaluation via the ordinary `dim_skips` path.
- **NaN table rows:** NaN table rows are cloned from `keep[0]` and do not poison healthy rows.
- **Dead critic:** a dead critic skips.
- **Refused head:** it raises the same ValueError as E14, at the first evaluation.
- **Guard and empty sets:** everything flagged is blocked by the guard. Everything flagged with the guard lifted, all rows already moved, and an empty keep set all return empty answers.
- **Other configurations:** a float64 trainer and odd N both work.

## Next experiments to break it
1. GPU at N=200k with z_dim 16/64 and 5% planted strays: peak memory (expect 20 GB or an OOM) and parent locality against E15.
2. Native-critic feature dumps: compare the float32 shortlist with float64 brute force, and CPU against GPU flags after quantizing or saturating features (ties).
3. Null calibration: 200 true-null evaluations for the false-flag rate, then plant strays with c = 0, 5, 24 and 25 higher calibration outliers to check the 40·(1+c) cliff.

---

What was done about it (this archive): finding 1 -> E19 chunks the stale-site check; finding 2 -> E19 rank cap k^2 (and E20/E21 tried other parent rules); finding 4 -> stated, the mechanism is inert on the tiny tables by design; the added tests are adopted as `tests/test_E17_extra.py` (E22: `tests/test_E22_extra.py`, which only changes the two duplicate checks to the E22 duplicate guard).
