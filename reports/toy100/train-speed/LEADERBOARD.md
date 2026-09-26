# Toy train-speed leaderboard (live — updates with winners)

Goal: fastest wall-clock full-gate training of the frozen simpler22 toy suite
with **identical** numerics gates. Every promotion is a full 22/22 regrade
PASS at frozen seeds/thresholds that is faster than the previous best.
Ranked by `train_seconds` on the pinned CPU profile (1 thread, AVX2,
seed 1234). Timing-only runs without a full-gate PASS are screening
evidence, not winners. Append rows; never rewrite history.

| Rank | Candidate (config / change) | 22/22 | Train seconds ↓ | vs baseline | Evidence | Status |
| --- | --- | :---: | ---: | --- | --- | --- |
| 1 | **Baseline**: `constraints_simple_regularization.json` (exact `b_cap` every step, foreach off) | PASS | 374.8 (toy100 ×3 incl. eval+GIF; per-problem 98.8 / 99.9 / 176.1) | — | `reports/toy100/simpler22/`, `reports/toy100/README.md` | **Current winner** |
| — | *Your candidate here* | — | — | — | `artifacts/toy-suite/<candidate>`, `<candidate>.live.log` | — |

## Screening log (falsified or non-qualifying — kept so rounds don't repeat them)

| Round / lane | Candidate | Verdict | Key numbers |
| --- | --- | --- | --- |
| R1 / optimizer_kernels ($0.09) | `fused_adam: false → true` | FAIL, no promotion | grid100 48/100, rotated100 2/100, staggered100 100/100; 20/22 regrade. Optimizer ≈2.7% of step time — no win available in this lane. |
| R1 / bcap_lazy_schedule ($0.07) | Lazy exact `b_cap` (`reg_every=4`, `=2`, rescaled coeff) | FAIL, no promotion | lazy-4: 0/3 (final modes 13/100/91); lazy-2: 2/3, rotated100 collapses to 5 modes (hq .08). Penalty schedule is load-bearing. foreach (non-fused) Adam preserves the gate. |
| R1 / eval_io_overhead ($0.08) | Lean logging, `--no-render`, wider eval intervals | No win alone | Eval/I/O ≈1.5% of wall clock (3-problem qual: train 857.4s / eval 12.5s). Render skip saves ≈5.3s per problem GIF. |
| R1 / host_data_pipeline ($0.09) | Cached static mode centers + in-place noise scaling (`benchmarks/toy100/problems.py`, +15/−1) | PASS but marginal (not promoted) | RNG-order unchanged, bit-identical on CPU/CUDA; sampling 0.140 → 0.072 ms/draw; saves ≈1s per 7k-step problem. Candidate for the micro-win stack. |
| R1 / graph_compilation ($0.24) | `torch.compile` on b_cap / D / G paths | No candidate qualified | `aot_autograd does not support double backward` (torch 2.13+cu126) — compile cannot touch the exact-penalty path. P3 CUDA run passed toy100 gate but off-profile (pinned profile is CPU). |
| R2 / alternating_penalty ($0.05) | Alternating real-only/fake-only exact `b_cap` at ×2 weight | FAIL, stop after one bounded test | Parity unit check PASS; grid100 final 3/100 (hq .061), peaked 14 modes @500 → 1 @1000. Held-out side drifts immediately. |
| R2 / shared_forward_bcap ($0.09) | Share one `D(real)`, `D(fake)` forward between Rp loss and penalty | FAIL (parity), disqualified as specified | Logit values identical, but 2nd-order accumulation order differs: param-grad maxdiff 1.19e-07 (6344 params > 1e-9), compounding 3.8e-06 @1 step → 0.25 @30 steps. Joint input-grad variant is exactly 0.0 but slower (keeps all 4 forwards). |
| R2 / profile_first ($0.10) | Measured hotspot split, then top hotspot only | No promotion (profiling-only, tree clean) | Pinned CPU 102.1 ms/step: D double-backward+opt **71.7%**, b_cap input-grads 13.7%, D loss-forwards 5.5%, G fwd 4.8%, G bwd+opt 3.5%, sampling 0.2%, EMA 0.1%. Second-order work ≈79% of step. Independently confirmed shared-forward's 30% probe speedup but trajectory drift. Near-miss data banked for R3 (fused input-grad, foreach). |
| R3 / micro_stack ($0.11) | Stack: foreach Adam + cached centers + uncompressed NPZ + buffered events | 22/22 PASS, **REJECTED after clean A/B** (see below) | Paired single-problem −5.2% did not replicate; full-suite head-to-head +2.2% slower. |
| R3 / gate_not_bitwise ($0.13) | Re-qualify near-misses on gates not bits | No promotion | Stack 22/22 PASS but not faster on that box; fused Adam diverges (37–42 modes @3000, killed). K=30 foreach-vs-base bit-identical 0.0. |
| R3 / penalty_subsample ($0.06) | Batch-subset b_cap ×2 weight | FAIL, stopped after bounded test | grid100 63/100, hq .78 from step 1000; ~22% time saving irrelevant without quality. |

Round spend: R1 ≈ $0.68, R2 ≈ $0.23, R3 ≈ $0.30. No promotion yet: the baseline
remains the winner. Batches: `/ml2/hypergan/gan-attempts/toyspeed-20260926T040626Z`
(R1), `/ml2/hypergan/gan-attempts/toyspeed-20260926T065716Z` (R2),
`/ml2/hypergan/gan-attempts/toyspeed-20260926T073425Z` (R3).

## Pending promotion (needs a clean full-suite timing on a quiet box)

**Micro-stack — REJECTED after clean paired A/B (2026-09-26).** Re-ran the full
22-suite head-to-head, sequential, pinned 1-thread AVX2 profile, both 22/22
PASS: baseline train 321.4/338.8/351.1s (sum 1011.3s) vs stack
343.7/335.7/354.0s (sum 1033.4s, **+2.2% SLOWER**). The agent's single-problem
−5.2% did not replicate — contention luck, plus foreach dispatch overhead on
the tiny optimizer slice (profiled: base 74.9ms vs foreach 75.1ms per opt
step). The stack's idea is gate-preserving but not faster. Evidence:
`/tmp/toyspeed-promote/{base22,stack22}` + live logs, worktree
`/tmp/toyspeed-promote/repo` (detached @64d0d744 + stack.patch).

## Update contract

- Each winner adds one row above plus its evidence paths (run dir, live log,
  regrade output). Keep the old rows.
- A winner must attach: exact config (or diff vs baseline), full
  `benchmarks.toy_suite run` + `regrade` commands, `train_seconds` with the
  train/eval/IO split, and confirmation that seeds, thresholds, budgets and
  the pinned CPU profile are unchanged.
- Screening rows (partial gates, single-problem timings) go in attempt
  `result.md` files, not in the table, until they earn a full-gate PASS.
