# ParticleGAN GPU leaderboard: one pinned GPU (2026-09-24, 22:42 MT)

**Source sha:** PR #139 `codex/epsilon-gan-followup` at
**`adafbe3f9d38f8991b2f9e3756393a77e7fe5470`** (head as of 21:26 MT, "Add probe-fast.py").
That commit is tooling only (a faster probe with identical receipts), not a new base. It sits on top
of the device fix `b664b65` and the direct-particle base `1af94be`.

**GPU, pinned for every row:** NVIDIA GeForce RTX 4090 (RunPod Serverless).
- Software: torch 2.14.0+cu130, CUDA 13.0, cuDNN 9.24 (92400), drivers 580.159 / 580.178 / 595.91.
- Math settings: FP32, deterministic algorithms, TF32 off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, one CPU thread, seed 0.
- Why pin: results change across GPU models. On the A6000 the original recipe scores 16/22; here it scores 14/22. #84's ring flipped between an A4000 and a 4090.
- Only compare rows inside this table.
- Across 4090 hosts with different drivers, same-GPU repeats were bit-identical (#120 ring, #84 ring).

**Old CPU ranks are superseded.** Nothing here reaches 22/22, and nothing here is a production claim.
- Every training step ran on CUDA. The device proofs count CUDA Adam updates, with 0 CPU parameter steps.
- CPU-built initial weights copied to CUDA (the #139 bundle fixtures) are used where the bundle declares them.

## How rows are scored

- **Ring** = frozen `mode_hold` gate: 8/8 modes, HQ ≥ .90, and the last 5 checks stable.
- **Unequal mass** = frozen `vector_unequal_mass` gate.
- **Full suite** = the 22 frozen toys. "X/22" counts PASS; unrun toys give no credit.
- **Stay** = constant-rate ring continued to 2400 updates, with 120 checks from update 1210 to 2400. A PASS needs 120/120.
  The in-repo cohort has no 2400-stay, so for those rows this column shows their own hold gate: first 200-check convergence after update 1200, then a 1200-update hold.
- **Rank key** (main track), in order:
  1. gates passed (ring, unequal mass)
  2. full-suite PASS count
  3. stay checks passed
  4. ring stable suffix
  5. modes, then HQ

  Ties share a rank.

**Harnesses.** All three use the frozen specs from the repo; none is hand-patched.
- **#139 research bundles:** `reports/toy100/{direct-particle-base,dimension-rms-base}` probes.
  - Commands are the ones in `replay.py`: probe-fast.py for direct-particle, probe.py for dimension-RMS.
  - Fixtures follow each bundle's replay rule. Toys without a fixture use native CUDA init.
  - Not run: the three native 100-mode toys. The bundle probes only load the 19-toy transfer declaration, and fail with "task not in declaration".
- **Original recipe:** the `gpu-known-winner-control` worker. **In-repo cohort:** the `gpu-leaderboard` worker.
  - Same archive and per-file sha256 checks as their `replay.py`.
  - Their A6000/torch-2.13 profile guard does not apply here; this is declared as a separate profile, `cuda_4090_t214`.
- **CL bets #82–#144:** the `gan_followup_probe.py` cold and stay phases in each PR's pinned tree (the #107 base plus that PR's files).
  - PR #139's device-fix commit `b664b65` is cherry-picked into each pin wherever its parent file is byte-identical, taking #139's files verbatim.
  - PRs that carry their own `observation.py` or `continuous_probe.py` keep them unmodified: #123, #125, #130, #133–#138, #142.
  - A launcher calls `apply_device_policy("cuda")` and replaces the CPU-only `canonical_env` check.

## Main track: GAN-native, ranked

| rank | PR / sha | idea | track | GPU | ring: modes · missing · HQ · suffix | unequal mass | full suite | stay | keep/kill | receipt |
|---|---|---|---|---|---|---|---|---|---|---|
| 1= | #139 `1af94be` direct-particle base (unchanged at adafbe3) | Rp logistic GAN, real R1 + fake RMS b-cap; direct sample-particle Adam β=(0,.9), LR×(1+clamp cos(g_t,g_prev)) | main | RTX 4090 | **FAIL** 4/8 · [1,3,4,7] · .9995 · 0 | **PASS** (min mass ratio .838) | **17/22** (fails: mode_hold, img_stripes2; not run: grid100/rotated100/staggered100) | not run: bundle has no constant-rate continuation | keep (co-leader; selected base) | E2 |
| 1= | #139 `74fac3a` dimension-RMS base | same critic penalty, no direct-particle response | main | RTX 4090 | **FAIL** 4/8 · [1,3,4,7] · .9995 · 0 | **PASS** (.838) | **17/22** (fails: two_pole mean_abs .104, mode_hold; not run: 3 native) | not run | keep as control (ties; loses two_pole) | E2 + I |
| 3 | shared_rms (in-repo cohort) | shared RMS optimizer, active rates | main | RTX 4090 | **PASS** 8/8 · none · .9866 · 11 | FAIL (min mass ratio .248) | **12/22** | hold: converged at 2061, FAIL after 10 checks | keep (ring pass) | G |
| 4 | H (in-repo cohort) | H-stability optimizer, active rates, AE/unused aux losses off | main | RTX 4090 | **PASS** 8/8 · none · 1.0 · 6 | FAIL (.708) | **12/22** | hold: NOT_CONVERGED (longest streak 136/200) | keep (ring pass) | G |
| 5 | #120 `cc02a5dd0b` | reach-stall + recover | main | RTX 4090 | **PASS** 8 · n/r · .9966 · 5+ (repeat bit-identical) | N/A: CL factory is not wired into the vector host | not run: CL harness has no suite wiring | **79/120** | keep (only CL ring pass) | F + H |
| 6 | #139 `54f1f22` base: original 22/22-CPU recipe (`constraints_simple_regularization`) | decaying rates, noise, original aux host terms | main | RTX 4090 | FAIL 5/8 · [5,6,7] · .753 · 0 | FAIL (.308) | **14/22** (A6000 record: 16/22) | not run | keep as reference | E |
| 7 | shared_column_rms (in-repo cohort) | shared column-RMS optimizer | main | RTX 4090 | FAIL 8/8 · none · .9753 · 2 | FAIL (.732) | 10/22 | hold: converged at 1433, FAIL after 43 | kill | G |
| 8 | eps_net_1m (in-repo cohort) | epsilon-net optimizer | main | RTX 4090 | FAIL 7/8 · [6] · .807 · 0 | FAIL (.610) | 9/22 | hold: converged at 1770, FAIL after 233 | kill | G |
| 9 | #84 (#149 `2ce2a05b84`) | stall-reach, post-acquire EMA D | main | RTX 4090 | FAIL 8/8 · none · .929 · 4 (repeat bit-identical) | N/A: not wired | not run | **116/120** (best stay; final 8, .9988; min 8 modes) | keep (best stay) | F + H |
| 10 | #138 `57f2bd5a67` | slope-utilisation reach, post-acquire G 0.125 | main | RTX 4090 | FAIL 8/8 · none · .9993 · 2 | N/A | not run | **112/120** | keep | F |
| 11 | #144 `c1d1a2afe7` | stall, cf 0.25 | main | RTX 4090 | FAIL 8 · n/r · .9966 · 2 | N/A | not run | **112/120** | keep (watch) | F |
| 12 | #141 `b270fc99f0` | delayed G budget, own-curvature 0.125 | main | RTX 4090 | FAIL 8 · n/r · .9966 · 2 | N/A | not run | 108/120 | kill | F |
| 13 | #121 `9262f8e3ef` | reach-stall S-game | main | RTX 4090 | FAIL 8 · n/r · 1.0 · 3 | N/A | not run | 107/120 | kill | F |
| 14 | #140 `ef4084a46d` | delay G 0.5 | main | RTX 4090 | FAIL 8/8 · none · 1.0 · 2 | N/A | not run | 107/120 | kill | F + H |
| 15 | #143 `3384976cc6` | hold window 15 | main | RTX 4090 | FAIL 8/8 · none · 1.0 · 2 | N/A | not run | 103/120 | kill | F + H |
| 16 | #142 `d13ae8149f` | delay + extra D | main | RTX 4090 | FAIL 8/8 · none · 1.0 · 2 | N/A | not run | 66/120 (final 1 mode) | kill | F + H |
| 17= | #107 `d5ec127b43` | reach-stall | main | RTX 4090 | FAIL 8/8 · none · 1.0 · 2 | N/A | not run | 51/120 | kill | F + H |
| 17= | #136 `0cc2c1d4b1` | reach-stall G β=0 | main | RTX 4090 | FAIL 8/8 · none · 1.0 · 2 | N/A | not run | 51/120 | kill | F + H |
| 19= | #130 `b2f1a35f38` | reach-stall rollback | main | RTX 4090 | FAIL 8 · n/r · .9954 · 2 | N/A | not run | 51/120 | kill | F |
| 19= | #134 `690171e2ff` | reach-stall advantage gate | main | RTX 4090 | FAIL 8 · n/r · .9954 · 2 | N/A | not run | 51/120 | kill | F |
| 21 | #133 `5dd333f3ff` | reach-stall trust-open | main | RTX 4090 | FAIL 8 · n/r · .9954 · 2 | N/A | not run | 38/120 | kill | F |
| 22 | #117 `5def098a0b` | translation trust | main | RTX 4090 | FAIL 7 · n/r · 1.0 · 0 | N/A | not run | 28/120 | kill | F + H |
| 23 | #137 `f26662a1a1` | common-mode null | main | RTX 4090 | FAIL 8/8 · none · .9988 · 0 (live HQ .895) | N/A | not run | 6/120 | kill | F + H |
| 24 | #125 `7bf2fc8279` | reach-stall G half | main | RTX 4090 | FAIL 8/8 · none · 1.0 · 0 (live HQ .589) | N/A | not run | not run: below top by gates | kill | F |
| 25 | #123 `d6095d84d3` | reach-stall extra D | main | RTX 4090 | FAIL 7/8 · [7] · .827 · 0 | N/A | not run | not run | kill | F |
| 26 | #122 `da1885d9fa` | reach-stall trust-fall | main | RTX 4090 | FAIL 6 · n/r · .543 | N/A | not run | not run | kill | F |
| 27 | #93 (#149) | pr93 | main | RTX 4090 | FAIL 5 · n/r · .804 · 0 | N/A | not run | not run | kill | F |
| — | baseline (#149, constant-rate GAN) | reference | main | RTX 4090 | FAIL 5/8 · [3,5,6] · .610 · 0 | N/A | not run | not run | reference | F |
| 28 | #119 `ee2c168e51` | reach-stall DD exit | main | RTX 4090 | FAIL 1 · n/r · .021 | N/A | not run | not run | kill | F |
| 29= | #82 (#149) / #124 `6824707ceb` / #126 `caabea0b55` / #128 `b50d5f7ce1` | pr82 / zero-mean / mean-restore / EMA | main | RTX 4090 | FAIL 0/8 at 1200 | N/A | not run | not run | kill | F |

Column notes:
- The idea column is a one-line summary taken from each method or config name.
- "n/r" means missing-mode ids are not recorded. Older probe versions log only the last five 50-update checks; for those rows, "5+" means all five logged checks pass.
- Unequal mass is N/A for every CL bet: their mechanisms are wired only into the mode_hold/trajectory hosts, and the factories raise KeyError on vector tasks. Porting each mechanism into the vector runner is not a mechanical change.

### Diagnostics (not ranked)

- **Original recipe with CPU-built initial weights on CUDA** (`cpu-recipe-gpu-port --profile cuda_cpu_init`, RTX 4090): 3/6 on its six-toy diagnostic.
  - mode_hold **PASS** (8/8, .9995). The A6000 record says FAIL.
  - img_blobs4 and img_intensity2 PASS.
  - trajectory, img_bars4 and vector_unequal_mass FAIL.
  - Receipt: E.
- **In-repo PR107/PR140/PR143 adapters** (`gpu-leaderboard`, one byte-identical archive): ring FAIL for all three, 8/8 · .9795 · suffix 2, identical results. Trajectory FAIL. Hold:
  - PR107: NOT_CONVERGED.
  - PR140: converged at 2291, FAIL after 6 checks.
  - PR143: converged at 3032, FAIL after 39 checks.
  - Receipt: G.

## Side track (can never lead)

Coverage, anchor, forward-KL, quota and Chamfer ideas: #85, #86, #88, #94, #99. **Not run:** none has a GPU entry point, a probe method, or a declared 22-toy adapter. They stay unranked.

## Skipped, with reasons

- **#129, #132, #135:** crash under the in-repo device fix. `reports/toy100/pr84_reach_candidate.py` builds an explicit `torch.Generator(device="cpu")`, which `b664` deliberately leaves alone.
  - A one-line fix in those PRs is needed.
  - Earlier runs with a local patch are not ranked here: all ring FAIL.
- **#81:** `torch.linalg.lstsq` with a non-`gels` driver is CPU-only. Changing the solver would change numerics.
- **#100–#105, #113:** bespoke CPU scripts, no probe method.
- **#116:** mechanism not built.
- **Warm phase (all CL bets):** the warm probe forks a CPU-only torch process.
- **Full suite for CL bets:** not wired, so N/A.
- **Native 100-mode toys for the #139 bundles:** the bundle probes don't support them.
- **2400-stay for the #139 bundles and the original recipe:** no constant-rate continuation exists. The direct-particle README notes the response history must be checkpointed first.
- **shared_rms_average2:** a 3-toy adapter; left out of job G to stay under the payload limit.
- **Stay for rank 24 and below:** outside the top by gates.

## Receipts (RunPod Serverless, RTX 4090; box paths under `/workspace/runpod-toy/results/`)

| key | job | contents | path |
|---|---|---|---|
| E | `392fc992-…-u2` | original recipe (22 toys), cpu-init diagnostic | `adafbe3f9d38f8991b2f9e3756393a77e7fe5470/20260924-213141` |
| E2 | `74eb33bd-…-u1` | direct-particle + dimension-RMS (19 toys each) | `adafbe3f9d38f8991b2f9e3756393a77e7fe5470/20260924-213626` |
| I | `655b4a8c-…-u2` | dimension-RMS, replay.py fixture fallback (6 toys) | `adafbe3f9d38f8991b2f9e3756393a77e7fe5470/20260924-223851` |
| F | `63da7759-…-u2` | CL rings (28) + 8 stays | `jobF-cl-139dev/20260924-214544` |
| G | `0968b611-…-u1` | in-repo cohort: 4×22 toys, 7 holds, PR107/140/143 adapters | `jobG-cohort/20260924-221443` |
| H | `c7a0cb8e-…-u2` | 8 more stays + 2 same-GPU ring repeats | `jobH-cl-stay/20260924-223022` |
