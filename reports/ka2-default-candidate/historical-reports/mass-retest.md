# rt_ka2_mass — ka2 vector_unequal_mass retest at host seeds {1,2,3}

Lane: own ka2's mass retest. Exact ka2 copy, no mechanism edits, 1 GPU worker.
GPU: GPU-72c1b506-891d-b8bc-b353-e020585e1c47 (cuda:0), FP32 deterministic TF32-off,
/tmp/pr38-default-env/bin/python, OMP_NUM_THREADS=1.
Frozen CUDA repo: qualify_a2_bounded_damp/20260925T041936Z-2509885/.../prepared/repos/cuda.
Candidate dir: /tmp/opencode/rt_ka2_mass (byte copies of lead-ka2-bundle + frozen k3p probe).

## Exact-copy verification (recorded hashes)
- mechanism.py 9f1d5eda4bb2f9e0e51832af77db27bd87fba214e29146b224d9963da390c478 (prefix 9f1d5eda MATCH brief)
- config.json a1475108a82f67a93e0cdcd793b920b0cc2b1e1ccf31285974adc3b341b2fca2 (MATCH K3P pinned)
- latent.py 197df6350f5295f7d396f7d3c821808be1d15168d6e5586a89ebfbd403586139 (MATCH K3P)
- response.py 7e71d60a343f9f47e1c16600279364f0482863ce116c00f4657355638615987d (MATCH K3P)
- probe.py e8653d7e450268310d9c7fc529262279f202a2cb10b3bf1470efc7763d4452bc (MATCH K3P gap-fill source)
- checkpoint.py 0185ba142eb5bffa1e28a47690b3526cab9e3a673a9c0ba0a6cd984c0c9444d9 (MATCH K3P)
- Floors .01/.05, horizon cap 1600, base LR .00425 (unchanged K3P schedules — labeled ablation per brief)
- ka2 rule per receipt: ka2-slowatk-fastrel, s=0.5 LR-decoupled, EMA 0.999 stateful asymmetric
  alpha (K_ATK 1/60, K_REL 0.5), surprise band 3.0/1.75, K3P guard C=5/200, 800-call pure-A warmup

## Seed convention (read this before trusting 1-3)
- The brief names seed-0 fixture cb5ddaeb, but that is the **mode_hold** fixture
  (gap-fill manifest fixtures.mode_hold). The **vector_unequal_mass** committed
  fixture is d5d6a1b39bf80354b060f0680381a85d27bd8c89f3b120b6357ab9f28e1e223a
  (manifest fixtures.vector_unequal_mass); every run here records it for seed 0.
- Frozen probe.py takes --initial-state but has NO host-seed flag for transfer tasks
  (unlike floor-seeds ring seedshift.py --host-seed). Frozen transfer setup hardcodes
  construction seeds 0/1/2 (+prior gen 0); eval seeds 990/991/402 fixed.
- Local capture/run shim (frozen probe.py + frozen host untouched; mechanism/latent/
  response/checkpoint exact ka2 copies; thresholds/steps/eval fixed):
  seed-0 capture reproduces d5d6a1b3 exactly; seeds 1-3 remap construction seeds
  {0:S, 1:S+1, 2:S+2} via torch.manual_seed/Generator.manual_seed wrapper at import
  time (eval 990/991/402 untouched), per-seed CPU fixtures (zero updates), then the
  frozen probe end-to-end on CUDA with --initial-state. Shim scripts (not run from
  a sibling cands/ dir; isolated /tmp/opencode/rt_ka2_mass):
  /tmp/opencode/rt_ka2_mass/seed_capture.py, /tmp/opencode/rt_ka2_mass/seed_run.py
- Fixture shas: s0 d5d6a1b3...e223a (== committed mass fixture), s1 bc95777e...69c65f,
  s2 f7e26f0c...15de151, s3 268c9eb7...c23c034.
- Seed-0 fidelity: this lane's s0 run is bit-identical to the t22_ka2 sweep artifact
  except per-observation wall-clock `seconds` (live metrics + full observation vectors
  equal; status FAIL 18/24, suffix 4, step-1000 cce 1.059).

## Verdict: ka2 FRAGILE on mass (stays, flagged) — passes at seeds 1 AND 3
- seed 0 (repo seed, reference): FAIL — 18/24 passing, terminal suffix 4 < 5.
  Final live all-PASS (hq 0.977) but gate FAIL on convergence suffix: single-observation
  covariance excursion at step 1000 (component_covariance_error 1.059 > 0.85; 4th-comp 3.78).
- seed 1: PASS — 15/24, suffix 8, first pass 200, stable_from 850, confirmed 1050, 182.5 s.
  Final live hq 0.982, cce 0.562, eigen 0.766, mass_tv 0.009, min_mass 0.975, sw1 0.014.
- seed 2: FAIL — 8/24, terminal suffix 0, 148.1 s. Final live FAIL on
  component_covariance_error 1.058 > 0.85 (eigen 0.436 PASS). Late collapse: step-800
  hq 0.451 excursion (min_mass 0.0), never re-stabilizes (fails at 950/1000/1050/1100/1150/1200).
- seed 3: PASS — 9/24, suffix 8, first pass 400, stable_from 850, confirmed 1050, 124.8 s.
  Final live hq 0.989, cce 0.358, eigen 0.752, mass_tv 0.016, min_mass 0.854, sw1 0.033.
- Per brief rule (passes anywhere → FRAGILE): **ka2 FRAGILE on mass (stays, flagged),
  mass DQ NOT confirmed.** 2/4 seeds pass including repo seed's two neighbors.

## Component covariance / eigen-ratio per seed (mass gate fails on min-eigenvalue ratio;
## here final-metric FAILs are all on covariance error, eigen PASSes everywhere at final)
| seed | status | passobs | suffix | final hq | final cce (thr ≤0.85) | final eigen (thr ≥0.15) | comp cce [c1..c4] | comp mass | failing obs (metric) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | FAIL | 18/24 | 4 | 0.977 | 0.193 PASS | 0.592 PASS | [0.156, 0.102, 0.186, 0.327] | [0.523, 0.334, 0.122, 0.021] | 50,100,150,200,250 + 1000 (cce 1.059) |
| 1 | PASS | 15/24 | 8 | 0.982 | 0.562 PASS | 0.766 PASS | [0.087, 0.075, 0.168, **1.918**] | [0.548, 0.292, 0.139, 0.020] | early eigen dips (400-550 eigen 0.06-0.17) + 350/800 cce spikes; clean suffix from 850 |
| 2 | FAIL | 8/24 | 0 | 0.990 | **1.058 FAIL** | 0.436 PASS | [0.345, 0.368, 0.413, **3.107**] | [0.580, 0.325, 0.080, 0.015] | 800 collapse (hq 0.451, eigen 0.000, minmass 0.0) + terminal cce fails 950–1200 |
| 3 | PASS | 9/24 | 8 | 0.989 | 0.358 PASS | 0.752 PASS | [0.238, 0.186, 0.262, 0.745] | [0.562, 0.303, 0.117, 0.017] | early eigen fails (100-350 eigen ≤0.04) + 450/600/650/700/750/800; clean suffix from 850 |

Where the seeds differ in that statistic:
- The rare 4th component (target mass 0.02) dominates final cce on every seed, but only
  seed 2's 4th-component error (3.107) pushes the mean cce over threshold; seeds 1
  (1.918) and 0 (0.327)/3 (0.745) absorb it. Seed 2 additionally under-occupies
  components 3-4 (mass 0.080/0.015 vs targets 0.130/0.020; min_mass 0.618).
- The brief's "fails on minimum-eigenvalue ratio" does not match these runs: final
  eigen-ratio PASSES on all four seeds (0.44–0.77, margins +0.29..+0.62). Eigen-ratio is
  instead the *early* binding constraint (seeds 1/3 fail eigen at steps 100–550 while
  covariance already passes), whereas the *late/terminal* binding constraint is the
  covariance error (seed-0 step-1000 spike; seed-2 terminal run). Seed 2 is the only run
  where eigen also collapses mid-run (steps 800–1050 eigen 0.00–0.03) alongside the
  mass collapse.
- Useful negative carried forward: ka2's fast-release re-anchor does not protect the
  rare-component covariance against single-observation excursions (seed 0) or full
  late collapse (seed 2), yet the same rule passes cleanly on seeds 1/3 — seed-fragile,
  not uniformly broken.

## Replay commands
CAND=/tmp/opencode/rt_ka2_mass
CUDA_REPO=/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda
FIXDIR=/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures
# seed-0 reference (frozen probe, exact committed fixture):
CUDA_VISIBLE_DEVICES=GPU-72c1b506-891d-b8bc-b353-e020585e1c47 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -u $CAND/probe.py --repo $CUDA_REPO --task vector_unequal_mass --output /tmp/opencode/rt_ka2_mass/out/s0 --config $CAND/config.json --backend cuda --initial-state $FIXDIR/vector_unequal_mass/initial-values.pt
# per-seed captures + runs (shim; frozen probe/host sources unedited):
/tmp/pr38-default-env/bin/python -u $CAND/seed_capture.py <S> /tmp/opencode/rt_ka2_mass/fixtures/cap<S>
/tmp/pr38-default-env/bin/python -u $CAND/seed_run.py <S> /tmp/opencode/rt_ka2_mass/fixtures/cap<S>/initial-values.pt /tmp/opencode/rt_ka2_mass/out/s<S>

## Artifacts
- /tmp/opencode/rt_ka2_mass/out/s0/result.json (FAIL, seed-0 fidelity vs t22 artifact)
- /tmp/opencode/rt_ka2_mass/out/s1/result.json (PASS, 182.5 s)
- /tmp/opencode/rt_ka2_mass/out/s2/result.json (FAIL, 148.1 s)
- /tmp/opencode/rt_ka2_mass/out/s3/result.json (PASS, 124.8 s)
- /tmp/opencode/rt_ka2_mass/fixtures/cap{0,1,2,3}/initial-values.pt (+shas above)
- t22 reference: /tmp/opencode/t22_ka2/out/ka2-toy-vector_unequal_mass/result.json (read-only)

## Remaining failures / non-claims
- ka2 mass is seed-fragile (2/4 pass), NOT robust: repo seed FAILs; seed 2 collapses.
- No mechanism edits, no other gates run (lane runs ONLY vector_unequal_mass per brief).
- Transfer host seeds here are construction-seed offsets (documented shim); ring-host
  seed semantics ({0..3} via train_mode_hold seed kwarg) do not exist for transfer tasks.
