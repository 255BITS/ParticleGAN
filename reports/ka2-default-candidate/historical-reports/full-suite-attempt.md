# t22_ka2 — ka2 full-22 verification: DQ'd at vector_unequal_mass

Lane: own ka2's full 22 (exact copy, no new proposals). STOP rule applied: any toy FAIL => DQ, stop sweep.
GPU: GPU-72c1b506-891d-b8bc-b353-e020585e1c47 (cuda:0), 1 worker, FP32 deterministic TF32-off, /tmp/pr38-default-env/bin/python.
Frozen CUDA repo: qualify_a2_bounded_damp/20260925T041936Z-2509885/.../prepared/repos/cuda + per-task fixtures.
Candidate dir: /tmp/opencode/t22_ka2/cand-ka2 (byte copies of lead-ka2-bundle + frozen probe/checkpoint/native drivers).

## Exact-copy verification (recorded hashes)
- mechanism.py sha256 9f1d5eda4bb2f9e0e51832af77db27bd87fba214e29146b224d9963da390c478 (prefix 9f1d5eda MATCH)
- config.json a1475108a82f67a93e0cdcd793b920b0cc2b1e1ccf31285974adc3b341b2fca2 MATCH K3P pinned config
- latent.py 197df6350f5295f7d396f7d3c821808be1d15168d6e5586a89ebfbd403586139 MATCH K3P
- response.py 7e71d60a343f9f47e1c16600279364f0482863ce116c00f4657355638615987d MATCH K3P
- ka2 mechanism: rule ka2-slowatk-fastrel, s=0.5 LR-decoupled, EMA decay 0.999 w/ stateful asymmetric alpha (K_ATK 1/60, K_REL 0.5), surprise band 3.0/1.75, K3P guard C=5/200, 800-call pure-A warmup
- Floors .01/.05, horizon cap 1600, base LR .00425 (unchanged K3P schedules — labeled ablation)

## Verdict: ka2 DQ'd from the final board
- mode_hold: PASS, 93.8 s (first_pass 500, confirmed 700). Receipt rule ka2-slowatk-fastrel.
- vector_unequal_mass: FAIL, 143.1 s. Final live metrics all PASS (hq 0.977, thresholds met) but convergence gate FAIL: 18/24 passing observations, terminal passing suffix 4 < required 5. Per-observation: passes steps 300–950 except a step-1000 excursion (component_covariance_error 1.059 > 0.85; 4th-component error 3.78), then passes 1050–1200. Stable_from/confirmed_step null. Sweep STOPPED here per lane rule — remaining 18 toys + 3 natives NOT_RUN.
- vector_unequal_width job was already in flight past STOP; it reached PASS 122.2 s but is diagnostic only (does not count; sweep had stopped). Recorded for uniformity only.
- IMG_STRIPES2 screen never started (STOP before it). No cherry-picking: the DQ stands on the declared gate order.

## Final ranking inputs
- ka2: NOT 22/22 in this round (1 PASS / 1 FAIL / 20 NOT_RUN incl. 1 diagnostic PASS post-STOP). DQ'd — no pricing, no final-board entry.
- Cited prior (not re-run): ka2 shift 50/81 FAIL, hold 120/120, stable 3520; shift seconds 736.1 from lead-ka2-bundle/result.json.

## Gate log (tests.jsonl convention)
- candidate=ka2, gate=mode_hold, status PASS, seconds 93.8
- candidate=ka2, gate=vector_unequal_mass, status FAIL, seconds 143.1
- remaining gates SKIPPED/NOT_RUN per DQ STOP (recorded in tests.jsonl).

## Replay commands
CAND=/tmp/opencode/t22_ka2/cand-ka2
CUDA_REPO=/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda
FIXDIR=/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures
CUDA_VISIBLE_DEVICES=GPU-72c1b506-891d-b8bc-b353-e020585e1c47 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -u $CAND/probe.py --repo $CUDA_REPO --task vector_unequal_mass --output OUT --config $CAND/config.json --backend cuda --initial-state $FIXDIR/vector_unequal_mass/initial-values.pt

## Artifacts
- /tmp/opencode/t22_ka2/out/ka2-toy-mode_hold/result.json (PASS)
- /tmp/opencode/t22_ka2/out/ka2-toy-vector_unequal_mass/result.json (FAIL — DQ evidence)
- /tmp/opencode/t22_ka2/out/ka2-toy-vector_unequal_width/result.json (post-STOP diagnostic PASS, not counted)
- /tmp/opencode/t22_ka2/sweep.log

## Narrowing for next mechanism choice
Failure mode is late-run instability, not acquisition: converged by step 300, held 650 updates, then a single-observation covariance excursion at step 1000 broke the terminal 5-suffix. Useful negative: ka2's fast-release (K_REL 0.5) re-anchor/agility corner does not protect the rare-component covariance on unequal mass. Next: slower release or covariance-targeted damping, not faster motion.
