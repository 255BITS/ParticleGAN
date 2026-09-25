No formulation qualified. The six-proposal cap is exhausted. All six ran both primary GPU gates; no candidate passed both, so no downstream qualification was run. The original recipe remains unchanged.

Measured leaderboard (rank by sustained primary passes, then mean frozen final normalized shortfall):

| Rank | Candidate | Primary PASS | Ring modes / HQ | Ring passing suffix | Unequal eigen ratio | Unequal covariance error | Unequal passing suffix |
|---:|---|---:|---|---:|---:|---:|---:|
| 1 | `c05_ra_r1r2` — Ra logistic + R1+R2 | 1/2 | 8 / 1.00000 | 5 | 0.38229 | 1.34110 | 0 |
| 2 | `c06_ra_r1_only` — Ra logistic + R1 only | 1/2 | 5 / 1.00000 | 0 | 0.33534 | 0.25129 | 6 |
| 3 | `c02_ra_cap` — Ra logistic + original cap | 0/2 | 7 / 0.99561 | 0 | 0.57846 | 0.88795 | 0 |
| 4 | `c01_r1r2` — Rp logistic + R1+R2 | 0/2 | 8 / 0.90137 | 1 | 0.02234 | 0.79466 | 0 |
| 5 | `c03_smooth_cap` — Rp logistic + smooth radial cap | 0/2 | 2 / 0.25952 | 0 | 0.00000 | 0.48501 | 0 |
| 6 | `c04_hinge_cap` — Rp hinge + original cap | 0/2 | 5 / 1.00000 | 0 | 0.00689 | 2.31124 | 0 |

The frozen ring gate needs 8 modes and HQ >= .90. Unequal mass includes minimum component eigenvalue ratio >= .15 and component covariance error <= .85, along with its unchanged other metrics. Both require at least five passing final observations. These are live-model sustained verdicts, not EMA or final-only passes.

`c05_ra_r1r2` is the strongest partial result: ring PASS with exactly five final passing observations; unequal mass FAIL with covariance error 1.34110. Its mass TV .00913 and eigen ratio .38229 are good, but the rare component covariance error is 4.25338. `c06_ra_r1_only` corrected unequal mass (PASS, six final passing observations, covariance error .25129, eigen ratio .33534, mass TV .01519), but ring coverage fell to five modes. These passes must not be combined.

R1+R2 alone reached eight final ring modes but had only one passing final observation and eigen ratio .02234 on unequal mass. Ra with the original cap remained at seven ring modes and narrowly missed unequal covariance (.88795). The smooth radial cap lost rare mass entirely and collapsed ring coverage to two modes. Hinge lost coverage on both distributions.

The six proposals were sequential adaptive experiments: existing R1+R2; existing Ra loss; smooth zero-centered radial cap; existing hinge loss; evidence-driven Ra + R1+R2 composition; then removal of fake-point R2 pressure from that composition. No coefficient grid or seed sweep was run.

Qualification status for **every candidate**: `trajectory`, `img_intensity2`, `img_bars4`, `img_blobs4`: **NOT_RUN**; own full 22 GPU suite: **NOT_RUN**; own-state convergence continuation: **NOT_RUN**. They were gated out by primary failures. Full gate matrices are in `repo/reports/toy100/critic-formulation-attempt/LEADERBOARD.md` and `leaderboard.json`. No 20/22, 22/22, or stable-convergence claim is supported.

Validation and compute:

- 12 executed training gates: **2 PASS, 10 FAIL, 0 ERROR**. Three regression/audit gates: **3 PASS** (smooth-cap analytic/finite-difference/zero-slope check; R1-only real-gradient/fake-independence/forward-count check; complete receipt audit). The ledger also records 18 explicitly skipped downstream stages. Setup metadata is not counted as a proposal.
- All 12 training runs used 1,200 D + 1,200 G/prior Adam updates: **28,800 optimizer calls total**, with no added optimization steps. Total measured probe time: 464.32 seconds.
- Equal frozen step budgets and equal network forward/backward counts. R1+R2 avoids a norm square root; Ra and smooth-cap kernels change scalar arithmetic, so these are not claims of identical FLOPs or wall time. The R1-only candidate retains the zero-weight fake penalty graph to keep the same network work and native draw order. No model forward/backward evaluations were added.
- CPU initialization fixtures were reused; parameters, gradients, Adam moments and training draws were CUDA. FP32, deterministic algorithms, TF32 off, one CPU thread, physical GPU `GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69`. Ordinary Adam, original learning-rate/noise schedules and auxiliary AE/token terms were preserved. No labels, centers, target statistics or metric feedback entered training.
- Receipt audit verified all 1,614 prepared CUDA source-file hashes and all 12 executed probe/config hashes, initialization fixtures, native random-stream hashes, sustained verdicts and exact update counts. Native random-stream receipts and initial parameters match the retained initialization-only controls; no controls were retrained.

The next authorized experiment should retain the Ra objective and test **R1 on real samples plus the original one-sided cap on fake samples**. This is an unmeasured location-specific critic constraint: C05 needs less fake-point flattening, while C06 shows that removing fake-point control entirely loses ring modes. Keep strengths fixed initially and require both cold primary gates again. Do not promote either partial winner. This recommendation was **NOT_RUN** because the cap is six proposals.

Artifacts and exact replay:

- Attempt report: `/ml2/hypergan/gan-attempts/formulations-20260924T233426Z/critic_formulation/20260924T233426Z-2040500/result.md`.
- Append-only gate ledger: `/ml2/hypergan/gan-attempts/formulations-20260924T233426Z/critic_formulation/20260924T233426Z-2040500/tests.jsonl`.
- Artifact root: `/ml2/hypergan/gan-attempts/formulations-20260924T233426Z/critic_formulation/20260924T233426Z-2040500/repo/reports/toy100/critic-formulation-attempt`.
- Immutable executed source snapshot: `prepared/repos/cuda/`; source hashes: `prepared/prepared-sources.json`. Each candidate retains `config.json`, exact executed `probe.py`, pre-launch `declaration.json`, complete command/environment receipts in `commands.jsonl`, per-gate logs and `runs/<gate>/result.json`.
- New mechanism code: `smooth_cap_install.py` and `r1_only_install.py`; their exact installed copies are embedded in the corresponding candidate probes. Checks: `check_smooth_cap.py`, `check_r1_only.py`, `audit_completed.py`; results: `smooth-cap-check.json`, `r1-only-check.json`, `receipt-audit.json`.
- Source preparation was executed once; first candidate GPU launch at 2026-09-24 23:36 UTC, within five minutes. No background workers remain.

Executed setup/batch commands (from `/ml2/hypergan/gan-attempts/formulations-20260924T233426Z/critic_formulation/20260924T233426Z-2040500/repo`; exact expanded subprocess argv and environments are retained in each `commands.jsonl`):

```bash
/tmp/pr38-default-env/bin/python reports/toy100/cpu-recipe-gpu-port/prepare.py --root reports/toy100/critic-formulation-attempt/prepared
/tmp/pr38-default-env/bin/python reports/toy100/critic-formulation-attempt/run_batch.py c01_r1r2 mode_hold vector_unequal_mass
/tmp/pr38-default-env/bin/python reports/toy100/critic-formulation-attempt/run_batch.py c02_ra_cap mode_hold vector_unequal_mass
/tmp/pr38-default-env/bin/python reports/toy100/critic-formulation-attempt/run_batch.py c03_smooth_cap mode_hold vector_unequal_mass
/tmp/pr38-default-env/bin/python reports/toy100/critic-formulation-attempt/run_batch.py c04_hinge_cap mode_hold vector_unequal_mass
/tmp/pr38-default-env/bin/python reports/toy100/critic-formulation-attempt/run_batch.py c05_ra_r1r2 mode_hold vector_unequal_mass
/tmp/pr38-default-env/bin/python reports/toy100/critic-formulation-attempt/run_batch.py c06_ra_r1_only mode_hold vector_unequal_mass
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/pr38-default-env/bin/python reports/toy100/critic-formulation-attempt/check_smooth_cap.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/pr38-default-env/bin/python reports/toy100/critic-formulation-attempt/check_r1_only.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/pr38-default-env/bin/python reports/toy100/critic-formulation-attempt/audit_completed.py
```

For a fresh single-gate replay without re-preparing sources or overwriting evidence:

```bash
cd /ml2/hypergan/gan-attempts/formulations-20260924T233426Z/critic_formulation/20260924T233426Z-2040500/repo
CF_ROOT="$PWD/reports/toy100/critic-formulation-attempt"
CF_CANDIDATE=c05_ra_r1r2
CF_TASK=mode_hold
env -u LD_PRELOAD -u PYTHONPATH \
  CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 \
  ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 \
  /tmp/pr38-default-env/bin/python "$CF_ROOT/candidates/$CF_CANDIDATE/probe.py" \
  --repo "$CF_ROOT/prepared/repos/cuda" --config "$CF_ROOT/candidates/$CF_CANDIDATE/config.json" \
  --task "$CF_TASK" --backend cuda \
  --initial-state "$PWD/reports/toy100/cpu-recipe-gpu-port/initialization-fixtures/$CF_TASK/initial-values.pt" \
  --output "$CF_ROOT/replay-$CF_CANDIDATE-$CF_TASK-new"
```

Logs remain easy to inspect: `tail -F /ml2/hypergan/gan-attempts/formulations-20260924T233426Z/critic_formulation/20260924T233426Z-2040500/repo/reports/toy100/critic-formulation-attempt/batch-c06.log`. Each batch has its own `batch-c01.log` through `batch-c06.log`; `progress.jsonl` contains all 12 measured gate completions.
