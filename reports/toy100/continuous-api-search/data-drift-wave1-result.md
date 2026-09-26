# Data drift mobility: three proposals, no qualified winner

**Review point reached.** All three policies have measured quality failures. No merge, publication or release recommendation. Base: `fa511ce010120b502f494d717d01b14b8551eed8`. All work/artifacts are in this attempt; no additional agents or seed variants.

Repository: `/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T211209Z-1398466/data_drift_mobility/20260926T211209Z-1398477/repo`. Artifact paths below are relative to `repo/reports/data-drift-api/`; the complete gate ledger is `tests.jsonl` beside this report. Earlier progress and diagnoses are retained in `progress-history.md`.

## Measured leaderboard

Same public fixture: seed0, 20,000 trainable particles, z_dim2, batch2048, width96/three layers/Fourier3; CPU initialization then assigned CUDA GPU; FP32, deterministic kernels, TF32 off, native non-fused Adam. All initial model, optimizer and RNG hashes match the retained public worker. Passing means **eight modes and HQ>=.90**, observed every10 with4096 isolated samples. All three initial arrivals were update790.

| Research lead | Pre-change1210–2400 | Changed-target first arrival | Passing/total since arrival | Post-arrival departures | Final suffix / min HQ since arrival | Result |
|---|---:|---:|---:|---|---|---|
| API-DV1 |120/120|2830 (+430)|178/178|none|178 checks,1770 updates / .900635|Single shift PASS; stationary FAIL|
| API-DV2 |117/120|2800 (+400)|181/181|none|181 checks,1800 updates / .902344|Pre-change retention FAIL|
| API-DV3 |111/120|2920 (+520)|169/169|none|169 checks,1680 updates / .907471|Pre-change retention FAIL|

Each single-shift run completed the declared4600 updates, with frozen comparator **0/220**. Times:96.10s,95.85s,99.14s. These passing shifted suffixes do not erase earlier failures.

- **DV1 stationary7500:**655/672 checks since acquisition790; all17 departures are4850–5010 every10. MinimumHQ0, minimum modes0. Final suffix249 checks starts5020 (2480-update span). Runtime155.07s. The real-data drive was always0 (max innovation2.572); own game instability caused collapse and subsequent recovery.
- **DV2 pre-change:**159/162 since acquisition; departures1390,1420,1430; minimumHQ.769287, eight modes. Final pre-change suffix97 checks starts1440.
- **DV3 pre-change:**153/162 since acquisition; departures1390,1400,1410,1420,1430,1440,1470,1480,1490; minimumHQ.357666, minimum modes6. Final pre-change suffix91 checks starts1500.

Exact every-check histories, minima over entire transition windows, comparison endpoint3600 and all retrospective suffixes: `runs/dv{1,2,3}-single/summary.json` and `runs/dv1-stationary/summary.json`. Raw observations: each `metrics.jsonl`. Every applied rate/noise/controller value: `learning-rates.jsonl`. Original DV1 result LR extrema include an unused configured maximum; summaries use actual per-update extrema.

## Mechanisms and retained evidence

DI2's historical research hold1200/1200+300/300 and77/81 recovery remain leads, not inherited passes. It lost a mode after novelty subsided and retained horizon-based rates/noise. DI3's snap worsened recovery to45/81. Existing public constant KA2's61/120 prehold and83 post-arrival failures, and decayed KA2's120/120/+1690 arrival, were background rather than rerun baselines.

- **DV1:** normalized fixed random nonlinear features distinguish real-data innovation; generator-gradient cosine is a separate game signal. Reversible mobility controls G/D/prior rates. It repaired the single shift, but surprise-driven critic memory still destabilized stationary training.
- **DV2:** independently smoothed data evidence authorizes KA2 memory movement/release; otherwise the anchor stays fixed. Recovery remains strong, but cold refinement loses three observations.
- **DV3:** restores K3P's slow.999 critic EMA on unchanged data. Also accumulates fast/slow feature-mean evidence with tracked variance/covariance after a measured batch32 detector failure. It detects that variance change but worsens cold retention. The preflight amendment was declared before any DV3 training; its earlier draft is retained.

All use actual `get_recipe(total_steps=None, continuous_policy="dv1"/"dv2"/"dv3")` and `GANTrainer.step`. No learner budget or stopping count; input noise0/output.029 constant. Neither real statistics nor evaluator scores fit or translate outputs. KA2's799-call initial critic bootstrap remains an absolute initialization only; later learning, rates and memory are autonomous at arbitrary ages. Smoothing constants do not encode target times or a forced recovery dwell. Controller tensors/history are checkpointed. Existing losses, trainable prior and general API families are retained; auxiliary hosts were not rewritten.

Declarations/source snapshots: `dv{1,2,3}.json`, copied `evaluation-protocols.json`, and each run's `declaration.json`, `source.zip`, `initial.json`, checkpoints and `artifact-sha256.json`. Historical files were not changed.

## Checkpoint defect and reusable API fix

DV1's **2400-update horizon prefix PASS**: all24 complete state receipts match under evaluator budgets4600 and7500. Its original fresh-process2400→2500 continuation **FAIL** remains recorded. Models, optimizer, controller and RNG restored exactly; all17 Adam step counters retained CPU float32 placement.

The shared defect is higher-order autograd accumulation order. Identical graphs have different cross-thread sequence-number priorities in a fresh process. Example: `MeanBackward0` versus `MmBackward0` priorities27/36 become204/88, reversing their order. Forward inputs/outputs match until the first critic update; weight gradients differ by up to5.96e-8 before Adam. Same-process restored branches agree, so those alone were insufficient. Evidence: `runs/autograd-order/graphs.json`, `runs/file-localize/`, `runs/repeated-restore/`, `runs/fresh-objects/`. Scalar and double-backward priming failed; retained in the ledger. PyTorch documents thread-local node numbers and orders its ready queue by them ([Node source](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/autograd/node.h), [queue source](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/autograd/engine.h)).

**Fix:** `GANTrainer(..., serial_backward=True)` scopes the entire update, including nested create_graph and both backwards, to one autograd thread. It restores caller context on success and exceptions, serializes the option, and rejects cross-option loads. False retains/inherits the caller's ambient setting; legacy checkpoints do not capture that ambient flag. Only True enforces the serialized serial constraint. This changes rounding relative to historical execution, so old quality scores are not assigned to True mode.

A full-fixture cold1100 reference saved at1000; a genuinely fresh process restored1000→1100. **Every model/optimizer/controller/EMA/RNG/data-stream hash and all10 observations match exactly.** CUDA parameters and moments verified. Evidence: `runs/serial-checkpoint-reference/` and `runs/serial-checkpoint-resume/`. This is DV1-policy state correctness under True mode, not quality qualification or a DV3 CUDA-continuation claim.

Standalone patch, with no drift-policy coupling: **`serial-backward.patch`**, SHA256 `cbc74e2bdd473478d3bdfd5e1cba2aae7597743d76ffc97d2a2e7cef3e1f57e6`. It applies cleanly to pinned PR195. Six standalone contract tests pass, including actual nested create_graph/outer-backward exceptions, context restoration, mode mismatch and legacy compatibility. Details: `serial-patch-receipt.json`.

Reference/resume source archives differ in **recipes.py docstrings** and **training.py documentation/checkpoint validation**, not step arithmetic. Constructor, public step, internal update, state_dict and sample ASTs match; recipe semantics match. Exact differences/hashes: `serial-checkpoint-source.diff`, `serial-source-comparison.json`. Initial serial snapshots omitted imported worker/benchmark helpers; a **post-execution** receipt verifies13 dependencies against the earlier DV3 archive or pinned base: `serial-dependency-receipt.json`, `serial-dependencies.zip`, `serial-complete-{reference,resume}-sources.json`. Collector failures and synthetic/alias exclusions remain documented.

## Test totals and missing qualification

Executed ledger: **30 gates**, 15 PASS / 15 FAIL; plus 14 explicitly SKIPPED. These include diagnostic reproductions and do not represent independent quality passes. One early10-update diagnostic lacked an elapsed timer (`seconds:null`), explicitly retained.

Regression sequence:90PASS/1FAIL (instantaneous detector misses variance change atbatch32); then94PASS with DV3's same threshold; final public API suite58PASS; standalone final contract6PASS. The detector checks are statistical/mechanics regressions, not22-task quality.

Declared **stationary7500, delayed/repeated9000 (changes6000/7800), uninterrupted30000 (also27000)** are retained unchanged. DV2/DV3 stationary, all delayed/repeated/30000 runs and all candidate-own22 suites are **NOT_RUN**, gated by observed failures. Matched K3P is **NOT_RUN here**; the supervisor assigned it to the precision lane. No source or score borrowing, deadline gate, seed sweep, manual phase switch, image-based selection, merge or publication.

## Replay and next review

Use each historical run's exact source.zip over PR195 for score replay; current library fixes are separately hashed in `final-library.patch`/`final-source-sha256.json`. Benchmark environment on every subprocess:

```sh
export CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
PY=/tmp/pr38-default-env/bin/python
# Original DV1 snapshot:
$PY -u reports/data-drift-api/worker.py --schedule dv1 --steps 4600 --output <fresh-output>
# Later worker snapshots (dv2/dv3 single-shift or dv1 stationary):
$PY -u reports/data-drift-api/worker.py --schedule dv2 --protocol single_shift --output <fresh-output>
$PY -u reports/data-drift-api/worker.py --schedule dv3 --protocol single_shift --output <fresh-output>
$PY -u reports/data-drift-api/worker.py --schedule dv1 --protocol stationary --output <fresh-output>
# State-regression pair in a fresh copy of this source bundle:
$PY -u reports/data-drift-api/serial_checkpoint.py reference
$PY -u reports/data-drift-api/serial_checkpoint.py resume
# Reusable standalone patch contract:
git apply reports/data-drift-api/serial-backward.patch  # in a clean PR195 checkout
$PY -m pytest -q tests/test_serial_backward.py
```

Tail: `tail -f repo/reports/data-drift-api/runs/<run-name>.log` from this attempt directory; all workers have finished.

Recommendations: establish one declared serial-backward runtime for future candidates and the shared comparator, then re-earn quality. Preserve DV1 as the strongest partial lead, while addressing the difference between useful cold critic-memory adaptation and self-induced stationary drift. Hard freezing and fixed slow EMA both have measured cold failures. Keep temporal data evidence as a tested small-batch improvement, but do not infer GAN stability from its detector test. Further mechanisms require supervisor review/replenishment; this attempt used exactly three. Finite tests cannot prove infinite stability.
