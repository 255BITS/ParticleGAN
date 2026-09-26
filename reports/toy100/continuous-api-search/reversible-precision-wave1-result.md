# Reversible precision — final attempt report

**No qualified release winner.** API-RP2 retained the ring through its own uninterrupted30000-update protocol, but failed frozen image quality and exact standalone continuation. API-RP3 passed its own short ring screen but also failed image quality. Three declared proposals are complete; no fourth, merge or publication. Supervisor accepted the worker-eager diagnostic as a separate evidence scope within the earlier proposal, not a fourth proposal. Search can continue after supervisor review/refill.

Base: `fa511ce010120b502f494d717d01b14b8551eed8`. All candidate evidence below uses actual public `get_recipe()`/`GANTrainer.step()`, except the explicitly scoped worker-eager diagnostic. Seed0 only. No seed sweep, coefficient grid, target fitter, metric feedback or research-host pass.

## Leaderboard — keep complete configurations separate

| Configuration | Pre-change retention (1210–2400) | Shift2400 → arrival; retention afterward | Additional evidence / verdict |
|---|---|---|---|
| API-RP1 ordinary Adam, interrupted at2298 | 89/109 through2290; minHQ.115479 | NOT_RUN | Valid partial retention failure |
| API-RP1-CUDA-EAGER worker diagnostic | 120/120 | 2900 (delay500);171/171 | stationary687/687; diagnostic only, no public qualification |
| **API-RP2 public eager-state option** | **120/120; minHQ.966553** | **2900 (delay500);171/171; minHQ.967529** | own stationary7500 and long30000 pass; image2/24 FAIL; standalone continuation FAIL |
| API-RP3 cancellation + serial backward | 120/120; minHQ.963623 | 2860 (delay460);175/175; minHQ.910400 | image0/24 FAIL; expensive qualification NOT_RUN |

RP1 cold arrival640;146/166 checks since arrival. Departure1620, every10-update check through1810 fails, minimum4 modes/HQ.115478515625; final suffix1820–2290 (48 checks). I unnecessarily interrupted at2298 because I initially misclassified CPU scalar Adam counters. They are valid runtime metadata. The old directory `runs/rp1-invalid-counters` is misleading; its scope.json and measured failure are authoritative.

RP2 cold arrival640;177/177 through2400, minHQ.908203125. Single-shift final suffix2900–4600, no departures. Events: close1546, reopen2437, close3170. RP3 cold arrival610;180/180 through2400, minHQ.90234375; final shifted suffix2860–4600, no departures. Events: close1809, reopen2432, close3303. Both frozen no-update controls passed0/220 after the shift. All ring passes require eight modes and HQ≥.90; EMA is diagnostic.

## API-RP2 own long evidence

Stationary7500: arrival640,687/687 thereafter, no departures or false reopening; finalHQ.98388671875. Long30000 was uninterrupted, with target changes fixed only in evaluator/data:

| Segment | First arrival (delay) | Passing / total since arrival | Departures | Minimum HQ since arrival | Final stable suffix |
|---|---|---|---|---|---|
| 0–6000 | 640 (640) | 537/537 | 0 | .908203125 | 640–6000 |
| 6000–7800, offset[1,0] | 6360 (360) | 145/145 | 0 | .9423828125 | 6360–7800 |
| 7800–27000, offset[1,1] | 8220 (420) | 1879/1879 | 0 | .984130859375 | 8220–27000 |
| 27000–30000, offset[0,1] | 27320 (320) | 269/269 | 0 | .932373046875 | 27320–30000 |

All post-arrival mode minima8. FinalHQ.9912109375. Close/reopen sequence:1546 close,6011 reopen,6788 close,7876 reopen,8401 close,27009 reopen,27543 close. Frozen controls:0/180,0/1920,0/300. Separately declared9000 prefix:537/537,145/145,79/79 after each arrival; second-change minimumHQ.98681640625. These finite windows and retrospective suffixes do not prove infinite stability or qualify another variant. All segment-wide minima, failures, and observations remain in summary.json/metrics.jsonl.

## Image rejection and mechanism finding

Both use the corrected frozen `plans/default_comparison.json` img_intensity2 card: residual_upsample,width16,32 particles,z8,batch32,600 updates,24 observations. Original scoring: RMSE≤.06, two modes with quality fraction≥.25 each, HQ≥.90, final passing suffix≥5. CPU G/D/prior initialization, shared CUDA data/latent stream and isolated output-noise seed402+update+1901 preserved. Source audit corrected an unexecuted draft before RP2 training; no wrong-card score exists.

- **RP2:**2/24 passes at450 and600;475,500,525,550,575 fail after first pass. Minimum HQ afterward.375. FinalHQ.90625/two modes but suffix1: FAIL. No close in600 updates.
- **RP3:**0/24, finalHQ.40625/one mode, suffix0: FAIL. No close. Coherence<.1 on436/600 updates (424 consecutive), so cancellation was observed.

The precise blocker is the combined gap condition: although both traces have301 consecutive contracting updates, their required *slow contraction after50 consecutive decreases* persists at most3 updates, below the25-update closing dwell. RP2 combined calm dwell1; RP3 dwell3. This supersedes the earlier incomplete explanation that activity magnitude alone prevented closing. RP3's ring transitions all used the original magnitude branch; its short-ring improvement cannot be attributed to cancellation rather than serial arithmetic. Entire image600 lies within inherited KA2 pure-A initialization (799 calls).

**Recommendation for review/refill:** preserve RP2's measured retention/reopening lead, but redesign excursion evidence to tolerate local sign reversals during net contraction; merely adding a cancellation predicate did not fix the limiting gap gate. Keep directional telemetry and declare serial execution in a new candidate. Test the unchanged frozen image gate early, then earn all own ring, continuation and broader scores. Do not adjust a cutoff to known image step600 or quality timestamps. The latest supervisor also proposes combining RP2 retention control with C6 implicit game correction in a fresh attempt: C6 reportedly passed its image screen but failed stationary retention. This is an untested integration lead, not borrowed quality evidence or a fourth proposal here.

## Public implementation and eligibility

Current experimental source: `particlegan/precision.py`, `recipes.py`, `training.py`; defaults remain finite/lazy. RP2 uses `get_recipe(total_steps=None,continuous_precision='rp1',adam_eager_state=True)`. RP3 selects `'rp3'` and `GANTrainer(...,serial_backward=True)`. `rp1` is the internal rule name; API-RP2 includes eager initialization.

The rule measures a slow critic-reference input-gradient gap (EMA.001, gap smoothing.01) and actual generator/prior displacement normalized by applied LR. It closes only after signed contraction plus update evidence; growing gap and update innovation can reopen at any age. Open G/D LR.000884, prior.00204; closed G/D.0000425, prior.000425. RP3 adds bias-corrected generator direction/path-energy coherence (EMA.05), excluding sparse prior rows, and checkpoints the direction vector. No learner receives evaluator budget, target-change notice, centers, labels, quality or planned endpoint.

Fixed360/720 noise initialization gives mature input0/output.029. Inherited KA2799-call initialization is a one-time state setup, not a convergence detector or caller phase switch; all later rate/noise/memory/particle behavior and API stopping are independent of the evaluator horizon. These rules apply at arbitrary later ages; they do not guarantee task quality. Adversarial loss, trainable prior, KA2 memory and A2 remain. Existing auxiliary losses were not removed; auxiliary quality tasks were not run.

Eager state is an explicit substantive public optimizer-factory option using torch Adam, not a new Adam algorithm. CUDA-vs-CPU scalar-power arithmetic changes KA2 telemetry; eager zero-state visibility also affects guard/A2. Ordinary/eager surprise first differs at860, among100-step model hashes at1000. The diagnostic worker's direct state mutation is never used to qualify RP2; RP2 re-earned every cited score through its factory.

## Verification and unpassed gates

- Final source: **76 regression tests PASS**, including serial scope restoration on success/error, execution-mode checkpoint rejection, exact CPU direction/policy/optimizer/EMA/RNG continuation, and inherited API tests. Earlier RP2 source had75 passing tests. These are not quality-task passes.
- RP2 differing evaluator budgets4600/7500: exact2400-update prefix PASS (2400 applied rate/noise/policy rows,240 observations,24 complete state/RNG receipts; total_steps=None).
- RP2 standalone2400→2500 replay: **FAIL/unresolved** versus original; restored branches agree with each other and are exact immediately after load. Shared diagnosis supplied by supervisor: higher-order autograd thread-local node priorities reorder gradient accumulation after fresh-object restore. RP3 declares scoped serial backward; its own cross-process gate is NOT_RUN after image rejection. No external corrected-runtime score is borrowed.
- RP2 in-process checkpoint audits at6000,7800,27000:100/100 actual regenerated batches and four complete-state receipts (offsets1,2,3,100) each PASS. The main run never reloads. These do not erase the standalone failure.
- Remaining21 frozen quality tasks **NOT_RUN** for RP2/RP3 under declared image fail-fast. RP3 stationary/long/prefix/subprocess gates also NOT_RUN. Legacy runners that drop continuous settings cannot supply missing passes.
- Actual-public matched K3P **NOT_RUN** here. Shared ownership had moved to constant lane C6; its later stationary failure now leaves no justified comparator execution, per supervisor. Prepared eager adapter is an unexecuted diagnostic, not the exact archived K3P reference; primary must use archived25812851 package math and ordinary lazy CPU scalar counters on the matched runtime. No altered comparator score exists.

Ledger:27 rows =16 PASS,6 FAIL,2 ERROR,3 SKIPPED, including diagnostics/regressions; never combine these as candidate qualification. Two errors are the unnecessary operator interruption and an initial zero-test bad-path invocation. Unknown diagnostic durations are null. Final integrity:181 snapshotted source hashes and129 artifact hashes verified, RP3 package identical across its two runs, image model/data initialization equal to RP2, ring initialization/RNG hashes equal to retained baseline, git diff --check PASS.

## Artifacts, logs and replay

All paths below are relative to this attempt's repo unless stated otherwise. Parent `../tests.jsonl` is append-only; this file is `../result.md`.

- Exact current patch/hash manifest: `reports/reversible-precision/public-api.patch` (SHA256 `9d4981c9f198a27db59c01ceab42c14462d61104c7412faa16feff3036007483`), `public-source-sha256.json`; preserved pre-RP3 API patch/hash: `rp2-public-api.patch`, `rp2-public-source-sha256.json` in that directory. Serial pattern provenance: `rp3-runtime-provenance.json`.
- Pre-training declarations: `rp1.md`, `rp2.md`, `rp3.md`, unchanged shared `evaluation-protocols.json`, `long-audit-declaration.md`, corrected `broader-declaration.md`, corrected K3P scope in `k3p-declaration.md`.
- Own evidence: `reports/reversible-precision/runs/{rp2,rp2-stationary,rp2-long,rp2-continuation,rp2-img_intensity2,rp3,rp3-img_intensity2}`. Diagnostic predecessors: `runs/{rp1-invalid-counters,rp1,rp1-stationary}`. Each trained run preserves source.zip/declaration.json, initialization/state hashes, actual rate/noise/controller logs, observations, result/summary and artifact hashes. `applied-rate-ranges.json` corrects old result maxima that included un-applied constructor defaults; raw logs/scores were not changed.
- Easy logs: `tail -f reports/reversible-precision/rp3-live.log` (ring), `rp3-image.log` (image). No worker remains running. Prior progress and corrected interpretations are retained in `progress-before-final.md`.

Replay from the chosen run's source.zip over the supplied base, into a **fresh output directory**, using the same benchmark runtime. Example for RP3 (replace worker/arguments for other declared gates):

```sh
CUDA_VISIBLE_DEVICES=GPU-72c1b506-891d-b8bc-b353-e020585e1c47 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/tmp/pr38-default-env/bin/python -u reports/reversible-precision/rp3_worker.py \
  --schedule rp3 --protocol single_shift --output reports/reversible-precision/runs/replay-rp3 \
  > reports/reversible-precision/replay-rp3.log 2>&1
```

RP2: `public_worker.py --schedule rp2 --protocol single_shift` or `stationary`; long: `long_worker.py --schedule rp2 --protocol long_continuation`; images: `image_gate.py` (RP2) or `rp3_image_gate.py` (RP3), `--task img_intensity2 --output <fresh-dir>`. All workers enforce FP32, deterministic algorithms, TF32 off and one CPU thread. Read supervisor.md before any new batch. No new candidate launch is authorized within this exhausted attempt.
