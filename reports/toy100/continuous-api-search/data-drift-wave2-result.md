# Data drift API: DV6 leads; no qualified release winner

**Three-proposal review point reached.** DV6 acquired, retained and repeatedly reacquired the public ring, including a complete uninterrupted30,000-update run. It then failed two frozen image tasks. No default promotion, merge or publication. Work stayed in `/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T221835Z-1504033/data_drift_mobility/20260926T221835Z-1504042/repo` on PR195 `fa511ce010120b502f494d717d01b14b8551eed8`.

## Own measured leaderboard

All ring runs use actual `get_recipe(total_steps=None, continuous_policy=...)` and `GANTrainer.step`:20,000particles/z2/batch2048,width96×3/Fourier3,seed0,CPU initialization then assignedCUDA. FP32, deterministic algorithms, TF32 off, native nonfused/nonforeach Adam, **serial_backward=True**. Observations remain eight modes and HQ≥.90 every10 updates using4096 isolated samples. No seed variants or image-based selection.

| Rank / policy | Original target through2400 | Change after2400: first arrival | Passing/total since arrival | Every later departure | Final suffix / minimumHQ since arrival |
|---|---|---|---|---|---|
|1 API-DV6|550;186/186;prehold120/120|2790 (+390)|182/182|none|182checks from2790 / .906738|
|2 API-DV5|550;186/186;prehold120/120|2750 (+350)|184/186|2760,2780|182checks from2790 / .584961|
|3 API-DV4|Not observed by2400;finalHQ.846191;prehold0/120|2860 (+460)|175/175|none|175checks from2860 / .911865|

All three frozen no-update controls passed **0/220**. Single-run wall times:61.603s/62.989s/63.719s forDV4/5/6. DV4's missing acquisition is a finite-window observation, not proof of impossibility. DV5's two early transitional misses leave longer retention **UNVERIFIED**, not irrevocably rejected by an all-checks rule. The historical worker's stricter screenFAIL flags remain in the ledger; overall assessment follows the shared protocol and supervisor clarification. No arrival deadline is imposed and no first arrival is rewritten as its final suffix.

**DV6 stationary7500:** first arrival550; **696/696** thereafter, no departures, minimumHQ.900391, all eight modes. Final suffix550–7500 spans6950 updates.102.290s.

## DV6 uninterrupted30,000 and delayed/repeated changes

Same learner and full source snapshot as single/stationary; target changes belong only to the evaluator.397.563s. Initial target: arrival550,546/546 through6000,minHQ.900391.

| Change after | First arrival / delay | Passing/total thereafter | Every departure | Final stable suffix | MinimumHQ since arrival / suffix |
|---|---|---|---|---|---|
|6000|6410 /410|140/140 through7800|none|6410–7800:140checks,1390updates|.910156 / .910156|
|7800|8220 /420|1877/1879 through27000|8280,8290|8300–27000:1871checks,18,700updates|.634521 / .933838|
|27000|27370 /370|263/264 through30000|27380|27390–30000:262checks,2610updates|.898682 / .907471|

All post-arrival mode counts in this long run remain eight. The complete transition windows reach minimumHQ0 during travel; no transient is omitted. The separately declared9000 prefix is the **same run**, not an independent pass:546/546,140/140,77/79 since the respective arrivals; last prefix suffix8300–9000 has71checks. The retained screen labels9000/full30000 FAIL because of departures; these early transition observations are distinguished from later stationary collapse. No departure occurred after8300 before27000, or after27390 through30000. Finite windows cannot prove literal infinite stability.

Policy closes at978, reopens at6007/7807/27006, and closes again at6919/8894/27759. Actual G/D rates span8.8483e-7–.00424998; prior3.4593e-6–.00849996. Noise remains0/.029. `policy-analysis.json` lists every close/reopen and suffix minimum; per-update `learning-rates.jsonl` contains actual rates/noise/controller values.

## Broader quality:2 PASS,2 FAIL,18 NOT_RUN

All four use the frozen residual_upsample16 architecture,32particles/z8/batch32,CPU fixture thenCUDA,600updates,24checks,original thresholds and required five-check suffix. Exact finite-prior enumeration and observation-noise seed402+step+1901 are retained; live weights determine scores. G/D/prior/EMA initial hashes and shared data RNG match the independently audited fixture. Only resource dimensions differ from the DV6 ring recipe.

| Task | Passing/24 | Final suffix | Final modes / HQ | Result |
|---|---:|---:|---|---|
|img_intensity2|14|14 from275|2 /1.0|PASS|
|img_stripes2|23|23 from50|2 /1.0|PASS|
|img_bars4|0|0|3 /1.0|FAIL|
|img_blobs4|0|0|2 /.9375|FAIL|

Bars4 final mode fractions:[.21875,.28125,.09375,.40625]; the third mode has3/32 against the unchanged minimum4/32. Blobs4 quality fractions:[.5,0,.4375,0]. High HQ does not repair missing qualified mode mass.

The data detector correctly stays quiet (maximum normalized scores1.428/1.279). Bars4 ends at mobility.1868,G/D LR.0008285; blobs4 has payoff error3.034, mobility≈1 and fullG/D LR.00425. **Both600-update failures occur before KA2's799-call bootstrap finishes**, so the surprise brake is still1 and no critic anchor has started. The remaining problem includes cold-game stability and mode acquisition, not just long-age data-change permission. Common global prior_reg0 and KA2 settings replace old solvability hyperparameters consistently; no task-specific supervised auxiliary objective was removed.

Remaining six vectors, three native100 tasks, tiny mode_hold and eight custom hosts are **NOT_RUN**, itemized in `broader22-status.json`. The eight custom routes also lack faithful public component-controller integration. Prepared vector/image adapters are not evidence of unexecuted passes.

## Mechanisms and eligibility

DV4 combines DV3's normalized temporal nonlinear real-feature detector with a separate unexplained critic-surprise brake. Features use32 fixed random projections, linear terms and sin/cos frequencies1/2; fast.1/slow.01 means track empirical minibatch variance and covariance. This can detect mean-preserving changes; it is an engineering detector, not a universally calibrated significance test. Real statistics influence scalar mobility only, never targets, losses or output translation.

DV4 kept cross-update generator-gradient coherence for mobility and failed cold acquisition. DV5 replaces that driver with EMA.02 of `max(0,(L_G-L_D_adversarial)/log(2))`, squared and capped for mobility. DV6 uses **current accumulated data_drive**, rather than stale slowly decaying data_memory, to authorize surprise exemption and critic-memory release. Unexpected normalized Adam-gradient surprise gives trust`1/(1+unexplained²)` and brakes all rates/anchor tracking. Its applied-rate evidence has one causal update of lag. Gradient coherence and data_memory remain diagnostics. None reads quality, labels, centers, task IDs, target-change notices, a planned endpoint or evaluator budget.

The799-call critic bootstrap initializes/calibrates one reference, not a user-operated acquisition/maintenance switch. Subsequent updates and reopening remain available at arbitrary ages, demonstrated at27000. Rolling smoothing constants never expire learning. Its lack of an early surprise brake is a measured image limitation. Nominal.01/.05 rate coefficients are signal-based scales, not scheduled decay floors; trust can reduce them further. The adversarial learner, native Adam, trainable prior, KA2/A2 mechanics and general API are retained.

Read-only predecessor evidence remains intact: DI2 research77/81 with late mode loss and schedule coupling; DI3 snap45/81; publicDV1 single-shift success but17 stationary misses4850–5010; DV2/DV3 initial misses3/9. Those are research leads, never descendant passes. DV4–6 each earned their own API measurements on the corrected runtime.

## Verification and provenance

- **87/87 regression tests PASS**, including public components, six serial-scope/exception/checkpoint contracts, normalized mean-preserving variance detection atbatch32/2048 and continuous helper/state tests.
- **Horizon prefix PASS:** all24 full-state receipts through2400 identical under evaluator budgets4600/7500.
- **Fresh-process2400→2500 PASS:** exact initial restore, all ten live/EMA observations, full final model/optimizer/controller/EMA/RNG/data-stream hashes. CUDA parameters, gradients and moments verified;17 ordinary Adam scalar counters remainCPU metadata. Archived script/PID and independently decoded state audit support the result.
- **529 source/artifact hash checks PASS**; DV6 package bytes match across every completed quality protocol and current checkout. Shared evaluation declaration is byte-identical.
- Ledger: **8 PASS /6 FAIL /1 ERROR gate rows**, plus29 explicitly SKIPPED. This includes overlapping9000/30000 evidence, read-only audits and regressions, not15 independent quality trials.

One initial long launch was interrupted after the1590 observation when an image-first supervisor update was read; its ERROR, raw trace and approximate21.122s duration are preserved at `runs/dv6-long`. Root clarified that ordering advice applies only before launch. The fresh `dv6-long-full` is a complete cold uninterrupted30000 with unchanged times/policy, never a resumed partial or selected endpoint. The first real candidate began about147s after attempt launch.

The reviewed standalone serial patch hash is`cbc74e2bdd473478d3bdfd5e1cba2aae7597743d76ffc97d2a2e7cef3e1f57e6`. Public option is scoped, restores caller autograd context and validates checkpoint configuration. Copied predecessor sources/hashes are in `predecessor-source-receipt.json`/`inherited-sources.zip`; exact mechanism deltas in `dv4-to-dv5.diff`/`dv5-to-dv6.diff`; final replay patch SHA256`307f064da9892744ee81e1c602c23029de4b75c711ce4f164e6afae85257d937` in `final-library.patch`. Every run stores declaration,source.zip,initial hashes,logs,state and artifact hashes. Independent copied audits cover DV5/DV6 transitions, image fixture and exact API replay.

**Ordinary matched K3P: NOT_RUN.** Root assigned this lane ownership after survival; no baseline job launched before the subsequent image regressions. Supervisor update22:49UTC confirms broader failures gate off remaining work and transfers conditional comparator ownership toRP5. Exact publicv0.8.0 commit`0ff9a7afe5dcb828239369446cfe71971bce687b` is prepared only in `k3p-public`, with no eager/relocated Adam state. Historical K3P scores remain different-runtime context and cannot establish a release comparison.

## Replay and next review

All artifact paths above are relative to `/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T221835Z-1504033/data_drift_mobility/20260926T221835Z-1504042/repo/reports/data-drift-api`. The experiment ledger is `/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T221835Z-1504033/data_drift_mobility/20260926T221835Z-1504042/tests.jsonl`. Tail logs with`tail -f repo/reports/data-drift-api/runs/<run>.log` from the attempt directory. All workers have exited.

```sh
export CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
PY=/tmp/pr38-default-env/bin/python
$PY -u reports/data-drift-api/worker.py --schedule dv6 --protocol single_shift --output <fresh-dir>
$PY -u reports/data-drift-api/worker.py --schedule dv6 --protocol stationary --output <fresh-dir>
$PY -u reports/data-drift-api/worker.py --schedule dv6 --protocol long_continuation --output <fresh-dir>
$PY -u reports/data-drift-api/image_gate.py --output <fresh-dir>
$PY -u reports/data-drift-api/remaining_images.py --task img_bars4 --output <fresh-dir>
$PY -u reports/data-drift-api/api_checks.py --candidate dv6
$PY -m pytest -q tests/test_training.py tests/test_recipe_defaults.py tests/test_ka2.py tests/test_k3p.py tests/test_serial_backward.py tests/test_data_drift_successors.py
```

Replay historicalDV4/5 from their own source.zip overlays, because later controller state schemas add fields. The exact API-check command expects the declared single/stationary artifact directories and a fresh `dv6-api-checks` output path.

**Recommendation:** retain DV6's separate data evidence and measured long-ring behavior, then investigate a cold-game stability signal that operates before critic-reference calibration. Blobs4's large payoff drives full mobility while only two modes qualify; bars4 has adequate HQ but insufficient mass in one mode. A later authorized mechanism should address these distinct failures through ordinary training signals, then re-earn its own ring and frozen-image scores. No seed sweep, coefficient grid, borrowed passes or a fourth proposal was run. DV5 longer retention remains unverified; its before800 arithmetic is unchanged fromDV6, so rerunning it alone is not a proposed repair for these image failures. Supervisor22:49UTC confirms this completed three-proposal cap and will review/refill; no fourth mechanism is authorized in this attempt.
