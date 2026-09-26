# Three proposals complete; no qualified release winner

**Review point reached.** DV7 is the strongest partial lead: it passed all four frozen image tasks, including DV6’s two failures, and demonstrated long-ring retention and repeated recovery. DV7 and DV9 nevertheless fail the frozen unequal-mass spread gate. DV8 has recurrent ring instability. No fourth proposal, release recommendation, merge or publication. Supervisor will review/refill after this attempt.

Checkout: `/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T225101Z-1555568/data_drift_mobility/20260926T225101Z-1555576/repo`. Base `fa511ce010120b502f494d717d01b14b8551eed8`. All artifact paths below are relative to `repo/reports/data-drift-api/`; complete gate ledger: `/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T225101Z-1555568/data_drift_mobility/20260926T225101Z-1555576/tests.jsonl`. Prior DI1–3 and DV1–6 evidence remains unchanged in the supplied read-only checkouts; none of their passes is inherited.

Actual `get_recipe(total_steps=None, continuous_policy="dv7"/"dv8"/"dv9")` and `GANTrainer.step` run the learner. Ring fixture:20,000 trainable particles,z2,batch2048,width96×3/Fourier3,seed0; CPU initialization then assigned CUDA. Native nonfused/nonforeach Adam,FP32,deterministic,TF32 off,serial backward,one CPU thread. All initial model/optimizer/RNG receipts match the retained public worker. No seed sweep or coefficient grid.

| Own single-shift evidence | Initial arrival / retention through2400 | Prehold1210–2400 | Changed arrival / retention through4600 | Final shifted suffix / minimumHQ since arrival |
|---|---|---|---|---|
| DV7 |640;177/177|120/120|2730(+330);183/188|182 from2790 / .608887|
| DV9 |580;181/183|120/120|2720(+320);189/189|189 from2720 / .916748|
| DV8 |640;175/177|118/120|2710(+310);171/190|73 from3880 / .075684|

All observations still require eight modes and HQ≥.90, every10 updates with4096 isolated draws. Frozen no-update controls each pass **0/220**. Every departure:

- DV7: none initially; shifted2740,2750,2760,2770,2780. Minimum shifted modes7. Early settling followed182 passes; not rejected for first-touch transients.
- DV9: initial590,600; no shifted departure. Initial minimumHQ.885742, all8modes; initial final suffix610–2400,180checks. Longer retention remains NOT_RUN after its separate broader failure.
- DV8: initial1930,1980, minimumHQ.832275; initial suffix1990–2400,42checks. Shifted2720,2740,2770,2780,2850,2880,2900,2980,3020,3100,3120,3280,3290,3300,3310,3320,3330,3340,3870; minimum2modes. These include separated later collapses, not only early settling.

Complete every-check histories, minima over entire travel windows, comparison endpoint3600 and retrospective suffixes are in `runs/dv{7,8,9}-single/summary.json` and `metrics.jsonl`. No endpoint or deadline was selected after observing the trace.

**DV7 stationary7500:** arrival640,687/687 thereafter; no departures; minimumHQ.921631; final suffix640–7500 spans6860updates.102.492s including setup/reporting.

**DV7 uninterrupted30000:**456.417s including setup/reporting, same unchanged policy/package. Original target:arrival640,537/537 to6000,minHQ.921631.

| Change after | First arrival / delay | Passing/total since arrival | Every departure | Final suffix | MinimumHQ since arrival |
|---|---|---|---|---|---|
|6000|6310 /310|146/150|6330,6360,6370,6390|141 from6400 through7800|.764404|
|7800|8070 /270|1894/1894|none|1894 through27000|.965820|
|27000|27250 /250|276/276|none|276 through30000|.905518|

All post-arrival long-run mode counts remain8. The9000 prefix reports537/537,146/150,94/94 for these first three segments; it is the same uninterrupted run, not another trial. No later stationary collapse was observed. Network mobility closes1017, reopens6007/7807/27008, then closes6754/8542/27748. `dv7-policy-analysis.json` gives the exact rates/state at each event. Rates,noise and controller values are logged for every update. Finite windows support only finite claims.

| Own broader quality | DV7 | DV8 | DV9 |
|---|---|---|---|
| intensity2 |PASS13/24,final12|NOT_RUN|NOT_RUN|
| stripes2 |PASS23/24,final23|NOT_RUN|NOT_RUN|
| bars4 |PASS20/24,final20|NOT_RUN|NOT_RUN|
| blobs4 |PASS15/24,final15|NOT_RUN|NOT_RUN|
| vector_two_broad |PASS18/24,final18|NOT_RUN|NOT_RUN|
| vector_unequal_width |PASS20/24,final20|NOT_RUN|NOT_RUN|
| vector_unequal_mass |FAIL0/24|NOT_RUN|FAIL0/24|
| Total22 |6PASS,1FAIL,15NOT_RUN|22NOT_RUN|1FAIL,21NOT_RUN|

DV7 unequal-mass final minimum covariance eigenvalue ratio **.021303<.15**; HQ.989746 and rare mass ratio.549316 pass. DV9 final eigenvalue ratio **.027236<.15** and covariance error **1.170231>.85** fail; HQ.997559 and mass ratio.317383 do not repair them. The completed vectors use verified canonical CPU parameter fixtures, promoted discriminator cards and CUDA data/scoring streams. Observation noise2303 is the declared isolated branch, distinct from historical global402. Original24-check/final-five bounds are unchanged. Images use the frozen32particle/z8/batch32 architecture and402+step+1901 observations. Their initial model/RNG hashes match the audited retained fixture.

A separate **post-run frozen support diagnostic**, without updates or added quality credit, enumerated all256 G(prior.z) outputs without noise. Both failures allocate exactly2 particle outputs to the rare component. Its centered learned covariance has rank at most1: raw normalized eigenvalues approximately[0,4.953] forDV7 and[0,7.104] forDV9. Fixed output noise contributes only(.029/.18)²≈.02596 per normalized direction. This explains why high HQ/adequate sampled occupancy coexist with the failed spread bound. Target centers appear only in this evaluator diagnostic. See `support-audit/`.

DV7’s original component classifier mistake was not repeated: fixture comparisons use parameter lists, with buffers separate. An ordering deviation is preserved in `batch-order-deviation.json`: unequal_width launched in a combined status-read/launch call before unequal_mass’s failed bound was inspected. The started fixed window completed; no expensive DV7 native/custom qualification followed.

**Mechanisms and interpretation.** DV7 retains DV6’s normalized temporal real-data change detector, payoff mobility and separate surprise brake, and adds critic-rate factor1/(1+e²), where e is EMA.02 of the positive adversarial payoff gap normalized by log2. This acted before799-call critic-reference calibration and repaired the measured image failures. DV8 added a statistically normalized temporal real/fake feature discrepancy to common mobility; the witness remained saturated and destabilized all network roles. DV9 confined that same demand to prior mobility and restored DV7 network control. It repaired the short ring instability but did not allocate enough rare-component support. No real statistic fits/translates outputs or enters a new loss. Real-data evidence alone authorizes memory release; generator mismatch cannot excuse surprise instability.

All retain constant input0/output.029 noise, bounded rolling controller state, native optimizers and trainable particles. No learner reads horizon,change times,centers,labels,task IDs,quality or caller phases. The799-call rule initializes one critic reference; reversible controllers continue at arbitrary ages, demonstrated at27000 forDV7. Smoothing windows never expire learning.

**Validation and limits.**98/98 regression tests PASS;1058 source/artifact/protocol checks PASS. DV7’s24 full-state prefix receipts through2400 match budgets4600/7500. Its own fresh process2400→2500 restores and reproduces every live/EMA observation and full model/optimizer/controller/EMA/RNG/data-stream receipt, with CUDA parameters,gradients,moments; ordinary17 Adam scalar counters remainCPU. Independent supervisor raw-state audits support the long/checkpoint/vector evidence. DV8/DV9 have CPU controller-state tests, not their own fresh-process CUDA qualification; those and their stationary/long runs are explicitly NOT_RUN after measured failures.

The first copied ring worker’s strict DV7 FAIL remains in the ledger as historical diagnostic. Later `measurements_complete` PASS means complete observations, **not quality qualification**. DV8 is rejected on its complete instability history despite successful collection. Frozen-support PASS likewise means diagnostic completion. Ledger:17PASS/3FAIL/0ERROR plus69SKIPPED; overlapping protocols/audits are not independent trials.

Two public component gaps remain: `scale_learning_rates(controller=...)` omits the extra critic multiplier; the detector’s CPU projection initialization assumes a CPU default factory device. Actual GANTrainer applies the correct multiplier, and evaluated host CUDA sampling/scoring scopes exit before step. Eight custom hosts need faithful component transactions retaining their auxiliary losses. Prepared native scripts were never run; their earlier CPU-constructor draft is explicitly superseded by `native100-cuda-host-audit.md`, which establishes canonical CUDA initialization. Do not execute that draft unchanged. Vector checkpoint files omit caller-owned data RNG; their quality evidence does not establish exact vector continuation.

Matched K3P is **NOT_RUN**: supervisor assigned conditional ownership toRP5. No release comparison, full22 pass or infinite-stability claim is made.

Source/declarations: `dv7.json`, `dv8.json`, `dv9.json`; each run’s immutable `source.zip`; exact predecessor receipt and `dv6-to-dv7.diff`, `dv7-to-dv8.diff`, `dv8-to-dv9.diff`. `dv7-library.zip`/`dv8-library.zip` preserve earlier learner bytes. Current final library isDV9 with all variant branches. `final-library.patch` SHA256 `49925937ab451183b3ba239154a45d7c891fe0f0492178ec04e4750fcf8f4388`; hashes in `final-source-sha256.json`. Historical runs replay from their own archives, never a newer branch’s score.

```sh
export CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
PY=/tmp/pr38-default-env/bin/python
# Restore the chosen run source.zip over PR195 in an isolated checkout first.
$PY -u reports/data-drift-api/worker.py --schedule dv7 --protocol single_shift --output <fresh-dir>
$PY -u reports/data-drift-api/worker.py --schedule dv7 --protocol stationary --output <fresh-dir>
$PY -u reports/data-drift-api/worker.py --schedule dv7 --protocol long_continuation --output <fresh-dir>
$PY -u reports/data-drift-api/vector_gate.py --candidate dv9 --task vector_unequal_mass --output <fresh-dir>
$PY -u reports/data-drift-api/remaining_images.py --candidate dv7 --task img_bars4 --output <fresh-dir>
$PY -u reports/data-drift-api/api_checks.py --candidate dv7
# API check expects declared dv7-single and dv7-stationary artifact paths.
$PY -m pytest -q tests/test_training.py tests/test_recipe_defaults.py tests/test_ka2.py tests/test_k3p.py tests/test_serial_backward.py tests/test_data_drift_successors.py tests/test_data_drift_dv7.py
```

Tail:`tail -f repo/reports/data-drift-api/runs/<run>.log` from this attempt directory. All workers have finished.

**Recommendation for the next reviewed attempt:** preserve DV7’s critic balance and demonstrated long-ring/image strengths. Address insufficient local particle support using ordinary adversarial information; global significance-driven rate reopening and scalar prior acceleration did not solve it. Do not infer conditional covariance from HQ, occupancy or mean payoff. Keep the rare-support diagnostic separate from learner input. Resolve component portability and canonical native initialization before future survivor qualification, then earn that successor’s own full evidence. No fourth mechanism was attempted here.
