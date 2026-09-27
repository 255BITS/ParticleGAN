# Three proposals complete; no qualified winner

**Review point reached.** API-DV10, DV11 and DV12 each have an actual public-API ring result and an own broader-quality failure. No fourth proposal, merge, publication or default promotion. Integration target:develop. All workers exited.

Checkout:`/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T232546Z-1609794/data_drift_mobility/20260926T232546Z-1609802/repo`. Base:PR195 `fa511ce010120b502f494d717d01b14b8551eed8`. Artifact paths below are relative to `repo/reports/data-drift-api/`; the complete ledger is `tests.jsonl` beside this report. Read-only DI1–3/DV1–9 evidence, root first-results/broader-results, source audits and shared protocol were inspected and preserved; no score is inherited.

|Research lead rank|Own original arrival; retention to2400|Own shifted arrival delay; passing/total to4600|Final shifted suffix / minimumHQ since arrival|Own broader result|
|---|---|---|---|---|
|1 DV10|620;179/179|+320;184/189|180 from2810 / .679932|unequal_mass PASS7/24; bars4 FAIL8/24,suffix0|
|2 DV12|590;182/182|+290;191/192|187 from2740 / .710449|unequal_mass FAIL2/24,suffix0|
|3 DV11|560;185/185|+290;190/192|187 from2740 / .617676|unequal_mass FAIL0/24|

Ranking identifies limited research leads, not release candidates. All three prehold1210–2400 windows pass120/120; frozen no-update controls each pass0/220. No original-target departures after arrival. Every shifted departure:DV10=2730,2750,2760,2790,2800; DV11=2720,2730; DV12=2730. All post-arrival mode minima are8. Initial minimumHQ:DV10 .905029,DV11 .904053,DV12 .903564. Every observation, full travel-window minima, declared3600 comparison and retrospective suffix are in each `runs/dv*-single/summary.json` and `metrics.jsonl`. No first-touch-perfect or recovery deadline gate was used.

Ring runtime was61.353s/65.723s/1168.141s forDV10/11/12, excluding setup. DV12's exact full-prior distance calculation is materially expensive; its started window completed unchanged.

|Unequal-mass comparison|Passing/24; final suffix|Final covariance error ≤.85|Final minimum eigenratio ≥.15|Final minimum mass ratio ≥.25|
|---|---|---|---|---|
|Released publicK3P0.8.0, own reference|21;16|.181380|.666039|.739934|
|RP5, retained matched-fixture evidence|18;18|.469706|.469114|.717398|
|DV7, retained evidence|0;0|.714704|.021303|.549316|
|DV9, retained evidence|0;0|1.170231|.027236|.317383|
|DV10, own|7;7|.745409|.828991|.292969|
|DV11, own|0;0|1.230177|.160566|.629132|
|DV12, own|2;0|.884388|.624825|.329590|

All six numerical vector gates and all24 observations remain in their result/metrics files. DV12 first passed at1000 and1050, then failed1100/1150/1200; rare-component covariance error3.055543 dominates its final mean. DV11's worst component error is3.801177. Higher HQ or passing minimum eigenratio does not repair those failures.

DV10's bars4 final result is four modes,HQ.8125<.90,8/24 passes and zero final suffix. Its latent bandwidths were.676–.916. DV11/DV12 bars4 are NOT_RUN, so their proposed image repairs have no image-quality credit. Own22-task status:DV10=1PASS/1FAIL/20NOT_RUN; DV11 andDV12 each0PASS/1FAIL/21NOT_RUN. `broader22-status.json` lists every task.

**Mechanisms.** All retain DV7's normalized nonlinear real-data temporal detector, separate payoff-based cold-game mobility, unexplained Adam-gradient surprise brake, and critic-only factor1/(1+payoff_error²). Real features authorize scalar mobility and critic memory release; they never fit/translate samples or create targets. Observed mobility closes/reopens/closes:DV10 977/2407/3149; DV11 1002/2407/3155; DV12 984/2407/3394. Exact applied rates and signals are in `dv*-control-events.json` and every-update logs.

DV10 adds persistent Gaussian latent neighborhoods, bandwidth EMA.01(prior coordinate std×N**(-1/z_dim)). This addresses the predecessor's finite-support rank deficit using learned prior geometry, with G trained only through the adversarial objective. It repairs the rare-vector gate but smears the image task.

DV11 adds a paired clean/full-width critic probe, normalized by sampling variance. A checkpointed reversible trust scalar controls applied jitter; full-width probes remain available when jitter closes. On unequal_mass,trust ranged.005600–1,real data_drive stayed0,payoff error reached1.7233 and mobility never fell below.7499. It failed covariance. No additional fitting loss was introduced.

DV12 returns to DV10's proposal and clips each perturbation norm to half the nearest distinct current particle distance. Exact chunked CUDA distances use the appropriate live/EMA prior. It improves minimum spread but still overdisperses the rare component. Per-update D/G radius ranges,clipping fractions and actual perturbation RMS are logged. No cached graph, evaluator clock, target labels or quality feedback enters this rule.

The entire learner has `total_steps=None`; input/output noise stay0/.029. New latent draws use the private training-noise stream and isolated evaluation streams, explicitly declared before execution. Existing latent-index and caller-data streams retain their host rules. G/D architecture,20,000particles,z2,batch2048,width96×3/Fourier3 and seed0 are unchanged on the ring. CPU initialization precedes CUDA training. Native lazy Adam,FP32,determinism,TF32 off and full-step serial backward are retained. The799-call critic reference bootstrap initializes one reference; subsequent mobility/noise/memory updates remain available at arbitrary ages. Rolling smoothers do not encode an ending or a caller phase switch.

**Supporting K3P reference.** `k3p-vector-reference/` is the unchanged reviewed bundle, manifest SHA256 `4495314edb77e5ac37e612169324d4d7d05e8df73df760fe2213aaf12dade417`. Exact release commit`0ff9a7afe5dcb828239369446cfe71971bce687b`; no package patches. Same canonical256particle/z4/batch128 unequal_mass fixture, promoted critic, host/scorer and Torch2.13/cu126/A6000 runtime. Explicit1200 benchmark horizon uses released120/240 noise milestones and released LR decay; it is not literal get_recipe()7000 or a continuous-learning candidate. Observation noise2303 matches current vector measurements and differs from historical402. First pass150,departure400,final suffix450–1200. Runtime21.413s. All24 observations,initial/final caller-data RNG checkpoints, actual rates/noise, source and native optimizer proofs at1/1200 are preserved. The prerequisite launch was delayed by the supervisor's temporary preparation hold and corrected bundle path; no baseline tuning followed scores.

**Verification.** Final regression130/130 PASS; final source/state/artifact audit699 checks PASS. All three ring model/optimizer/training-stream/global-RNG/caller-data/target-initial hashes match the retained public worker. Saved model tensors and Adam moments are CUDA; ordinary15/17 Adam scalar steps remainCPU. Future vector checkpoints now contain caller-owned data RNG. `scale_learning_rates(...,controller=...,critic=...)` applies the extra critic factor with explicit role ownership; detector projection allocation explicitly usesCPU. These repairs were declared before candidate training.

CPU restoration and sampling-isolation contracts pass. Own fresh-process CUDA continuation and differing-budget prefix experiments are **NOT_RUN**, as are stationary7500,delayed/repeated9000,and long30000 after the measured broader failures. Their unchanged declarations remain in `evaluation-protocols.json`; no ancestor pass is borrowed. The later matched K3P recovery-ring comparator belongs toRP5 and is NOT_RUN here. Eight custom hosts still need faithful public component bindings retaining auxiliary losses. Finite windows cannot establish infinite stability.

Ledger totals:{'PASS': 13, 'FAIL': 3, 'SKIPPED': 78}. PASS rows include measurement completeness and regression/audit gates; they do not mean candidate qualification. Unrun gates are explicit SKIPPED/NOT_RUN. No benchmark ERROR or interrupted evaluation window occurred.

Sources:`dv10.json`,`dv11.json`,`dv12.json`; each run's immutable `source.zip`; `predecessor-source-receipt.json`,`inherited-source.zip`,`predecessor-to-dv10.diff`,`dv10-to-dv11.diff`,`dv11-to-dv12.diff`; per-variant libraryZIPs. Current final library contains all branches. `final-library.patch` SHA256 `0acc52bf495ccce74a1da25f0568aaa33e44a890d585a3c48dfdfdded6ef91b1`; `final-source-sha256.json` binds its sources. Replay historical variants from their own archives.

**Recommendation for the next reviewed attempt:** preserve DV10's demonstrated continuous support, and address local output shape through ordinary adversarial learning. A full-rank neighborhood alone does not provide the correct covariance. Global critic-based suppression lost the vector strength; geometric clipping remained inaccurate and cost19× the ring runtime. Investigate learned local perturbation shape with a computationally bounded implementation, not a scalar gain sweep. Re-earn ring,image,vector and long-run evidence for any successor. This attempt ends at its three-proposal review point.

```sh
# From this repo; restore the chosen run's source.zip over PR195 first.
# run.sh fixes GPU/CUBLAS/thread variables and reads supervisor.md; workers enforce FP32/determinism/TF32 off.
bash reports/data-drift-api/run.sh dv10-replay reports/data-drift-api/worker.py --schedule dv10 --protocol single_shift --output reports/data-drift-api/runs/dv10-replay
bash reports/data-drift-api/run.sh vector-replay reports/data-drift-api/vector_gate.py --candidate dv10 --task vector_unequal_mass --output reports/data-drift-api/runs/vector-replay
bash reports/data-drift-api/run.sh k3p-replay reports/data-drift-api/k3p-vector-reference/worker.py --output "$PWD/reports/data-drift-api/runs/k3p-replay"
# Tail across consecutive worker logs:
tail -F reports/data-drift-api/runs/current.log
```
