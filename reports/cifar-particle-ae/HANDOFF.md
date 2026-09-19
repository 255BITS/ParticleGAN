# CIFAR AE-GAN: duration experiments and particle-scaling research

## Latest: plain deconvolution scout, September 18 at 19:22 MDT

User requested `linear -> deconv -> deconv -> deconv -> tanh` with **no normalization**, 16k particles, GPU 1, and explicitly authorized a subagent to implement it. Implemented by `/root/plain_deconv_impl`: new standalone `experiments/train_cifar_ae_deconv.py`, 298,595 G parameters, preserved historical source files. Ten tests passed including E-only isolation, D/E initialization RNG equivalence and exact full-state split resume. Actual GPU 1 pipeline smoke passed with 16,384 particles, 16 updates, lazy double backprop, FID128/reconstruction and checkpoint writing; smoke scores are not benchmarks.

**Active GPU 1:** pipeline PID273670, `experiments/cifar_ae_deconv_scout.py`, scratch G/E/D head and 16,384 independent centers, fixed sigma0.212616428732872 (same as previous), otherwise existing recipe. Train0->40k, FID50k every5k, retain checkpoints, no automatic promotion. Frozen pretrained ResNet18 D features retained. Plan/config/launch under `deconv_16k_scout`. Follow `tail -F runs/cifar_particle_ae/deconv_16k_scout/PIPELINE.log`. Historical CNN16k40k/80k scores17.0982/15.7527 are contextual references: those expanded a trained1k prior at10k, so this scratch scout is not a matched architecture ablation. Read results and grids before deciding next steps. Sibling information is not applicable to independent scratch centers.

**Active GPU 0:** unchanged existing CNN16k continuation to160k, pipeline PID271619; about135k at19:20. **Completed:** CNN32k80k plus endpoint probe; FID16.2461, best16.1271 at50k, density0.59478, coverage62.23%, sibling information3.74238/5bits. Best overall remains CNN16k80k15.7527. Earlier active/idle descriptions below are historical and superseded by this section.

Branch `feat/cifar-ae-gan-pretrained-encoder`. Latest user authorized32k40k→80k and asked us to choose64k versus longer16k for the other card. Chose16k80k→160k. Launched2026-09-18: GPU0 pipelinePID271619, GPU1PID271620. Latest snapshot:16k at123,100/160k;32k training finished80k and endpoint probe running. See final section for current curves and research framing. Prior16k80k/32k40k and all endpoint probes completed/certified. New bestFID50k15.7527 at16k80k;32k40k16.3451. Current plan `particle_duration/PLAN.md`, launch `particle_duration/LAUNCH.json`; completed review `particle_next/FINDINGS.md`, curve `particle_next/curves.png`. Tail `runs/cifar_particle_ae/particle_duration/PIPELINE.log`. Historical active/idle statuses below are superseded by this paragraph and the final section. No further promotion queued.

Read `particle_information/FINDINGS.md`, `LEADERBOARD.md`, `scaling_metrics.png`; `particle_scaling_scout/FINDINGS.md`, `LEADERBOARD.md`, `CHECKPOINTS.json`; `particle_expansion_40k/FINDINGS.md`. Full prior history archived in `HANDOFF_INFORMATION_PENDING.md` (contains stale running/queued statuses; this file supersedes them). Earlier research archives: `HANDOFF_BALANCE.md`, `HANDOFF_DISCRIMINATOR.md`, `HANDOFF_TRANSGAN.md`.

## Matched particle-count scout outcomes

Same original CNN E-only10k checkpoint, initialFID50k19.4482. G/E.0003,prior.003,D.00045; oneDupdate, bcapcoeff1every8×8, EMA.995. Counts expanded with matched initial distributions and saved sigma/d0, reference-count standardization and regularizer corrections, copied prior Adam moments, saved separate child RNG. Architecture/backbones unchanged.1024/4096 benchmarks reused for8192/16384, no repeat seeds.

| Particles | FID15k | FID20k | Train minutes | Additional sibling bits20k | Density20k | Coverage20k |
|---|---:|---:|---:|---:|---:|---:|
|1024|19.8699|19.8932|7.25|0|0.6281|58.13%|
|4096|18.9357|18.5207|7.37|1.1166/2|0.6344|60.28%|
|8192|18.6479|18.1602|7.28|1.7195/3|0.6352|60.29%|
|16384|18.3401|18.0136|7.46|2.2756/4|0.6294|59.43%|

FID improves at both sampled steps with count, but marginal20k gains diminish:1.3724,0.3605,0.1466. Larger counts produce more decodable feature information; restricted decoder recovers~56–57% of available additional label bits. Coverage is flat beyond4096 and lower at16384, density~0.63. More distinguishability alone does not establish greater useful semantic diversity. One trajectory per count does not estimate training-run uncertainty. Training costs remain close.

## Completed4096 versus1024 duration check

| Particles | FID25k | FID30k | FID35k | FID40k |
|---|---:|---:|---:|---:|
|4096|17.8876|17.4796|16.5033|17.2350|
|1024|21.8973|21.0193|21.3182|26.0216|

Best observed is16.5033 at35k (3.5033 above target13); final4096 checkpoint17.2350. Do not call its trajectory monotonic or treat selected minimum as endpoint.4096 remained better than1024 at all evaluations. Both40k sample grids inspected: variedoutputs with shape/detailerrors, no total-collapse claim.

4096 from35k→40k: density0.6303→0.6148, coverage62.58%→61.39%, decodablebits1.4938→1.4854, decoderaccuracy90.63%→90.44%. FID rebound coincides with declining real-distribution proxies while sibling identity remains distinguishable. The0.0084bitdifference is not meaningful versus parentSE0.062/0.098; one40k parent has98%accuracy but−1.253bits due to overconfident validation-selected decoder and rare testmistakes. Retainnegative scores, do not tune ontest.20kscalingarms have no negative per-parent observedestimates.1024 at40k density0.4662,coverage45.14% despite featurevariance/real1.096: totalvariance alone is insufficient.

## Previous continuation plan (completed; latest results below)

Continue8192 and16384 from respective20k checkpoints to40k on the twoGPUs; compare with the existing4096 duration trajectory. Difference between8k/16k at20k is too small to pick a clearwinner, and eachcenter receives fewer direct samples atlarger counts. Reuse4096 benchmark, no unnecessaryretrain. Do not automatically escalatecount or promote200k solely onbits. User subsequently authorized this continuation; it has now completed.

Endpointpaths/hashes: `particle_scaling_scout/CHECKPOINTS.json`;4096/1024 duration35k/40k in `particle_expansion_40k/CHECKPOINTS.json`. Original10kparent `runs/cifar_particle_ae/transgan_scout/cnn_e_only/checkpoint_010000.pt`, SHA256 `d75fca4bc42ec09f1423ce1a671b4cbd10caefe0abccae3ac2bdb05d5d93237c`.

## Information diagnostic implementation and validation

User explicitly requested subagent; `/root/particle_information_metrics` implemented standalone `experiments/probe_cifar_ae_information.py`, tests in `tests/test_cifar_ae_information.py`. Root reviewed and added `cifar_ae_information_pipeline.py`, `queue_cifar_ae_information.py`. Implementationcommit1e95c36. Historical/shared certified sources untouched. FiveCPUtests pass; real4096 GPU smoke passed; seven full probes certified. Countscout CUDApreflight7/7 and both8-step smokes also passed.

Sevenpoints:1024/4096/8192/16384 at20k;1024/4096 at40k;4096 at35k. Locked/validated identical reference tensor for10000CIFARtrain images,10000generated samples percheckpoint,k5, fixedtorch-fidelity Inception2048. ExactchunkedFP32distances; generator/extractorTF32disabled (historicalFIDgenerator oftenenabled, noted). Classifier/ANOVAfloat64. Parent/frozenstatehashes and sourcearchives checked. NoFIDrecomputed; external50kscores takenfrom exactstepmetrics.

Same32originalparents,64train/32validation/64test examplesperchild,16separatevariancedraws. Train-onlycentroids/diagonalvariance; validationselectsshrinkage/temperature includinguniformfallback. Testneverusedforselection. Bits=log2K−heldoutCEbits; decoder-dependent estimatedconditional-MIlowerbound, not exactentropy or guaranteedfinite-samplebound. Baselinezero meansnoadditional siblinginformation, notzeroimageinformation. Allidentical-featureclonecontrols~0; shuffledlabelcontrols−0.0017to0. NestedANOVAfractions finite-sampledescriptive, notsemanticmodecounts. Morebits canreflectartifacts; weakbits canreflectlimiteddecoder.

Summary fields in `particle_information/results.json`: `information.observed/shuffled_labels/identical_clones` eachhasdecodable_bits,test_ce_bits,test_accuracy,parent_standard_error; available_sibling_bits andper_parent arrays. `variance` has within_child/between_siblings/between_parents fractions. `density_coverage` includesdensity,coverage,feature_variance_trace_ratio_to_real,feature_mean_distance_squared. Featurepanelscached in eachrun/features.pt allowfutureCPUanalysis; realcache in results/cifar_ddgan/information_cache. Preserve rawmeasurements/certificates if changingestimator later.

Completedlogs: `runs/cifar_particle_ae/particle_scaling_scout/PIPELINE.log` (queuePID267443, complete16:41:59MDT2026-09-18); `runs/cifar_particle_ae/particle_information/PIPELINE.log` (queuePID268218, complete16:46:09,7probes~3.8min). Avoidtailing rawCOMPLETElines into modelcontext (largeper-parentJSON); readleaderboards orselectfields.

Userpreferences: no seedexperiments; efficientcommunication; tail-ablelogs; completedleaderboard/explanations/recommendations. Branchonly. Unrelateduntracked `.claude/`, `results/failures.txt`, `results/hopfield*`, `results/motion/`, `runs/`, `sparse-ucd.log` preserved. Sources remainpinned: usenewstandalonetrainers forfurtherchanges toavoidinvalidating certificates. Expansionsnum_particlesconfig remainsreference1024; expansion_factor4/8/16 determinesliverows. Only freshfactor1expansion orunchangedsamefactorresume is implemented; do not naivelyrecalibratecoincidentclones.


## Completed8192/16384 extension (launch details retained)

PID269541, launched17:26MDT2026-09-18, `experiments/cifar_ae_scaling_extend.py`. Track `particle_scaling_40k`; GPU0split_8192,GPU1split_16384. Actual restoration step20000 and first update20001 confirmed; original source/checkpointcertificatesverified; expansion_audit.intervention=false forunchangedsamefactors. Parent hashes in `particle_scaling_scout/CHECKPOINTS.json` and launch in `particle_scaling_40k/LAUNCH.json`.

`tail -F runs/cifar_particle_ae/particle_scaling_40k/PIPELINE.log`

20kadditionalupdates to40ktotal, FID50kevery5k at25/30/35/40k, numberedfullcheckpoints. Unchangedrates/noise/EMA/oneD/E-onlyrecipe. Existing4096 durationcurve reused, no retrain. Trainer `train_cifar_ae_scaling.py` unchanged; existing7CUDApreflighttests included exactresume for8/16, so no redundanttesttraining.

Pipeline automaticallycertifies/results/leaderboard/plot, then probesboth40kendpoints using existinginformationtrainer withfullbudgets. Diagnostictrack/report `particle_scaling_40k_information` includescomparisonwith cached4096 at40k andsameexactrealfeaturetensor. Estimated20–25min training+diagnostics. Followoverall `runs/cifar_particle_ae/particle_scaling_40k/launcher.log` forstagefailures; diagnosticprogresslog appearsinitsowntrackaftertraining. No furtherpromotionafter40kisqueued. OncompletioninspectFIDcurve/samplegrids and newinformation/qualitytable, thenupdatehandoff/reportto user.


## Latest completion review:8k/16k at40k

Alltraining and endpointprobes passedcertification. PipelinePID269541 exited successfully. GPUs0/1idle. No follow-upjob launched. Fullreports: `particle_scaling_40k/{LEADERBOARD.md,FINDINGS.md,results.json,CHECKPOINTS.json,curves.png}` and `particle_scaling_40k_information/{LEADERBOARD.md,FINDINGS.md,results.json}`.

| Particles | FID25k | FID30k | FID35k | FID40k | Additionaltrainmin |
|---|---:|---:|---:|---:|---:|
|4096reused|17.8876|17.4796|16.5033|17.2350|14.66|
|8192|17.5680|17.6849|17.7297|17.7683|14.43|
|16384|17.3601|17.5677|17.4324|17.0982|14.77|

16kbest40kendpoint, butonly0.1368betterthan4k anddoesnotbeatbestobserved16.5033at4k35k.8k stopsimprovingafter25k;16kimproveslasttwoobservations.16kendpoint4.0982above13target. Bothfinalgridsinspected,variedobjectswithshape/detailerrors,no totalcollapseclaim. Identicalrates,fulloptimizer/EMA/RNG,frozenfeatures/sigma/sourcecertchecks passed. Canonicalendpointpaths/hashes inCHECKPOINTS.json.

Endpointinformation:8k1.9164/3bits,density0.5956,coverage59.15%;16k2.6886/4bits,density0.6160,coverage62.17%.4kreference1.4854/2bits,density0.6148,coverage61.39%.16kbeats8konallthreequalitycomparisonmeasures(FID,density,coverage), butitsadvantageover4k40kis small.4k35kstillhasbetterFID16.5033,density0.6303,coverage62.58%. Newdecoder parentSE0.0913/0.0971,allobservedper-parentbitspositive;shufflecontrols−0.00077/−0.00149,clones~0. SamecachedrealfeatureSHAverified. Conditionalbits rise from20k inbotharms, whilequalitydoesnotimproveproportionally.8kdensity/coveragedecline;16kcoverageimprovesbutdensitydeclines. Do notinterpretbitsasexactentropy,semanticcoverage,oraguaranteedscalinglaw.

Recommendednext,NOTlaunched:16k40k→80k bounded durationtest,FID50kevery10k,endpointmetrics. Thisfollowsitsrecenttrend;8khaslittlemomentumatthissetting. Do notautomaticallyescalatecountor200k. Userlatestaskedcompletionstatus,notyetapprovedthenexttest. Preserveprevioussourcecerts;existingtrainerunchangedcanresumesamefactor16.


## Active next experiments (supersedes prior recommendations/status)

16k uses unchanged certified `train_cifar_ae_scaling.py`, resumes checkpoint40k SHA507f0795cbb9d6ad10612c6f6cde1eb2a94c6c65a4f3ec25bde8f1b47efc4867; FID50k at50/60/70/80k. 32k uses new standalone `train_cifar_ae_scaling32.py`, expands original10k parent SHAd75fca4bc42ec09f1423ce1a671b4cbd10caefe0abccae3ac2bdb05d5d93237c; initialFID50k and15/20/25/30/35/40k evaluations. Live count32768 = reference num_particles1024 × expansion_factor32. All other recipe settings unchanged. Compare32k to existing16k at matched40k (FID17.0982), and16k80k to own40k endpoint.

New trainer differs from historical scaling trainer only in description/allowed factors/error message. Four CUDA tests passed: exact unchanged control, actual parent factor32 mapping/output/Adam, exact full resume and reconstruction gradient recipients, CPU mapping. Training8-update smoke certified. New standalone information32 probe pins new trainer and preserves prior child0–15 noise draws while adding a collision-free seed band for children16–31. Six CPU tests passed, including seed uniqueness across parents/splits/children; actual32k GPU probe smoke certified. No historical/shared source edits.

Orchestrator `experiments/cifar_ae_particle_next.py --arm {16k_80k,32k_40k}`. Separate runner roots/logs, combined append-only PIPELINE.log. Per-arm launch logs under runs/cifar_particle_ae/particle_next. Full training reports go to particle_16k_80k and particle_32k_40k; probes to corresponding _information report directories. After completion inspect certified FID trajectories, sample grids, and diagnostics; summarize leaderboard, interpretation, recommendations. Do not use smoke FID128 as a benchmark or call unequal step endpoints a matched count comparison.


## Latest completion and new duration stage

Completed16k40k→80k FID50/60/70/80k:16.5965,17.4222,16.8014,15.7527. Completed32k10k→40k FID15/20/25/30/35/40k:18.5834,17.3152,16.9717,16.3545,16.2066,16.3451.16k80k new best,2.7527 above13target;32k improves matched16k40k by0.7532 but40k rebounds from35k. Both sample grids inspected: varied objects, residual shape/detail errors. Train minutes29.17/22.59 for40k/30kupdates, respectively.

Probes certified:16k80k3.1149/4bits,density0.6469,coverage64.26%;32k40k3.4838/5bits,density0.61684,coverage62.17%. Same cached real-feature tensor. Clone and shuffled controls nearzero.16k improves all three versus its40k;32k40k has morebits/lowerFID but almostunchanged density/coverage comparedwith16k40k. Rawreports particle_16k_80k{,_information},particle_32k_40k{,_information}.

User authorized next32k80k and delegatedothercard;chose16k160k on evidence of latestqualityimprovement and towards200k durationgoal. Neworchestrator `experiments/cifar_ae_particle_duration.py`, --arm16k_160k or32k_80k. Trainers/probes unchanged and pinned. Parents16k80kSHA9ddcef1fb0bf82c47a581fa1b2d58872315d5d67b302eadc533c0abc41b7d5c7,32k40kSHA533ff24ebb99cab375c46b19af58a4bff25a21848ed0c93c829a2dc3a25a6473. FID50kevery10k, checkpoints retained, automaticendpointprobes. Estimated65–75minGPU0,35–40minGPU1. Compare32k80k to completed16k80k;160k16k is separate durationtest. Do notqueue64k or200kautomatically.


## Compaction checkpoint: latest research framing and live results

User says they will compact and continue. No request to stop jobs or launch anything new. Preserve current jobs. Latest snapshot after~36min of duration pipelines:16k at123100/160000;32k training finished80000, full information probe still running onGPU1.

16k newFID50k:90k15.7901,100k16.4609,110k17.3499,120k18.1365. Recent curve is worsening; do not repeat the earlier optimistic late-improvement assessment as current evidence. Best retained16k checkpoint remains80k15.7527. Authorized160k job continues; no early-stop action requested.32k newFID50k:50k16.1271,60k16.4369,70k16.6663,80k16.2461. Best32k now50k16.1271;80k endpoint does not beat16k80k15.7527 (difference+0.4934). Matched40k count advantage did not persist at80k.32k training report/cert/checkpoint inparticle_32k_80k; endpoint diagnostic underparticle_32k_80k_information is pending.

Latest scientific discussion: user is bothered/interested that ordinary Gaussian-input GANs have an implicit fixed-sigma MoG count1, while increasing particle count improves ourFID. They are especially interested in techniques that could have worked5–10years ago: cheap parameterizations that make practical learning easier, not merely theoretical expressivity or modern architectural advances. Respond to this interest directly; do not dismiss it with “G can represent the transformation anyway.”

Relevant precedent discussed: DeLiGAN (CVPR2017) learns mixture means and scales. Our approach comes from particles (user likened it to “elo”; do not invent an expansion of that acronym), learns center positions with fixed sharedsigma, and includes multiple encoder formulations. Shared learned-MoG ingredient does not make formulations identical or establish novelty. Source: https://openaccess.thecvf.com/content_cvpr_2017/html/Gurumurthy_DeLiGAN__Generative_CVPR_2017_paper.html . User emphasizes DeLiGAN recognized importance ofMoG; our particle perspective/count scaling/inference machinery remain the relevant distinctions.

Model explanation given in terminal-friendly form: E(x) outputs normalized continuousqueryq and64Doffsetu; k=nearestparticle(q); z=mean[k]+sigma*3*tanh(u/3); reconstructG(z). Hardnearest selection forward, softdistance-weightedcenter surrogate backward. E does not choose sigma; fixed sharedsigma~0.212616. Generation bypassesE: randomuniformparticleindex + sigma*standardGaussian noise ->G. Current E-only reconstruction updatesE only, so cannot directly move particles/G or cause L2 averaging inG. Adversarial objective and priorregularization update generative side. Gaussian-input GAN is conceptuallyonecomponent, but current normalization implementation is not a testedliteralK1 configuration.

Count scaling adds cheap parameters:32,768×64≈2.1million centerparameters. Exactnearest search uses batch×center matrix multiplication, not a batch×center×dimension tensor.16k→32k costs~3–4% trainingthroughput in measured runs. Duplicate centers preserve initialdistribution, then independentcenters/optimizerhistories evolve; capacity and optimization remain entangled. Moreinformationbits do not prove semanticcoverage. Existing frozenpretrainedD and priorregularization may condition the effect.

Potential future mechanism tests discussed ONLY as ideas, not queued/authorized: checkpoint fork freezing versus movingcenters; eventually test count benefit with ordinarytrainedD to assessgenerality/oldertechnologycompatibility. User previously prioritizescountscaling and dislikes detouring intoGaussianbaseline/repeatedseeds; currentauthorized durationjobs takepriority. No automatic64k or200k launch. After currentjobs finish review fullcurves, certificatediagnostics, samples, leaderboard, explanations and recommendation.
