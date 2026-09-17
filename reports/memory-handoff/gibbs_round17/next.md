# Round17 handoff

Requested round complete. Read assessment.md, comparison.md, gibbs_comparison.md, diagnostic_assessment.md, and design_review.md. Six2k scouts, both GPUs, all diagnostics complete, zero failures, no qualifying5k extensions. Nothing queued/running. Saved round12 winner unchanged; all full circles0/128. Do not rerun sealed queues.

New default-off helper experiments/memory_gibbs.py integrates GibbsReader and separate GibbsCritic in memory_handoff_scout.py. Config: gibbs_latent_dim (0off), gibbs_steps(totaldecodecalls>=1), gibbs_weight, gibbs_width, gibbs_condition, gibbs_ramp_steps. Opposing same-event inference/generation joints; temporary latent, fixed history/time/particle, no internal writer calls. E/P cooperate, including real E gradients. Main D writer objectives unchanged. Steps>1 use detached warmup/final connected E->P, so h seed projection stays initialized but direct particle learns. Runtime and all old objectives call the same reader. Matching architecture controls weight0.

Important limit: direct particle enters every decoder, so actual innerstate(z,h), while K sees(h,x) conditionedM/time. Deterministic adaptation, not original stochastic GibbsNet or stationary guarantee. Inner stability is separate from outer memory survival. Default RpGAN trains both E branches; do not silently swap to vanilla G loss (would ignore real E score).

Best new gibbs1_joint25 minQ.007228 (+21.7% overmatchedcontrol .005940, -33.7% againstoriginal2k .010901). Three-decode models settle by3decodes and7doesnotimprove local read; their continuation is worse. Every model loses tested radius/speed information by128writes. M still strongly used; h sensitivity small inone-decodejointmodels but substantial inthree-decode. NoMSEobjective: diagnosticerrors/probesonly.

New diagnose_memory_gibbs.py requirescompletedcheckpoints and examines latent/memoryinterventions, correctproducerjoint, and1/2/3/7fixedtimeinferencecalls. Defaults1024freshhistoriesprefix8/32. Botholdinformation/process/transition diagnostics retained. Reports include all panelsandhashes.

Round17 runner is reports/memory-handoff/gibbs_round17/run_round.py --round NAME --diagnostics, addsnewGibbsdiagnostic topreviouspipeline. Existinground15 assess.py/audit.py reused. Static tail unchanged: tail -F runs/memory_path/core_round1/train.log.

112focusedtests passed,2GPU4stepsmokes,completedcheckpointdiagnosticsmoke. Source files frozen during scouts and panelidenticalbaseline. This round and prior15/16 remainuncommitted/unpushed. Preserve unrelated.claude/,results/motion/,sparse-ucd.log. No next experiment selected; recommendations inassessment requirediscussion. Userconstraints persist: noseedrepeats, noMSEtraining, nobulk temporalrollouts, noanalyticcursor, publicdefaultBcap, fixedparticlepertrajectory, D-ownedmemory, noexpert runtime, configdriven, metricsaftercompletion.
