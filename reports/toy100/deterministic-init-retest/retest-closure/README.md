# Bounded retest coverage

Scope is frozen at97 reviewed research cases (85 grouped + KA2 +11 simple). At 2026-09-27T03:29:03.680135+00:00, 53/97 are archived and independently audited; 44 still await execution or archive. This is not a claim that the remaining batches have finished.

All49 API/control quality windows are complete (46 historical configurations, including the explicitly scoped RP1 eager diagnostic, plus3 public controls):4 PASS and45 FAIL. Four earlier logging ERROR records remain preserved. A passing tiny screen does not override a later breadth failure.

The190 research definition rows comprise97 reviewed execution cases,4 exact duplicate definitions, and89 NOT_RETESTED rows. Another15 rows are explicit priority aliases. These mappings plus46 API configurations and3 public controls account for all254 coverage rows.

| NOT_RETESTED reason | Definition rows |
|---|---:|
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER | 50 |
| NOT_RETESTED_MISSING_EXACT_SOURCE_BINDING | 30 |
| NOT_RETESTED_OLDER_PACKAGE_INTERFACE | 8 |
| NOT_RETESTED_UNTESTED_DRAFT | 1 |

These gaps are preserved as untested, not failed quality. Existing source availability does not prove that a different host wrapper preserves initialization and learner arithmetic. No new adapter or source reconstruction is authorized by this closure.

| Source/binding group | Rows | Historical labels |
|---|---:|---|
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe ee83c8ad1d94 / binding ee06ca1a4bee | 1 | c01_coherence |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe a3d3ce67d168 / binding a3d2646caf4a | 2 | c02_constant_stationary, c03_energy_ratio |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe cb746b1ccf09 / binding e64a049aa747 | 1 | c1_loss_envelope |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe 521091e05863 / binding 557cdb5591ba | 2 | c2_motion_hysteresis, c3_directed_motion |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe eb79c0d4537c / binding 9bea8e6be9f2 | 6 | gd1_predictive_scheduled, gd2_predictive_stationary, gd3_bounded_reversal, nm1_actual_displacement, nm2_critic_actual_displacement, nm3_critic_shrink_only |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe 13729130c240 / binding dae1010329d5 | 3 | shared_coordinate, shared_geometry, ra_r1r2_shared_coordinate |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe 025ad6a9b319 / binding ee675c684d50 | 4 | exposure_trust, exposure_mean, visit_isotropic, ra_r1r2_exposure_mean |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe a02621b46329 / binding c8e05114de45 | 1 | density_mobility |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe 411d9e5c7845 / binding 195797057320 | 2 | optimistic_adam, optimistic_gradient |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe cff86fa476e2 / binding 31e0df864170 | 3 | predictive_critic, joint_lookahead, extragradient |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe 68f7ec7ec1d6 / binding f882d86c5602 | 1 | optimistic_critic |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe e6743c601bdd / binding 36d9b3aa8a79 | 2 | c01_condition_r1_data_cap, c02_conditional_cap |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe 66a118d1ea46 / binding 42247cd148bf | 1 | c03_conditional_cap_original_adam |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe 7c31326166b8 / binding c1e24acb39c4 | 3 | c01_real_cap_linear, c02_hybrid_release_linear, c03_gradient_budget_hybrid |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe d4375bab9e71 / binding 6deb39a52728 | 10 | dimension_rms_hybrid, dimension_rms_bcap, detached_fake_scale_cap, real_deadzone_010, real_smooth_quartic, real_smooth_sextic, r1_smoothstep256, r1_smoothstep128, r1_floor01_smoothstep128, dimension_rms_hybrid |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe e76b9ee23c3f / binding 04e10d848b56 | 1 | prior_recent_rms |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe e5b3a7f924d6 / binding 17168128b2e0 | 1 | coherent_particle_response |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe e5b3a7f924d6 / binding 508ea46b5c09 | 2 | direct_particle_response, direct_particle_response |
| NOT_RETESTED_MISSING_EXACT_SOURCE_BINDING; probe unbound / binding unbound | 30 | rg5_reversal_gated_guard, baseline-baseline, pr107-reachstall, pr117-trans, pr119-reachstall_ddexit, pr120-reachrecover, pr121-reachstall_sgame, pr122-reachstall_trustfall, pr123-reachstall_extrad, pr124-zeromean, pr125-reachstall_ghalf, pr126-meanrestore, pr128-ema, pr129-reachstall_drop, pr130-reachstall_rollback, pr132-reachstall_gsustain, pr133-reachstall_trustopen, pr134-reachstall_adv, pr135-reachstall_modettur, pr136-reachstall_gbeta0, pr137-cmnull, pr138-reachstall_post_g125, pr140-delayg05, pr141-delayed_g125, pr142-delayextrad, pr143-holdw15, pr144-stall_cf25, pr82-pr82, pr84-pr84, pr93-pr93 |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe 76f335544c04 / binding 0e222a7cb31f | 1 | eg1_scheduled |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe 6f727479d5f4 / binding ca080108242e | 1 | rg5-a2 |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe 6f727479d5f4 / binding 1cfd2b8571ee | 1 | rg5-bcap |
| NOT_RETESTED_OLDER_PACKAGE_INTERFACE; probe unbound / binding unbound | 8 | eps_net_1m, H, shared_rms, shared_column_rms, shared_rms_average2, PR107, PR140, PR143 |
| NOT_RETESTED_NO_REVIEWED_INITIALIZATION_ADAPTER; probe MISSING / binding fdf9120441e9 | 1 | constraints_simple_regularization |
| NOT_RETESTED_UNTESTED_DRAFT; probe e8653d7e4502 / binding 845bf475fb39 | 1 | PD2 |

The [frozen scope](coverage-scope.json) lists every exact case, duplicate mapping, alias and untested reason. [Progress](coverage-progress.json) binds current archive/audit hashes. The frozen scope cannot be expanded by rerunning this report.

Refresh evidence status only with `python reports/toy100/deterministic-init-retest/build_retest_closure.py`. Use `--require-complete` for the final closeout; it fails until all97 fixed research cases have independently audited archives. This command never runs training, changes a score, launches a worker or stops a process.
