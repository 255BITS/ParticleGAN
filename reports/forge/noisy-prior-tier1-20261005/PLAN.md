> Archived type-only plan. Completed results are in [README.md](README.md). Authenticated message717 supersedes this scientific plan with separate positive Noisy025 and existing-MoG tracks. The command below requires the documented copied-module AE admission handling; it is not a one-command four-case reproduction.

# Atlas with NoisyParticlePrior: fixed Tier 1 experiment

This isolated experiment asks whether the public NoisyParticlePrior type supports the fixed toy hosts and the full Atlas controls, then whether the runnable hosts pass their original numerical gates. It freezes one Atlas configuration and seed 0. It changes the prior type while preserving the parent sampling widths, networks, data, initialization, losses, observation clocks and budgets.

| Original problem | Question | Original gates | Updates | Wall cap |
| --- | --- | --- | ---: | ---: |
| gaussian1d_acquisition | Acquire the declared one-dimensional Gaussian with its location and spread. | sample_count >= 4096; finite_fraction == 1.0; mean_error_sigma <= 0.2; std_ratio >= 0.8; std_ratio <= 1.2; cdf_ks <= 0.05 | 1,000 | 120s |
| two_pole | Move the original twelve zero-start coordinates while keeping stored critic gradients bounded; both-pole balance is ungated. | mean_abs >= 0.3; grad_med <= 1.0 | 80 | 300s |
| unused_token_hold | Fit the active embedding while protecting the coupled unused embedding. | unused_hold >= 0.85; concept_move >= 0.85 | 200 | 300s |
| ae_gan_hold | Reconstruct the fixed target while keeping mean anchor-to-nearest-generated distance within its hold bound. | recon_mse <= 0.05; hold <= 0.35 | 250 | 300s |
| ring16_acquisition | Acquire all sixteen declared components with balanced masses and the required per-component geometry. | sample_count >= 4096; modes >= 16; mass_tv <= 0.15; hq >= 0.85; component_covariance_error <= 0.85; component_min_eigen_ratio >= 0.15 | 400 | 300s |
| five_word_joint_acquisition | Cover the five-word vocabulary with balanced mass and paired inverse reconstruction. | sample_count >= 1024; quality_fraction >= 0.95; modes == 5; mass_tv <= 0.1; reconstruction_exact == 1; minimum_reconstruction_token_probability >= 0.9 | 20,001 | 900s |

Every accepted numerical result requires all 24 original scored observations. PASS additionally requires the original terminal run of at least five passing checks; a complete FAIL remains a measured result. The six caps total 2,220 seconds; this is an allowance, not measured cost. Four owners are implemented (Gaussian, two-pole, AE and ring), with 1,020 seconds of combined caps. The two unsupported cases stay in the six-row denominator and spend no training time.

The actual class is particlegan.noisy_particle_prior.NoisyParticlePrior, task kind noisy_particle_cloud, factory kind noisy_particles. Gaussian, AE and ring retain absolute latent sigma .025 and unstandardized tables, preserving their original mixture emission law. Two-pole and the declared unsupported variants retain sigma zero; direct coordinates consume no kernel draw. This tests type/control compatibility. It cannot establish that adding more latent noise caused recovery. Learned or scheduled output noise and serving behavior remain separate and unchanged.

The fixed five-word owner has conditional rows and N5, which conflict with the requested independent Atlas controls and the birth/death minimum neighborhood. The unused-token owner has a coupled shared vector and only two routed slots, with no latent N12 bank; inventing rows or sampling slots would change the problem. Those variants are BLOCKED under this fixed-input experiment, without a numerical failure or a conclusion about general family capacity.

The ordinary Forge Queue preserves leases, original deadlines, frozen source and independent grading. It completes runnable peers in this tier even after a numerical FAIL. Owner/evidence INVALID stops further admission; already active jobs finish. There is no scientific retry, tuning, seed sweep or deeper-tier promotion in this campaign.

~~~sh
python -m experiments.forge run atlas \
  --view noisy_prior_tier1_686_v1 --through-tier 1 \
  --device cuda --cuda-model 'NVIDIA RTX A6000' --gpus 1 \
  --campaign configs/forge/campaigns/atlas-noisyprior-tier1-686-v1.json
~~~

This command uses the maintained common-repository Queue. Choose an available matching GPU and keep one worker per GPU; the CPU tasks retain one thread each. A copied-source initializer metadata check must precede AE reservation. The experiment readout will retain all six statuses, original scalar metrics, source/recipe/runtime/init provenance, physical costs and goal GIFs made from already scored observations. Target contours are declared analytic geometry, not sampled target clouds.

Results belong to this explicit prior-substitution cohort. They do not change original qualification rows, the family leaderboard, a speed ranking or shipping defaults. Review the completed branch before any develop integration.
