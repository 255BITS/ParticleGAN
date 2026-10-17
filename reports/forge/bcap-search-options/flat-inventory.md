# Complete flattened Recipe inventory

Generated from the public dataclass by [audit.py](audit.py). All 92 fields are included,
including task conditions and inactive policy settings. This is an inventory, not a
search declaration. `null` means inherit for role moments/epsilon or disable for optional
schedules. Pair components must be reassembled into their parent Recipe field in a grid.

`Grid` means whitelist membership only; activity and mechanism checks still apply.
Defaults are public presets before Forge binds task conditions. See [the report](README.md)
for bounds, algorithm applicability, effective task laws and the Forge v1 preset distinction.

| Flat path | bcap default | bcap_adam default | Owner | Grid |
| --- | --- | --- | --- | --- |
| `name` | `"bcap"` | `"bcap_adam"` | technique | no |
| `critic_formulation` | `"bcap"` | `"bcap"` | technique | no |
| `model` | `"gan"` | `"gan"` | technique | no |
| `z_dim` | `2` | `2` | task | no |
| `num_particles` | `20000` | `20000` | task | no |
| `prior_kind` | `"particles"` | `"particles"` | task | no |
| `sigma_rel` | `0.0` | `0.0` | task | no |
| `standardize` | `true` | `true` | task | no |
| `num_classes` | `null` | `null` | technique | no |
| `conditioning` | `"scalar"` | `"scalar"` | technique | no |
| `ucd_target` | `"class"` | `"class"` | technique | no |
| `ucd_weight` | `0.02` | `0.02` | hyperparameter | no |
| `alpha_bar[0]` | `1.0` | `1.0` | hyperparameter | no |
| `alpha_bar[1]` | `0.9` | `0.9` | hyperparameter | no |
| `alpha_bar[2]` | `0.5` | `0.5` | hyperparameter | no |
| `alpha_bar[3]` | `0.05` | `0.05` | hyperparameter | no |
| `alpha_bar[4]` | `0.0001` | `0.0001` | hyperparameter | no |
| `batch_size` | `2048` | `2048` | task | no |
| `total_steps` | `7000` | `7000` | task | no |
| `continuous_policy` | `null` | `null` | technique | no |
| `lr` | `0.012` | `0.00425` | hyperparameter | yes |
| `d_lr_mult` | `1.5` | `1.0` | hyperparameter | yes |
| `prior_lr_mult` | `2.5` | `2.0` | hyperparameter | yes |
| `betas[0]` | `0.0` | `0.0` | hyperparameter | yes |
| `betas[1]` | `0.999` | `0.999` | hyperparameter | yes |
| `prior_betas[0]` | `null` | `null` | hyperparameter | yes |
| `prior_betas[1]` | `null` | `null` | hyperparameter | yes |
| `reg_arm` | `"b_cap"` | `"b_cap"` | technique | no |
| `reg_coeff` | `1.0` | `1.0` | hyperparameter | yes |
| `reg_kappa` | `1.0` | `1.0` | hyperparameter | yes |
| `reg_every` | `1` | `1` | hyperparameter | yes |
| `prior_reg` | `0.0` | `0.0` | hyperparameter | yes |
| `ema_decay` | `0.0` | `0.0` | hyperparameter | no |
| `lr_anneal_start` | `0.6` | `0.6` | hyperparameter | yes |
| `lr_floor` | `1.0` | `1.0` | hyperparameter | yes |
| `network_lr_floor` | `1.0` | `1.0` | hyperparameter | yes |
| `network_lr_horizon_cap` | `null` | `null` | hyperparameter | no |
| `reg_anchor_min_decay` | `0.9` | `0.9` | hyperparameter | no |
| `reg_anchor_weight` | `0.0` | `0.0` | hyperparameter | no |
| `direct_particle_gain` | `false` | `false` | technique | no |
| `d_guard_ratio` | `0.0` | `0.0` | hyperparameter | no |
| `d_guard_min_steps` | `200` | `200` | hyperparameter | no |
| `latent_damping_max_rate` | `0.0` | `0.0` | hyperparameter | no |
| `direct_particle_betas[0]` | `0.0` | `0.0` | hyperparameter | yes |
| `direct_particle_betas[1]` | `0.9` | `0.9` | hyperparameter | yes |
| `input_noise_std` | `0.0` | `0.0` | hyperparameter | no |
| `input_noise_anneal_end` | `0.1` | `0.1` | hyperparameter | no |
| `output_noise_std` | `0.0` | `0.0` | hyperparameter | no |
| `output_noise_warmup` | `0.0` | `0.0` | hyperparameter | no |
| `encoder_mode` | `"none"` | `"none"` | technique | no |
| `routing_temperature` | `0.25` | `0.25` | hyperparameter | no |
| `distance_reduction` | `"sum"` | `"sum"` | technique | no |
| `observation_sigma` | `0.03` | `0.03` | hyperparameter | no |
| `reconstruction_weight` | `1.0` | `1.0` | hyperparameter | no |
| `amsgrad` | `false` | `false` | hyperparameter | yes |
| `critic_r1_real` | `true` | `true` | technique | no |
| `critic_payoff_damping` | `true` | `true` | technique | no |
| `output_noise_mode` | `"fixed"` | `"fixed"` | technique | no |
| `lr_control` | `"mobility"` | `"mobility"` | technique | no |
| `particle_birth_death` | `false` | `false` | technique | no |
| `row_evidence_gate` | `false` | `false` | technique | no |
| `table_release_rule` | `"any"` | `"any"` | technique | no |
| `row_evidence_hot` | `true` | `true` | technique | no |
| `row_evidence_exclude` | `true` | `true` | technique | no |
| `row_evidence_hold` | `true` | `true` | technique | no |
| `birth_death_space` | `"data"` | `"data"` | technique | no |
| `serve_average` | `0.0` | `0.0` | hyperparameter | no |
| `reopen_signal` | `"data"` | `"data"` | technique | no |
| `reopen_anchor` | `"hold"` | `"hold"` | technique | no |
| `reopen_guard` | `null` | `null` | technique | no |
| `row_evidence_null` | `"theory"` | `"theory"` | technique | no |
| `birth_death_isolation` | `false` | `false` | technique | no |
| `birth_death_feature_scale` | `"none"` | `"none"` | technique | no |
| `birth_death_backend` | `"knn"` | `"knn"` | technique | no |
| `birth_death_cells` | `64` | `64` | hyperparameter | no |
| `birth_death_metric_rank` | `8` | `8` | hyperparameter | no |
| `birth_death_chunk` | `256` | `256` | hyperparameter | no |
| `birth_death_parent_policy` | `"real_anchor"` | `"real_anchor"` | technique | no |
| `row_policy` | `"independent"` | `"independent"` | technique | no |
| `optimizer_family` | `"dualnorm"` | `"adam"` | technique | no |
| `optimizer_momentum` | `0.0` | `0.0` | hyperparameter | yes |
| `optimizer_smoothing` | `0.0` | `0.0` | hyperparameter | yes |
| `optimizer_convolution` | `"none"` | `"none"` | technique | no |
| `optimizer_adam_lr` | `null` | `null` | hyperparameter | yes |
| `eps` | `1e-08` | `1e-08` | hyperparameter | yes |
| `beta2_end` | `null` | `null` | hyperparameter | yes |
| `beta2_anneal_end` | `0.2` | `0.2` | hyperparameter | yes |
| `reg_coeff_end` | `null` | `null` | hyperparameter | yes |
| `reg_coeff_anneal_end` | `0.2` | `0.2` | hyperparameter | yes |
| `loss` | `"relativistic"` | `"relativistic"` | technique | no |
| `d_betas[0]` | `null` | `null` | hyperparameter | yes |
| `d_betas[1]` | `null` | `null` | hyperparameter | yes |
| `d_eps` | `null` | `null` | hyperparameter | yes |
| `prior_eps` | `null` | `null` | hyperparameter | yes |
| `loss_labels[0]` | `0.0` | `0.0` | technique | no |
| `loss_labels[1]` | `1.0` | `1.0` | technique | no |
| `loss_labels[2]` | `1.0` | `1.0` | technique | no |
| `adam_variant` | `"pytorch"` | `"pytorch"` | technique | no |
| `lr_schedule` | `"cosine"` | `"cosine"` | technique | no |
| `lr_decay_rate` | `0.96` | `0.96` | hyperparameter | yes |
| `lr_decay_steps` | `50000` | `50000` | hyperparameter | yes |
| `lr_decay_staircase` | `false` | `false` | technique | no |
