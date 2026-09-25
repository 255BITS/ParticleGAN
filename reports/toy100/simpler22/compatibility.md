# One-recipe 22-toy gate: PASS

A PASS requires the same global optimizer, loss, schedule, and noise settings on all 22 hosts. The three 100-mode problems must pass both coverage and accuracy gates; the 19 canonical cases must sustain their frozen live gates. Architecture and host resource sizes follow the frozen task declarations.

| Evidence | Live passes | Status |
| --- | ---: | --- |
| Three 100-mode problems, coverage + accuracy | 3/3 | PASS |
| Same candidate on 19 canonical hosts | 19/19 | PASS |
| Candidate six-vector full-noise screen | 6/6 | PASS |
| Public v3 installed-wheel control | 0/19 | MISSING |

Global fields identical: **True**. Noise applied on all 19: **True**. 

Native policy archive scope: **native-policy-public-package-v2**. Full public package source coverage: **True**.

## Per-case result

| Group | Case | Live | Detail |
| --- | --- | --- | --- |
| 100-mode | `grid100` | PASS | coverage PASS; accuracy PASS |
| 100-mode | `rotated100` | PASS | coverage PASS; accuracy PASS |
| 100-mode | `staggered100` | PASS | coverage PASS; accuracy PASS |
| canonical | `two_pole` | PASS | noise applied True; eval learned_particles_and_critic_gradient; final suffix 13 |
| canonical | `trajectory` | PASS | noise applied True; eval generated_samples; final suffix 15 |
| canonical | `residual_student` | PASS | noise applied True; eval generated_samples; final suffix 20 |
| canonical | `unipolar` | PASS | noise applied True; eval learned_residual_parameters; final suffix 18 |
| canonical | `ae_gan_hold` | PASS | noise applied True; eval generated_and_reconstructed_samples; final suffix 22 |
| canonical | `cover_leftover` | PASS | noise applied True; eval learned_residual_parameters; final suffix 14 |
| canonical | `unused_token_hold` | PASS | noise applied True; eval learned_embedding_parameters; final suffix 15 |
| canonical | `mid_scale_identity` | PASS | noise applied True; eval learned_residual_parameters; final suffix 17 |
| canonical | `mode_hold` | PASS | noise applied True; eval generated_samples; final suffix 5 |
| canonical | `vector_two_broad` | PASS | noise applied True; eval generated samples; final suffix 23 |
| canonical | `vector_unequal_mass` | PASS | noise applied True; eval generated samples; final suffix 6 |
| canonical | `vector_unequal_width` | PASS | noise applied True; eval generated samples; final suffix 8 |
| canonical | `vector_anisotropic` | PASS | noise applied True; eval generated samples; final suffix 19 |
| canonical | `vector_overlap` | PASS | noise applied True; eval generated samples; final suffix 8 |
| canonical | `vector_spiral` | PASS | noise applied True; eval generated samples; final suffix 23 |
| canonical | `img_stripes2` | PASS | noise applied True; eval generated samples; final suffix 8 |
| canonical | `img_bars4` | PASS | noise applied True; eval generated samples; final suffix 7 |
| canonical | `img_blobs4` | PASS | noise applied True; eval generated samples; final suffix 19 |
| canonical | `img_intensity2` | PASS | noise applied True; eval generated samples; final suffix 5 |

The public v3 control uses its own recipe and cannot supply missing candidate passes. A single-case or six-case screen is incomplete for 22/22. EMA is recorded separately and never determines the live gate.
