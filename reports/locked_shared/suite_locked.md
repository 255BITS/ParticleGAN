# Original conceptmod suite: locked_shared only

**2 FAIL, 19 PASS. Cover-posture columns excluded.**

| Toy | Original suite result | Evidence |
| --- | --- | --- |
| locked_shared_floor | **PASS** | demo LOCKED stamp |
| leaderboard_honesty | **PASS** | locked_shared won; declared negatives failed |
| shared_trajectory | **FAIL** | identity_mse=0.23389 |
| residual_student | **PASS** | identity_mse=0.00219 success=1.000 |
| orbit_hold | **PASS** | radius hold |
| unipolar | **PASS** | locked_rpgan |
| ae_gan_hold | **PASS** | locked AE-GAN hold |
| cover_leftover | **PASS** | demo cover + faithful guard |
| erase_keep_backend | **PASS** | cpu locked geometry |
| particle_posture | **PASS** | locked_tiny n=12 |
| mode_hold | **FAIL** | modes=3 HQ=0.494873046875 |
| late_collapse | **PASS** | val gate exports good checkpoint |
| keep_critic | **PASS** | frozen host + demo stamp |
| lm_target | **PASS** | trajectory |
| field_lift | **PASS** | locked_2d+locked_lift |
| unused_token_hold | **PASS** | unused hold |
| path_suffix_lora | **PASS** | locked_suffix+locked_regex |
| mid_scale_identity | **PASS** | locked mid-scale grid |
| dsl_macro_expand | **PASS** | documented triple + negatives |
| dsl_phrase_jobs | **PASS** | documented phrases + negatives |
| dsl_game_geometry | **PASS** | locked game + geometric right |

These are the original suite scorer results, not new ParticleGAN acceptance gates. The original suite mixes measured behavior with identity, selection and DSL checks; this audit reproduces that row without importing its gate implementations into ParticleGAN.

The original suite converts any `mode_hold` result other than PASS, including INCONCLUSIVE, to **FAIL**. The standalone behavioral leaderboard preserves INCONCLUSIVE.

The table supplied in the comparison had two extra cover-posture columns. The current reference removed those columns without changing the other toy implementations or scorer logic.

Reference: `5571213f5e8e129cfda45c785c3f30aad9c1d8c9`. Python 3.12.13; PyTorch 2.13.0+cu126; PEFT 0.21.0. CPU, seed 0, one thread.

The initial PEFT 0.20 run could not import `NoMatchingPeftModuleError` for `path_suffix_lora`. PEFT 0.21 resolves that dependency error; no toy code or thresholds are changed.

Reproduce: `python -m benchmarks.locked_shared.suite_reference --reference /path/to/conceptmod`. Full results: [suite_locked.json](suite_locked.json). The command exits nonzero on FAIL or ERROR.
