# Named Atlas GPU diagnostic results

One fixed LR .0053125 / prior-rate 1.5 pair across five explicitly named host laws, seed 0, GPU-only numerical evidence. Five separate full views; no pooled ranking or qualification.

This immutable cut contains 3/5 terminal family ledgers, 6/8 attempted adaptations and all **130 required cells** (five separate 26-slot views, each 5/19/2).

Current paid **235.707335s**, conservative reserve **0.000000s**; prior engineering **25.322955s** is debited once. Inclusive charged **261.030290/10500s**. These are supervised costs, not convergence timing or FLOPs.

| Family / cohort | GPU | Original numerical PASS | FAIL | INVALID | INCOMPLETE | BLOCKED | NOT_RUN | Family status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| atlas_conditional / `conditional_policy_selected_cloud_v1` | 0 | 4 | 0 | 0 | 0 | 0 | 22 | COMPLETE_DIAGNOSTIC |
| atlas_ae_routed / `ae_routed_policy_v1` | 0 | 1 | 0 | 0 | 0 | 0 | 25 | COMPLETE_DIAGNOSTIC |
| atlas_routed / `routed_policy_selected_cloud_v1` | 1 | 0 | 0 | 1 | 0 | 0 | 25 | INVALID |
| atlas_multibank / `multibank_policy_v1` | 1 | 0 | 0 | 0 | 0 | 0 | 26 | NOT_RUN |
| atlas_word_joint_min11 / `word_joint_policy_min11_v1` | 1 | 0 | 0 | 0 | 0 | 0 | 26 | NOT_RUN |

Numerical PASS/FAIL applies only to the original named variant's terminal-five gates. INVALID and INCOMPLETE have no numerical gate. Unexecuted questions remain NOT_RUN; blocked original N5 has no min11 resource-law credit.

| Adapted question | Original grade / execution status | Original thresholds | Actual goal GIF |
|---|---|---|---|
| Recover each original identity's conditioned eight-point trajectory. (`trajectory_conditional_policy_selected_cloud_v1`) | **PASS** | identity_mse <= 0.02 | [recorded 400-update goal GIF](gifs/trajectory_conditional_policy_selected_cloud_v1.gif) |
| Preserve each paired trajectory, successful action and unused padding. (`residual_student_conditional_policy_selected_cloud_v1`) | **PASS** | identity_mse <= 0.02; success_rate >= 1.0; wrong_pad_rate <= 0.0 | [recorded 400-update goal GIF](gifs/residual_student_conditional_policy_selected_cloud_v1.gif) |
| Apply the positive concept edit while preserving neutral identity and suppressing off-caption leakage. (`unipolar_conditional_policy_selected_cloud_v1`) | **PASS** | cover >= 0.85; off_caption <= 0.05; neu_hold >= 0.85 | [recorded 400-update goal GIF](gifs/unipolar_conditional_policy_selected_cloud_v1.gif) |
| Recover both signed concept directions and magnitudes while preserving identity at zero and intermediate scale. (`mid_scale_identity_conditional_policy_selected_cloud_v1`) | **PASS** | concept_cos_plus >= 0.85; concept_cos_minus >= 0.85; concept_mag_plus >= 0.75; concept_mag_plus <= 1.25; concept_mag_minus >= 0.75; concept_mag_minus <= 1.25; identity_at_0 >= 0.85; identity_at_mid >= 0.85 | [recorded 800-update goal GIF](gifs/mid_scale_identity_conditional_policy_selected_cloud_v1.gif) |
| Reconstruct each original hard-AE input and keep generated points near both fixed anchors under the routed MoG law. (`ae_gan_hold_ae_routed_policy_v1`) | **PASS** | recon_mse <= 0.05; hold <= 0.35 | [recorded 250-update goal GIF](gifs/ae_gan_hold_ae_routed_policy_v1.gif) |
| Separate the original unused-token nuisance while preserving concept geometry under complete routed ownership. (`unused_token_hold_routed_policy_selected_cloud_v1`) | **INVALID** | unused_hold >= 0.85; concept_move >= 0.85 | Unavailable; no invented media or gate |
| Keep content and identity while separating the original odd/even residual poles and controlling leakage. (`cover_leftover_multibank_policy_v1`) | **NOT_RUN** | u_kept >= 0.85; content_kept >= 0.75; leak_ratio <= 0.2; pole_rel_err_plus <= 0.2; pole_rel_err_minus <= 0.2; same_dir <= 0.25 | Unavailable; no invented media or gate |
| Acquire all five canonical words and recover their paired reconstructions using eleven rows, free E and the declared joint-code law. (`five_word_joint_acquisition_word_joint_policy_min11_v1`) | **NOT_RUN** | sample_count >= 1024; quality_fraction >= 0.95; modes == 5; mass_tv <= 0.1; reconstruction_exact == 1; minimum_reconstruction_token_probability >= 0.9 | Unavailable; no invented media or gate |

`unused_token_hold_routed_policy_selected_cloud_v1` retains **INVALID**, not a numerical FAIL: RuntimeError — Expected all tensors to be on the same device, but got min is on cuda:0, different from other tensors on cpu (when checking argument in method wrapper_CUDA_clamp_min__Tensor). Completed updates are UNAVAILABLE. Recorded CUDA peak allocation 67318784 bytes; peak reserve 69206016 bytes. The generic public_trainer exception fallback is not an executed owner receipt; intended law is public_components. No gate or GIF is invented.

For AE, `recon_mse` is mean squared input/reconstruction error; `hold` is the mean distance from each fixed anchor to its nearest generated point. The two gates test reconstruction and anchor retention; they do not require the generated population to match full Gaussian density. Conditional questions use the original known finite contexts. The trajectory question tests paired correspondence; residual-student adds action-success and unused-padding controls; unipolar tests a positive edit with neutral/off-caption protection; mid-scale tests both polarities and intermediate identity.

The AE variant uses the original fixed .025-width MoG with explicit routed AE ownership. The word variant has eleven actual rows, five target words, a free encoder and a distinct same-effective-code joint law; original N5 remains BLOCKED. Conditional, routed and multibank variants retain their declared source-owned target/probe and sampling laws. Selected states, full checkpoints and optimizer owners are hash-bound, without replay by this publisher.

The structural-readiness card is metadata/API readiness only; it proves neither representability nor learned quality. There is no ordinary tier, calibration, shipping-default, speed-ranking or cross-family/cohort qualification credit.

Scientific source `ff94453b45e02fd451b21c26d5991f6ed435c292` / `87fcd4e28bdcd9f347379b7db1fbcf0af3edcc0fbd5e97e4a98d9a49f8987a87`; protocol `f2cf72408406c5b8ab280b586fe28010ad9bc21cf45a4f4ea94a0d7a82d65790`. Exact applied Recipes, runtime, independent grades, source/task/evaluator hashes and all 130 statuses are in [results.json](results.json); input identities are in [input-index.json](input-index.json). No raw logs, tensors, state dumps or private attempt tokens are copied.

Reproduce this display-only export with `python publish_results.py --card final-inputs.json --trusted-sha256 b4becc8dd901d4732efad7852349efbe18632263e8e30c41798d965d3ab5e157 --output /NEW/external/publication`, using the hash-pinned original local archives. This launches no model or scoring. The JSON/GIF/grade/media bytes remain those of the frozen export; this README adds only goal explanations. [Integrity and privacy proof](publication-proof.json).
