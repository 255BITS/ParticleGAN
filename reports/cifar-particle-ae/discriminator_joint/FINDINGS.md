# Plateau investigation: conclusions

The existing discriminator can learn strong held-out separation when G is fixed. Stronger classification alone does not solve the joint-training plateau. A one-time D warmup made FID substantially worse, while 10× weaker bcap produced only a small, reversing difference from control. No arm merits long promotion.

## Matched FID results

All three arms restore the same CNN E-only10k checkpoint (FID50k19.4482) and add10k joint updates. One D update per G update, unchanged architecture, E-only reconstruction, fixed evaluation protocol, full original G/E/prior/EMA/optimizer/RNG state. Warmup changes only initial D/Adam through2048 preceding D-only updates; weaker changes only bcap coefficient1→0.1 while retaining lazy8.

| Arm | FID50k at15k | FID50k at20k | Final difference vs control |
|---|---:|---:|---:|
| Control | 19.7875 | 20.0119 | — |
| Weaker bcap | 20.1344 | 19.7848 | −0.2271 |
| D-only warmup | 27.4657 | 25.2598 | +5.2479 |

The weaker/control ranking reverses between evaluations. Neither improves on the common parent. Warmup recovers2.21 points in its last5k but remains substantially worse than control. Test reconstruction MSE is essentially unchanged across endpoints (~0.148). Each joint run takes7.2–7.4 training minutes plus about2.4 minutes of evaluation; total queue19.3 minutes. Warmup additionally costs48.6 D-only training seconds. Final sample grids remain varied; no claim of total mode collapse is supported by those grids.

## What the tests distinguish

- **Pretrained-feature information:** existing D learns test AUC0.5255→0.9194 with G fixed and original regularization. Both pixel and feature branches learn (AUC0.8896 and0.8762). This weakens a simple representation-capacity explanation. It does not prove all feature gradients reward the image statistics needed for better FID.
- **Regularization:** weaker bcap improves D-only AUC to0.9552 without more compute. Applying coefficient1 every step reaches0.9471 but costs3.3× the D-only training time. Only weaker bcap was tested in joint FID here; its benefit is unconvincing. Every-step regularization remains an untested joint intervention.
- **Persistent D strength:** after8 warmstart joint updates, AUC0.9389 and G gradient4.7574 remain strong. By20k, AUC falls to0.4322, input gradient0.0520 and G gradient0.0589. The warmup did not maintain its advantage. Control20k G gradient is0.3315; weaker0.2810. Classification and raw gradient magnitude do not predict a useful FID improvement.
- **Branch conflict:** no strong pixel/feature cancellation in image, G-parameter or prior-parameter gradients. G branch cosine is near0 at original10k/control20k/weaker20k, +0.137 at original50k, and −0.065 at warm20k. Contributions were checked against the actual combined GAN gradient in FP32.
- **Which component moves the samples:** eight-step traces show D improves held-out ranking and G/prior then reduces it on all observed updates. Counterfactuals isolate the immediate change to G: mean AUC afterD0.5524; applying only the actual prior update0.5519; only G0.4616; both0.4613. This is expected adversarial behavior and does not by itself prove a pathological learning-rate ratio. It makes a targeted update-balance test more informative than freezing the encoder or prior.

## Recommendation

Next test a lower **G-only** learning rate from the saved10k parent, with D/E/prior rates and all other settings fixed, alongside an unchanged control. For example G0.0003→0.00015, retaining one D update, lazy8 and original bcap coefficient1. This tests whether a smaller generator response lets D maintain more useful feedback without paying for two D updates. The existing global lr_scale knob also scales E/prior/D, so a new explicit G-only setting is required. This experiment is a recommendation, not launched.

If a controlled update-balance change also fails to lower FID, useful adversarial directions versus the generator/prior distribution family remain open questions. A classifier learning real/fake separation with a frozen target is insufficient to settle them. No additional backbone freeze or replacement is justified by this round alone.

## Validation and artifacts

All3 joint scouts and4 initial diagnostics certified; all6 preflight smokes passed. Exact unchanged old/new full-state continuation passed with deterministic CUDA in the test only. Parent/source hashes, frozen state, D Adam counts and nonmutating probes verified. Historical/shared sources unchanged. Five additional read-only checkpoint probes certified. Instrumented eight-step traces have their own source/parent hashes and no comparable FID/runtime. Production CUDA is nondeterministic; one run per intervention does not quantify run-to-run uncertainty. No seed experiments.

See LEADERBOARD.md, curves.png and CHECKPOINTS.json here; detailed D-only and gradient evidence in ../discriminator_diagnosis/, and endpoint probes in ../discriminator_joint_probes_early/ and ../discriminator_joint_probes_final/. Both GPUs are available; nothing further is queued.
