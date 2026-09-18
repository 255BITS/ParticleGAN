# Discriminator diagnosis

From CNN E-only 10k (FID50k 19.4482), preserve G/E/prior/EMA and train only D for 2048 updates. Same full parent D/Adam, restored training RNG, same real/fake draws across arms. Current bcap coefficient 1 every8, weaker coefficient0.1 every8, coefficient1 every step. No seed experiments.

Probe 2048 fresh fake draws and unaugmented CIFAR test images at 0/512/1024/1536/2048; independently sample train split to detect generalization gaps. Fixed evaluation streams independent of training. Read-only 50k parent probe supplies plateau comparison. Scores and image gradients attributed to pixel and three frozen-feature heads; report branch cancellation and adversarial G/prior gradient norms. AUC measures full pooled ranking, not a zero-logit classification threshold. Cap activation measured on total image gradients; branch statistics are diagnostic only.

Successful preflight requires original weighted branch sum agreement, model/RNG nonmutation during probes, frozen G/E/prior/features unchanged, immutable parent/source hashes, correct D Adam step increment, and finite losses. Use explicit D-only delta artifacts, no fabricated joint-training step counts.

Choose a joint continuation only after inspecting held-out diagnostics; use unchanged same-parent control and FID50k. No automatic long promotion. Diagnostic AUC alone cannot establish useful generator feedback. Historical pretrained feature replacement retained mismatched heads, so it does not isolate feature quality.

Tail: `tail -F runs/cifar_particle_ae/discriminator_diagnosis/PIPELINE.log`
