# Discriminator diagnosis

1/1 runs certified. Higher AUC means better real/fake ranking; it is not FID.

| Run | D-only updates | Test AUC | Train AUC | Fake image gradient | G adversarial gradient | Pixel/features cosine | Total / sum gradient norms | Train seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| warmstart_20k | 0 | 0.4322 | 0.4332 | 0.0520 | 0.0589 | 0.0044 | 0.9044 | 0.0 |

Frozen G/E/prior/features, original checkpoints and D optimizer step counts verified. Fixed diagnostic draws are independent of training; CIFAR test split is never trained on. Within the matched D-only experiment, all arms start with identical D state and the same parent training RNG. Production CUDA allows small floating-point differences. No seed replicates.

AUC/gradient changes require a joint FID continuation to establish value. D-only warmup does not change the frozen generator FID. Detailed branch scores, cap activation and all evaluation points are in results.json.
