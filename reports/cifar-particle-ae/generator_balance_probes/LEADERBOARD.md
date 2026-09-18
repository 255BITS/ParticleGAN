# Discriminator diagnosis

2/2 runs certified. Higher AUC means better real/fake ranking; it is not FID.

| Run | D-only updates | Test AUC | Train AUC | Fake image gradient | G adversarial gradient | Pixel/features cosine | Total / sum gradient norms | Train seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 0 | 0.4743 | 0.4731 | 0.1558 | 0.1435 | -0.0007 | 0.7509 | 0.0 |
| half_g | 0 | 0.5575 | 0.5543 | 0.4921 | 0.5724 | 0.0055 | 0.7194 | 0.0 |

Frozen G/E/prior/features, original checkpoints and D optimizer step counts verified. Fixed diagnostic draws are independent of training; CIFAR test split is never trained on. Within the matched D-only experiment, all arms start with identical D state and the same parent training RNG. Production CUDA allows small floating-point differences. No seed replicates.

AUC/gradient changes require a joint FID continuation to establish value. D-only warmup does not change the frozen generator FID. Detailed branch scores, cap activation and all evaluation points are in results.json.
