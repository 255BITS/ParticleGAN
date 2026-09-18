# Discriminator diagnosis

4/4 runs certified. Higher AUC means better real/fake ranking; it is not FID.

| Run | D-only updates | Test AUC | Train AUC | Fake image gradient | G adversarial gradient | Pixel/features cosine | Total / sum gradient norms | Train seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control_8 | 0 | 0.4707 | 0.4796 | 0.3631 | 0.4244 | 0.0035 | 0.7467 | 0.0 |
| warmstart_8 | 0 | 0.9389 | 0.9322 | 1.1843 | 4.7574 | 0.0124 | 0.9092 | 0.0 |
| control_20k | 0 | 0.4134 | 0.4200 | 0.4556 | 0.3315 | 0.0013 | 0.8016 | 0.0 |
| weaker_20k | 0 | 0.5575 | 0.5700 | 0.6394 | 0.2810 | 0.0001 | 0.8103 | 0.0 |

Frozen G/E/prior/features, original checkpoints and D optimizer step counts verified. Fixed diagnostic draws are independent of training; CIFAR test split is never trained on. Within the matched D-only experiment, all arms start with identical D state and the same parent training RNG. Production CUDA allows small floating-point differences. No seed replicates.

AUC/gradient changes require a joint FID continuation to establish value. D-only warmup does not change the frozen generator FID. Detailed branch scores, cap activation and all evaluation points are in results.json.
