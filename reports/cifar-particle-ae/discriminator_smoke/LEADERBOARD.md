# Discriminator diagnosis

3/3 runs certified. Higher AUC means better real/fake ranking; it is not FID.

| Run | D-only updates | Test AUC | Train AUC | Fake image gradient | G adversarial gradient | Pixel/features cosine | Total / sum gradient norms | Train seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 0 | 0.5576 | 0.4893 | 0.3541 | 0.3893 | 0.0043 | 0.7568 | 0.0 |
| current | 8 | 0.6938 | 0.5818 | 0.5356 | 2.0101 | 0.0261 | 0.7470 | 1.6 |
| weaker | 0 | 0.5576 | 0.4893 | 0.3541 | 0.3893 | 0.0043 | 0.7568 | 0.0 |
| weaker | 8 | 0.6938 | 0.5818 | 0.5359 | 2.0074 | 0.0261 | 0.7464 | 1.6 |
| every_step | 0 | 0.5576 | 0.4893 | 0.3541 | 0.3891 | 0.0043 | 0.7568 | 0.0 |
| every_step | 8 | 0.6946 | 0.5833 | 0.5382 | 2.0051 | 0.0265 | 0.7472 | 1.9 |

Frozen G/E/prior/features, original checkpoints and D optimizer step counts verified. Fixed diagnostic draws are independent of training; CIFAR test split is never trained on. All D-only arms start with identical D state and the same parent training RNG. Production CUDA allows small floating-point differences. No seed replicates.

AUC/gradient changes require a joint FID continuation to establish value. D-only warmup does not change the frozen generator FID. Detailed branch scores, cap activation and all evaluation points are in results.json.
