# Discriminator diagnosis

4/4 runs certified. Higher AUC means better real/fake ranking; it is not FID.

| Run | D-only updates | Test AUC | Train AUC | Fake image gradient | G adversarial gradient | Pixel/features cosine | Total / sum gradient norms | Train seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_50k | 0 | 0.4588 | 0.4531 | 0.0534 | 0.0575 | -0.0062 | 0.7821 | 0.0 |
| current | 0 | 0.5255 | 0.5235 | 0.3531 | 0.3735 | 0.0035 | 0.7546 | 0.0 |
| current | 2048 | 0.9194 | 0.9202 | 0.8891 | 11.1698 | 0.0801 | 0.7495 | 48.6 |
| weaker | 0 | 0.5255 | 0.5235 | 0.3530 | 0.3735 | 0.0035 | 0.7546 | 0.0 |
| weaker | 2048 | 0.9552 | 0.9552 | 1.2606 | 13.9328 | 0.0725 | 0.7455 | 47.7 |
| every_step | 0 | 0.5255 | 0.5235 | 0.3530 | 0.3734 | 0.0035 | 0.7546 | 0.0 |
| every_step | 2048 | 0.9471 | 0.9446 | 1.0744 | 13.1588 | 0.0606 | 0.7641 | 159.5 |

Frozen G/E/prior/features, original checkpoints and D optimizer step counts verified. Fixed diagnostic draws are independent of training; CIFAR test split is never trained on. All D-only arms start with identical D state and the same parent training RNG. Production CUDA allows small floating-point differences. No seed replicates.

AUC/gradient changes require a joint FID continuation to establish value. D-only warmup does not change the frozen generator FID. Detailed branch scores, cap activation and all evaluation points are in results.json.
