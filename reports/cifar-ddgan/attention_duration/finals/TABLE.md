# CIFAR-10 particle DDGAN comparison

| Run | Updates | Final samples | Final FID | Training min | Total min | G params | D trainable | D total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 50,000 | 50,000 | 25.397 | 45.73 | 48.27 | 1,044,835 | 1,000,256 | 3,783,040 |
| attention | 50,000 | 50,000 | 26.334 | 53.09 | 55.79 | 1,148,963 | 1,000,256 | 3,783,040 |

Seeds: 24002. See each full config for the controlled differences.
Progress and final FIDs with different sample counts have different sample-count bias.
Labels are used in training. Global FID does not verify class fidelity.
See ../../README.md for architecture, full metric protocol and reproduction commands.
