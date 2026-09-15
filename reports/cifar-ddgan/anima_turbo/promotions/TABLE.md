# CIFAR-10 particle DDGAN comparison

| Run | Updates | Final samples | Final FID | Training min | Total min | G params | D trainable | D total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 10,000 | 50,000 | 31.306 | 20.29 | 22.98 | 157,339,107 | 1,000,256 | 3,783,040 |
| turbo | 10,000 | 50,000 | 37.330 | 20.88 | 23.60 | 157,339,107 | 1,000,256 | 3,783,040 |

Seeds: 24002. See each full config for the controlled differences.
Progress and final FIDs with different sample counts have different sample-count bias.
Labels are used in training. Global FID does not verify class fidelity.
See ../README.md for architecture, full metric protocol and reproduction commands.
