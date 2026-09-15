# CIFAR-10 particle DDGAN comparison

| Run | Updates | Final samples | Final FID | Training min | Total min | G params | D trainable | D total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrained | 10,000 | 50,000 | 30.263 | 19.98 | 22.62 | 157,339,107 | 1,000,256 | 3,783,040 |
| random | 10,000 | 50,000 | 32.550 | 20.82 | 23.54 | 157,339,107 | 1,000,256 | 3,783,040 |

Seeds: 24002. See each full config for the controlled differences.
Progress and final FIDs with different sample counts have different sample-count bias.
Labels are used in training. Global FID does not verify class fidelity.
See ../README.md for architecture, full metric protocol and reproduction commands.
