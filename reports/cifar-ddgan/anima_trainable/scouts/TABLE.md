# CIFAR-10 particle DDGAN comparison

| Run | Updates | Final samples | Final FID | Training min | Total min | G params | D trainable | D total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 1,000 | 5,000 | 74.709 | 2.57 | 3.13 | 157,339,107 | 1,000,256 | 3,783,040 |
| pretrained | 1,000 | 5,000 | 82.771 | 2.52 | 2.99 | 157,339,107 | 1,000,256 | 3,783,040 |

Seeds: 24002. See each full config for the controlled differences.
Progress and final FIDs with different sample counts have different sample-count bias.
Labels are used in training. Global FID does not verify class fidelity.
See ../README.md for architecture, full metric protocol and reproduction commands.
