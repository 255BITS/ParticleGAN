# CIFAR-10 particle DDGAN comparison

| Run | Updates | Final samples | Final FID | Training min | Total min | G params | D trainable | D total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| g64 | 1,000 | 5,000 | 78.684 | 2.16 | 2.69 | 4,129,475 | 1,000,256 | 3,783,040 |
| r34 | 1,000 | 5,000 | 79.293 | 2.35 | 2.72 | 1,044,835 | 1,000,256 | 9,170,560 |

Seeds: 24002. See each full config for the controlled differences.
Progress and final FIDs with different sample counts have different sample-count bias.
Labels are used in training. Global FID does not verify class fidelity.
See ../../README.md for architecture, full metric protocol and reproduction commands.
