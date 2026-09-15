# CIFAR-10 particle DDGAN comparison

| Run | Updates | Final samples | Final FID | Training min | Total min | G params | D trainable | D total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r34 | 10,000 | 50,000 | 32.506 | 11.11 | 12.68 | 1,044,835 | 1,000,256 | 9,170,560 |
| g64 | 10,000 | 50,000 | 33.332 | 10.93 | 12.63 | 4,129,475 | 1,000,256 | 3,783,040 |

Seeds: 24002. See each full config for the controlled differences.
Progress and final FIDs with different sample counts have different sample-count bias.
Labels are used in training. Global FID does not verify class fidelity.
See ../../README.md for architecture, full metric protocol and reproduction commands.
