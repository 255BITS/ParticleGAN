# CIFAR-10 U-Net DDGAN comparison

| Run | Updates | Final samples | Final FID | Training min | Total min | G params | D params |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant | 30,000 | 50,000 | 43.678 | 33.95 | 39.08 | 1,044,835 | 647,722 |
| cosine | 30,000 | 50,000 | 49.390 | 31.29 | 36.18 | 1,044,835 | 647,722 |

Seeds: 24002. See each full config for the controlled differences.
Progress and final FIDs with different sample counts have different sample-count bias.
Labels are used in training. Global FID does not verify class fidelity.
See ../README.md for architecture, full metric protocol and reproduction commands.
