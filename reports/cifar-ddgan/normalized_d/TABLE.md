# CIFAR-10 first U-Net DDGAN baseline

| Run | Final FID (50k) | Training min | Total min | G params | D params |
|---|---:|---:|---:|---:|---:|
| image_lr | 76.863 | 11.26 | 13.95 | 1,044,835 | 647,722 |
| toy_lr | 62.819 | 10.40 | 12.92 | 1,044,835 | 647,722 |

Single seed (24002), 10k updates each. See each full config for the controlled differences.
The 5k progress FIDs and 50k final FIDs have different sample-count bias.
Labels are used in training. Global FID does not verify class fidelity.
See ../README.md for architecture, full metric protocol and reproduction commands.
