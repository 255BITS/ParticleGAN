# CIFAR-10 first U-Net DDGAN baseline

| Run | Final FID (50k) | Training min | Total min | G params | D params |
|---|---:|---:|---:|---:|---:|
| concat32 | 180.069 | 8.76 | 11.31 | 1,044,835 | 629,473 |
| ucd32 | 186.564 | 8.18 | 10.63 | 1,044,835 | 646,634 |

Single seed (24002), 10k updates each. See each full config for the controlled differences.
The 5k progress FIDs and 50k final FIDs have different sample-count bias.
Labels are used in training. Global FID does not verify class fidelity.
See ../README.md for architecture, full metric protocol and reproduction commands.
