# CIFAR-10 first U-Net DDGAN baseline

| Run | Final FID (50k) | Training min | Total min | G params | D params |
|---|---:|---:|---:|---:|---:|
| width32 | 191.516 | 8.23 | 10.62 | 1,044,835 | 646,634 |
| width64 | 220.189 | 17.98 | 20.80 | 4,129,475 | 2,538,442 |

Single seed (24002), 10k updates each. See each full config for the controlled differences.
The 5k progress FIDs and 50k final FIDs have different sample-count bias.
Labels are used in training. Global FID does not verify class fidelity.
See ../README.md for architecture, full metric protocol and reproduction commands.
