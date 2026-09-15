# CIFAR-10 particle DDGAN comparison

| Run | Updates | Final samples | Final FID | Training min | Total min | G params | D trainable | D total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrained_d | 10,000 | 50,000 | 31.741 | 20.05 | 21.50 | 1,044,835 | 1,000,256 | 3,783,040 |
| combined | 10,000 | 50,000 | 38.719 | 25.49 | 27.31 | 2,113,932 | 1,000,256 | 3,783,040 |
| flat_particle_g | 10,000 | 50,000 | 52.156 | 16.56 | 18.42 | 2,113,932 | 708,552 | 708,552 |
| joint_control | 10,000 | 50,000 | 54.931 | 10.25 | 11.73 | 1,044,835 | 708,552 | 708,552 |

Seeds: 24002. See each full config for the controlled differences.
Progress and final FIDs with different sample counts have different sample-count bias.
Labels are used in training. Global FID does not verify class fidelity.
See ../README.md for architecture, full metric protocol and reproduction commands.
