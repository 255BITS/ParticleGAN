# Completed scout interpretation

The 298,595-parameter plain deconvolution generator without normalization completed 40k updates with 16,384 independently initialized particles. FID50k improved at every sampled point: 77.23, 46.65, 37.62, 33.33, 30.70, 28.81, 27.01, 26.22 at 5k intervals. Final is the best sampled checkpoint. Training took23.94 minutes; total33.20 minutes including evaluation. Current-source/config certificate verified after completion. Both GPUs are now idle; no continuation queued.

| Reference | Step | FID50k |
|---|---:|---:|
| Residual CNN, 16k, historical best | 80000 | 15.7527 |
| Residual CNN, 32k, best sampled | 50000 | 16.1271 |
| Residual CNN, 32k endpoint | 80000 | 16.2461 |
| Residual CNN, 16k historical comparison | 40000 | 17.0982 |
| Residual CNN, 16k long endpoint | 160000 | 19.0584 |
| Plain deconvolution, 16k, best/final | 40000 | 26.2232 |

These are contextual references, not matched architecture ablations: the CNNs expanded a trained 1,024-particle checkpoint at10k, while deconv started with16,384 independent particles atstep0. Parameter count, normalization, residual/conditioning structure, upsampling and initialization history differ. The experiment cannot isolate any one cause or demonstrate that particle learning prevented collapse.

Visual inspection of the deconv40k and CNN40k/160k sample grids: deconv retains visibly varied subjects, colors and layouts, including recognizable vehicles and animals, but has softer and less coherent object structure. There is no obvious wholesale collapse in this100-image panel. This is a limited visual observation; feature variance ratio1.0605 is not a coverage measurement, and partial mode loss remains possible. Deconv density/coverage has not yet been measured.

Deconv reconstruction MSE0.12820 is lower than CNN16k40k0.14578 despite much worse FID. Pixel reconstruction therefore does not rank sampling quality here. In this recipe reconstruction updates only E, so reconstruction L2 cannot directly explain G's poorer visual detail. E uses1,488 distinct particles on10k test images, with entropy-effective usage about500; this describes encoder routing and does not measure generation mode count. Generation samples all16,384 centers uniformly. About50.5% of offset coordinates are saturated by the existing diagnostic criterion.

The last5k steps improved FID by0.7864; improvement from30k to40k was2.5856. This is evidence of continued learning, not an established capacity ceiling. Recommendation: extend this exact40k checkpoint to80k before interpreting the endpoint as an architectural plateau, and obtain an independent density/coverage measurement to distinguish poor fidelity from missing coverage. A matched plain deconv with normalization is a useful subsequent test; do not attribute this gap to normalization without that intervention. No jobs have been launched from this recommendation.
