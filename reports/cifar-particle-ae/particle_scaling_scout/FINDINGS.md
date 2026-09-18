# FID improves with count in the scout, with diminishing marginal gains

Both8192/16384 runs completed and certified. Same original10k checkpoint, initialFID50k19.4482 preserved, unchanged architectures/rates/sigma/EMA/Adam, oneDupdate and E-only reconstruction. SevenCUDApreflighttests and both8-update smokes passed. Fullstate/frozenfeature/parenthash and originaldata/noiseRNG checks passed. Historical1024/4096 benchmarks are reused, not rerun.

| Particles | FID15k | FID20k | Training minutes |
|---|---:|---:|---:|
|1024|19.8699|19.8932|7.25|
|4096|18.9357|18.5207|7.37|
|8192|18.6479|18.1602|7.28|
|16384|18.3401|18.0136|7.46|

The count ranking agrees at both evaluations. At20k, gains for successive count increases are1.3724,0.3605 and0.1466FID; the8192-versus16384 difference is small. Runtime differences are small as well and include ordinary GPU/runtime variation. One trajectory per count does not estimate stochastic uncertainty.

Read-only feature diagnostics add context: sibling bits rise1.1166→1.7195→2.2756 across4096/8192/16384, while density stays around0.63 and coverage is60.28%/60.29%/59.43%. Greater distinguishability does not produce proportionally greater real-feature coverage. See ../particle_information/FINDINGS.md.

Recommend extending both larger-count20k checkpoints to40k, comparing with the existing4096 duration trajectory. Reuse the4096 benchmark rather than retraining it. More centers receive fewer direct selections, so longer adaptation is a useful next test. No automatic next-count escalation or200k promotion. Both GPUs are idle; no further training or probes are queued.
