DualNorm now drops numerically null singular directions before normalizing matrix updates. This prevents rounding-scale gradient differences from becoming unit-size moves, using the every-update rule supported by the archived Ring16 diagnostic: `U diag(s > max(rows, columns) * eps * s_max) Vh`.

The public optimizer uses this cutoff by default, including DualNorm D-only. Float32/float64 retain their dtype; half inputs compute in float32. Exact reduced SVD replaces the large-matrix Newton–Schulz path to apply the same cutoff at every size; large full-rank matrices may cost more. Bias and sampled-prior row rules remain unchanged.

Validation: 103 CUDA optimizer/public-checkpoint software tests passed; two native-Adam isolation checks passed after removing the former CPU workaround. The public factor matches the frozen experimental helper bit-for-bit on both original saved Ring16 update-401 matrices. Forge validation and diff checks passed.

The original two 1,600-update CUDA diagnostic receipts, protocols, frozen helper, actual-training GIFs and archive hashes retain their original source identity. These results are not regraded as qualification for the new default. The user's requested ordinary Tier 1 and gated Tier 2 rerun follows this merge and a separate project-wide autograd scheduling PR.

[Report and historical metrics](https://github.com/255BITS/ParticleGAN/blob/research/ring16-spectral-truncation/reports/forge/ring16-truncation/README.md) · [GPU integration receipt](https://github.com/255BITS/ParticleGAN/blob/research/ring16-spectral-truncation/reports/forge/ring16-truncation/default-adoption.json)
