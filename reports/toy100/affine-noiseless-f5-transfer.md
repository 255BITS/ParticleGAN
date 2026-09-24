# Noiseless affine Fourier-5 transfer replay

The exact `accuracy_affine_public_noiseless_f5.json` candidate passed **all 19 frozen older-host gates** in a fresh, fixed-seed replay. Independent regrading of the relocated archive also returned valid `PASS 19/19` with no source, config, action, receipt, or verdict mismatch. This establishes compatibility of the zero-noise transfer path under the declared affine/H1600 policy; it does not establish a 100-mode or combined 22-task pass.

- Source: clean `1c1a0865fe605c9f212d832c06596c12510f937e`; exact config SHA-256 `51f6bd91a60e523708b51a36bd2f38907a28bc138a1d0d1b366e6a6a200586b0`.
- Shared settings relevant here: β₂=.99, input/output noise std=0, native Fourier count=5, `toy100_model=affine_square_v1`, `network_lr_horizon_cap=1600`, and effective network LR floor=.05. Older host architectures and resources remained frozen.
- Command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ionice -c2 -n4 /tmp/pr38-default-env/bin/python -u -m benchmarks.transfer_suite.toy100_compatibility --config /dev/shm/particlegan-affine-noiseless-v1-1c1a086/configs/accuracy_affine_public_noiseless_f5.json --output /dev/shm/toy100-affine-noiseless-f5-full19-1c1a086 --all`.
- Retained evidence: `artifacts/toy100-accuracy/affine-noiseless/full19-f5-1c1a086/`, including the exact config, protocol, executable source archive, all 19 compressed episodes, index, summary, and independent regrade.

The zero-noise receipts have zero nonzero output/input-noise steps, no isolated-RNG selector, and the original unwrapped vector/image models. The archived `noise_applied=true` field means the declared zero-noise policy was applied correctly; it does not mean random noise was injected. Every case has 24 frozen checkpoints and a sustained passing suffix of at least five. This result is a gate pass on the older hosts, not a claim of bitwise equality to an earlier public-default run with other recipe fields.
