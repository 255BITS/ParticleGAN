# Optimizer kernel audit

The cubic polar solver failed its numerical contract on actual large-critic
gradients. A fixed five-step filter and Frobenius normalization are separate
optimizer hypotheses, with quality still to be measured. These software
microbenchmarks grant no training-quality or default-selection credit.

[The compact receipt](kernel-audit.json) binds four complete saved gradient
batches to their exact artifact hashes, prototype sources, final measurements
and diagnostics. Bulk tensors and logs remain local. Its inputs come from the
[source-bound guard capture](guard-capture.json), protocol seed 0, the public
initializer and DCGAN mock-data task at 256×256.

Four failed batches produced bitwise identical saved, eager, cold-cache,
warm-cache and mixed-shape outputs. CUDA graph corruption was ruled out.
Evaluating the same guards in float64 did not resolve their failures. Extending
cubic iteration to 64 eventually restored orthogonality while worsening the
original-input relation: three early batches reached maximum relative residuals
0.11, 0.47 and 6.75. The unchanged limit is 0.001. Raising the iteration budget or
weakening the guard would not repair that computation.

The missing condition was visible in the second 512-channel critic layer,
which the original first-matrix-per-shape sample did not cover. One actual
update-4 matrix had largest singular value 96,029 and absolute smoothing 0.001;
the native float32 rank cutoff retained only 270 of 512 singular directions.
An unmasked polar root therefore has a materially different target.

The following are synchronized algebra-function medians for the saved
update-4 second-layer batch of sixteen 512×512 matrices. They exclude data
loading, reference evaluation and the public optimizer's validation/dispatch.
Contention was unknown and the historical prototype logs did not record a
physical GPU UUID; these are unpaired observations on an RTX A6000.

| Algebra prototype | Median ms | Relative direction difference from native |
| --- | ---: | ---: |
| Native CUDA SVD, Jacobi | 749.51 | 0 |
| CUDA SVD, QR driver | 417.82 | 0.00019 |
| Float64 Gram eigensolve, original float32 rank cutoff | 89.45 | 0.00017 |
| Five-step augmented quintic filter, divided by 1.21 | 7.87 | 0.74353 |
| Smoothed Frobenius direction | 0.32 | 0.99157 |

The approximate SVD driver failed to converge on this low-rank input. The
masked float64 eigensolve was close to native but remained expensive. The two
fast filters are different optimizers; numerical distance alone does not
predict their training quality.

The quintic uses coefficients `(3.4445, -4.775, 2.0315)` associated with
[Muon](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html) and its
[reference implementation](https://github.com/KellerJordan/Muon/blob/master/muon.py).
Our adaptation applies five float32 steps to `[M; λI]`, normalizes by the
spectral upper bound `sqrt(||M||F² + λ²)`, and divides the top block by 1.21.
It preserves absolute smoothing but makes no exact-polar or convergence claim.
The scalar polynomial maps `[0, 1.2023686052]` into itself, supporting a unit
operator bound after rescaling in exact arithmetic. Independent SVD scalar
oracles and floating-point bound checks live in
[the primitive tests](../../../tests/test_convolution_quintic.py).

The tighter initial norm bound reduced actual generator examples from twelve
to eight cubic steps, including 17.68→11.58 ms on a repeated 512×512 batch. It
did not resolve the ill-conditioned critic failures.

To replay these algebra prototypes without launching training, restore the
byte-exact input artifacts and run:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=1 \
python reports/forge/bispf-optimized/reproduce_kernels.py \
  --matrix-root /home/martyn/dev/ParticleGAN/runs/forge/bispf-optimized/guard-replay-v1 \
  --device cuda:0 --repeats 3 \
  --output runs/forge/bispf-optimized/kernel-replay.json
```

Use `--device cpu` for a separate CPU cohort. Missing tensor artifacts block
this replay; fresh training would create new evidence rather than recover the
recorded inputs. Final recipe latency and selection belong to the Forge
performance receipts and current technique inventory.
