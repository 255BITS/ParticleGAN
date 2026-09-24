# Bidirectional coverage: three frozen-state counterfactuals

The declared unit-mean bidirectional Chamfer correction repairs all three
captured failures of one-sided coverage on the original fixed evaluation draws.
This is an isolated filter result, **not a warm-run, cold-acquisition or sustained
hold pass**. No training was performed. A fresh warm test is the next gate.

| Captured update | Original one-sided noisy HQ | Symmetric ideal target HQ | Symmetric actual HQ | Modes | Accepted alpha |
|---:|---:|---:|---:|---:|---:|
| 1133 | .84619140625 | 1.00000000000 | 1.00000000000 | 8 | 1 |
| 1148 | .89697265625 | .975341796875 | .974609375000 | 8 | 1 |
| 1186 | .891845703125 | .973144531250 | .973144531250 | 8 | 1 |

All three improve and pass the unchanged diagnostic requirement of eight modes
and HQ at least .9. All twelve corrected clean particles are HQ at each state.
The states are independent counterfactuals from the original failed policy;
they do not establish the trajectory of a policy using this correction.

## Why this single mechanism was chosen

The independently hash-matched replay showed that one-sided Lloyd targets
themselves fail the same noisy checks as the actual latent projection.
At update 1133, the GAN proposal moves rows 8 and 9 off the ring, and their empty
assignment cells leave them there. Update 1186 likewise has an uncorrected empty
row 10. At 1148, row 4 has one assigned real sample: the GAN result is .2185 from
its nearest center, but the assigned sample is .2679 from that center. The
one-sided correction follows that tail sample and changes the whole model's
paired noisy HQ from .952881 to .896973. Thus nonlinear pullback error is not the
main cause. Centers and HQ enter this diagnosis and scoring only.

For one-sided coverage, each particle's objective gradient is weighted by its
number of assigned real samples, whereas each particle still generates 1/12 of
the model's mass. Empty and sparse cells expose this mismatch. The bounded next
mechanism adds the reverse nearest-real term, with **unit means and no tuned
coefficient**:

```
C = mean_real min_particle ||real - output||²
Q = mean_particle min_real ||output - real||²
target_j = (sum_assigned_real/B + nearest_real_j/N) / (n_j/B + 1/N)
```

This is the minimizer with both nearest-neighbor assignments held fixed. Empty
cells get a nearest-real target. It does not impose balanced transport masses.
In the sparse update-1148 case, row 4 now targets approximately
`(-3.068919, -.060751)` and lands at `(-3.068720, -.060054)`, avoiding the original
single assigned tail sample. This is an explicit sampled-data objective change,
not an equilibrium-preserving GAN optimizer or a distribution-matching theorem.

The new helper is separate from the frozen one-sided helper. It keeps the same
clean, row-independent generator and supplied post-Adam diagonal metric,
solves `P Jᵀ(J P Jᵀ)^†(target-G(z))` with `pinv_rtol=1e-6`, and checks actual
recomputed C+Q at alpha 1 through 1/256. Zero correction or exhausted acceptance
rests. There is no target-center access, extra sample, clock decay, Adam update,
or positive movement requirement. Here all three full steps decrease C+Q:
.03121447→.00929579, .01287594→.00893319, and .01329812→.00738514.

## Verification and evidence

Eight focused tests pass: exact weighted target and its frozen quadratic
stationarity; distinct B/N normalization; a forward-coverage decrease that
worsens reverse coverage; exact minimum-metric latent solution; matched and
zero-Jacobian resting; nonlinear backtracking and rejection; preservation of
network, gradient, Adam state and RNG; and exception rollback. The filter
independently checks exact reconstructed clean support, exact original noisy
HQ, unchanged generator/metric/real/RNG/gradient buffers, and zero optimizer
calls. Host integration and moment accounting remain for the training adapter.

Evaluation uses the original 4,096 prior indices from seed 9 and output noise
from seed `402+step`, sigma .029. These are the existing evaluation draws, not
new training seeds. The independently captured warm state is
`6cc79b6e0d11eafae176b68e7d9d8c26c02c886866370134cd70c864fe882e21`;
the original failed policy's final state is
`e73693ece461eb81074c1dc3c3df4c73b04e1535e0178d3a891bc3716db7a236`.

The first driver invocation encountered integer rather than string sidecar
keys before any correction. Its source and error declaration are preserved
under `invalid-input-schema`; only key normalization changed for the successful
invocation. The helper and declared mechanism remained unchanged.

All raw results, declarations, input tensors, compressed replay diagnosis and
executed sources are in
[`continuous-evidence/chamfer-state-filter`](continuous-evidence/chamfer-state-filter/manifest.json).
The helper SHA256 is
`5a3943622dbc15f9ea77befe6b2b16edeed0739b722592bcfec12ed29df1ceda`.
The filter source SHA256 is
`91af6d2be91a41a7ffbddd3ce314b3e6c4510d925e277f09639b5c9b482a3cc9`.

Reproduce from a repository checkout:

```bash
gzip -dc reports/toy100/continuous-evidence/chamfer-state-filter/diagnosis.json.gz > /tmp/chamfer-diagnosis.json
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' /tmp/pr38-default-env/bin/python reports/toy100/chamfer_state_filter.py --states reports/toy100/continuous-evidence/chamfer-state-filter/states.pt --diagnosis /tmp/chamfer-diagnosis.json --output /tmp/chamfer-counterfactual-rerun
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' /tmp/pr38-default-env/bin/python -m pytest -q tests/test_chamfer_pullback.py
```
