# Warm failure attribution for the sampled-real coverage pullback

The coverage candidate still **fails** the unchanged warm filter at updates
1133, 1148 and 1186 (197/200 checks, minimum observed HQ .8462). This
read-only replay captured the clean 12-particle support before the GAN update,
after the bounded GAN update, and after the prior-only Lloyd pullback at those
updates and their passing neighbors. It captured the exact D real minibatch
used by the pullback. Target-centered measurements and Lloyd cell calculations
were performed only after training and final-state verification; they do not
enter the candidate policy.

The replay exactly matches the original warm prefix hash
`6cc79b6e0d11eafae176b68e7d9d8c26c02c886866370134cd70c864fe882e21`
and final full-state hash
`e73693ece461eb81074c1dc3c3df4c73b04e1535e0178d3a891bc3716db7a236`.
Every non-timing observation, noise receipt, optimizer-final receipt and
candidate dynamics receipt matches the original artifact. The clean observer
also checks that its extra forward passes consume no global, data, input or
output RNG. The candidate source bytes are unchanged.

| Update | Original noisy live HQ | Clean HQ before GAN | After GAN | Ideal Lloyd targets | Actual pullback | Empty cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1132 PASS | 1.0000 | 1 | 1 | 1 | 1 | 0 |
| **1133 FAIL** | **.8462** | **1** | **.75** | **.8333** | **.8333** | **2** |
| 1134 PASS | 1.0000 | .8333 | 1 | 1 | 1 | 0 |
| 1147 PASS | .9214 | 1 | .9167 | .9167 | .9167 | 1 |
| **1148 FAIL** | **.8970** | **.9167** | **.9167** | **.9167** | **.9167** | **0** |
| 1149 PASS | .9600 | .9167 | .8333 | 1 | 1 | 1 |
| 1185 PASS | .9089 | 1 | .9167 | .9167 | .9167 | 2 |
| **1186 FAIL** | **.8918** | **.9167** | **.8333** | **.9167** | **.9167** | **1** |
| 1187 PASS | .9172 | .9167 | .9167 | .9167 | .9167 | 1 |

Clean HQ is the fraction of the twelve *deterministic* support points within
the ring's .21 radius; the original live HQ samples 4096 points with .029
output noise, so these columns are attribution diagnostics, not substitute
gate scores. All eight clean modes remain represented after the pullback in
each failed update.

The [fixed evaluation replay](continuous-evidence/coverage-failure-replay/fixed-eval.json.gz)
also scores every support stage with the **exact same** 4096 particle indices
and Gaussian output-noise draw that the host uses at that update. Its
postprojection grade equals the original live grade bit-for-bit at all nine
captured steps. At the three failures:

| Update | Noisy HQ before GAN | After GAN | Ideal Lloyd targets | Actual pullback |
| --- | ---: | ---: | ---: | ---: |
| 1133 | .999756 | .750488 | .841553 | .846191 |
| 1148 | .921875 | .952881 | .897949 | .896973 |
| 1186 | .910400 | .858887 | .891846 | .891846 |

Thus the 1148 projection itself changes a passing post-GAN check into a
failing one. At 1133 and 1186 the bounded GAN step first crosses the HQ
threshold; the one-sided correction partly repairs quality but cannot finish
because the off-target cells are empty. The ideal centroid cloud also fails
the real noisy .9 HQ threshold at all three steps, not only the clean proxy.

At 1133, the bounded GAN step moves rows 4, 8 and 9 outside the HQ radius
(nearest-center distances .2693, .2276 and .2971). The pullback corrects row 4
with a .2800 clean-output move, but rows 8 and 9 have **empty sampled-real
cells**, receive no correction, and stay off target. At 1186, row 10 was
already just outside the radius before the GAN step (.2268), worsens to .2641,
and likewise has an empty cell. The pullback repairs another off-target row,
but cannot move row 10. These empty rows are redundant support in modes 2/4;
their mass still matters to the observed HQ even while all eight modes remain
covered.

At 1148, row 4 has **one** assigned real sample. Its clean distance to mode 4
is .2185 after GAN, while that one-sample Lloyd target is .2679 from the mode
center. The actual pullback lands at .2682, faithfully following a target that
is worse for ring HQ. The maximum distance from an actual corrected particle
to its assigned centroid is only .0199/.0089/.0014 at the three failed
updates. In all three cases, the ideal centroid cloud has the same clean HQ
count as the actual projected cloud. Thus the local failures are explained by
empty/sparse sampled-real cells and the GAN state entering the correction;
there is no evidence here of a gross nonlinear Jacobian-pullback error. This
is local step attribution, not a claim that earlier projections did not shape
the state that the next GAN update sees.

A single bounded follow-up hypothesis is a generated-to-real support term in
addition to the present real-to-generated coverage objective: it gives empty
generated cells a data-derived signal. Its nearest-sample target may still be
noisy for a one-sample cell, so it must pass the original 200/200 warm filter
without changing thresholds and then cold acquisition. No such candidate was
tested in this replay.

The [evidence manifest](continuous-evidence/coverage-failure-replay/manifest.json)
hashes the exact original/reference warm JSON, replay controls, all nine raw
support/minibatch captures, analyzed cell targets and per-particle distances,
the fixed-noise evaluator, the source, and a tensor sidecar. The sidecar stores the post-GAN clean
generator `state_dict`, prior `z`, prior Adam diagonal metric and actual D
real minibatch at 1133, 1148 and 1186 for a separate nonlinear
counterfactual. Load it with `torch.load(path, weights_only=True)`.

The evaluator is reproducible without any model call: sample 4096 particle
indices from `torch.Generator().manual_seed(9)`; in a forked global RNG seeded
to `402 + step`, add `.029 * torch.randn_like(selected_support)`; call the
frozen `mode_hold.diversity` against its known ring centers. This is only an
offline diagnostic. `coverage_fixed_eval.py` checks its postprojection score
against the original recorded live score at every captured step.

Reproduce once on the pinned CPU/AVX2 environment with a new output directory:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 DNNL_MAX_CPU_ISA=AVX2
/tmp/pr38-default-env/bin/python -u reports/toy100/coverage_failure_replay.py \
  --output NEW_DIRECTORY \
  --reference ORIGINAL_WARM_FORK_COVERAGE.json
```
