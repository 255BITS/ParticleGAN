The observed word law has a structural joint-distribution mismatch. This does not establish that the finite toy gates are impossible or identify the cause of the recorded training failure.

This independent review uses immutable source `fb7acc775b3a1a6184d36b55e035b9da04531492`, snapshot digest `f380eed990931bacb205e6537beaf387fdbb676ffc97b6f32f19ff903ae1cfed`, and the retained `atlas_word_joint_min11` / `word_joint_policy_min11_v1` evidence. The original execution remains **INVALID**, its numerical gate remains **UNAVAILABLE**, and its 558.5739127129782 paid seconds remain unchanged. The rejected owner-health grade is retained as provenance, not an accepted numerical FAIL. This report performs no numerical regrading, qualification, default adoption or speed comparison.

The real joint is `(one_hot_word, E(word))`. Its code marginal has at most five deterministic atoms. The fake joint is `(G(z_effective) + word_noise, z_effective)`: DV12 perturbs a uniformly drawn prior row before the same effective code reaches both G and the joint critic. Gaussian output noise touches only the 168 word coordinates. E receives adversarial gradients through the real-joint term; this caller has no reconstruction training loss. Selected observation turns additive output noise off but retains DV12 for both prior generation and encoded reconstruction queries.

The actual backend is `knn/controller_reference`, selected because `caller_callbacks_own_representation`. This conclusion does not assume that every policy backend adds continuous noise. The reference DV12 law scales a Gaussian by geometry-derived bandwidth and clips its norm to half the nearest nonzero support distance. Zero bandwidth or degenerate support can eliminate perturbation. Such a degeneracy is not what the retained observations show.

All 43 inputs from the prior goal analysis were independently hash-checked. This report also pins that analysis, its human report and the feature-backend source, for 46 proof inputs. Every saved code comparison was reproduced directly from the existing NPZ arrays with `allow_pickle=False`; no checkpoint was deserialized, model imported, forward executed, sample drawn or official score recomputed.

| Saved evidence | Exact count |
| --- | ---: |
| Selected observations | 24 |
| Generated codes compared | 24,576 |
| Generated effective codes differing from their raw row codes | 24,576 |
| Generated effective codes equal to any corresponding E(word) code | 0 |
| Encoded inverse queries compared | 120 |
| Inverse effective codes differing from E(word) | 120 |
| Prior tables retaining 11 distinct rows | 24 of 24 |

Final recorded training bandwidth is `[0.12243864685297012, 0.18044087290763855]`. The last two recorded training applications have perturbation RMS `0.10547961294651031` and `0.11041469871997833`, with positive minimum radius `0.07111953943967819`. Learned training output sigma is `0.028999999165534973`; real words remain one-hot with input noise zero. All 24 selected snapshots were `fast`.

Under the usual continuous-Gaussian idealization, the observed positive two-axis bandwidth and support radii make the fake code distribution non-atomic, including its radially clipped boundary component. It assigns the five deterministic real-code points zero probability, so exact full joint equality is incompatible at those fixed observed states. Positive fake-only word noise adds another mismatch. This mathematical statement is conditional on the usual real-arithmetic noise model; actual finite-precision/PRNG support is discrete. The saved arrays independently demonstrate disjoint observed code sets, not a universal distribution theorem for every possible family state.

The finite evaluator asks for normalized, confident canonical word probabilities, five modes, mass TV at most .1, correctly paired all-five reconstruction and minimum reconstruction token probability at least .9. It does not demand equality of latent code distributions. Mapping neighborhoods to confident word outputs could satisfy these finite measurements without exact joint equality; no such capacity witness has been established here. Nor have the available observations established how much the finite regularized critic exploits the mismatch. The scaffolds and the arithmetic allocation `(2,2,2,2,3)` alone do not certify capacity under DV12 and the actual selected law.

At most two rate-only contrasts remain **proposed and unexecuted**:

| Tuple `(lr, prior_lr_mult, d_lr_mult)` | Narrow question and limit |
| --- | --- |
| `(.001328125, 1.5, 1)` | Quarter nominal motion at preserved role ratios. Continued failure would weaken a simple excessive-nominal-step explanation; endogenous policy trajectories need not match. |
| `(.0053125, .15, 1)` | Slow nominal row updates tenfold. Continued failure would weaken nominal prior chasing; birth/death transport and DV12 geometry remain active. |

These preserve seed 0, architecture, owners, objective, 20,001 updates, 24 observations of 1,024 draws, original thresholds and the terminal-five requirement. Each needs its own source-bound declaration and complete 900-second allowance, with at most 1,800 new seconds, no retries or budget reset. Neither contrast establishes capacity or causal identification. A successful finite measurement would not imply exact full joint matching.

If both fail, a later matched-noise family would be a separately declared objective and serving law: specify the real encoded-code kernel against the fake prior-code kernel, word-coordinate noise on both critic distributions, critic-input/regularization consistency and selected inverse law. Matching kernels alone does not match mixture centers or masses. Fresh complete owner, gradient, RNG, checkpoint and capacity evidence would be necessary; no original credit or predicted winner follows from this concept. No such variant is implemented or proposed for execution by this review.

The remaining evidence gaps are a capacity witness under the exact law, same-state DV12-off inverse observations, complete per-update parameter/loss trajectories, earlier complete checkpoints and evidence of which distinction the finite critic actually uses. Original INVALID and numerical UNAVAILABLE remain authoritative.

Source lines below refer to the exact pinned snapshot. [review.json](review.json) contains full file hashes, byte counts, the 46-input roster, all 24 per-observation code counts and the preserved status/cost/source identities.

| Pinned source | Lines | Meaning |
| --- | --- | --- |
| `experiments/forge/word_joint_policy_adapters.py` | 43–60, 138–146, 153–174 | Complete effective-code joint, words-only noise and actual objective/updates |
| `experiments/forge/word_joint_policy_adapters.py` | 262–293 | Selected generation/inverse law, saved effective codes and purity |
| `particlegan/feature_policy.py` | 100–135 | Callback-owned knn/backend selection |
| `particlegan/continuous.py` | 134–192 | Geometry-derived bandwidth and DV12 clipping |
| `particlegan/policy.py` | 137–155, 919–957, 966–991, 1131–1159 | Actual training/selected perturbation, output sigma and complete selected state |
| `particlegan/particle_prior.py` | 94–114 | Uniform raw row selection |
| `particlegan/gan_loss.py` | 6–23 | Paired logistic joint loss |
| `benchmarks/toy_audit/api_images.py` | 1020–1047 | Original finite G/E/D scaffolds |
| `benchmarks/toy_audit/definition_quality.py` | 102–127 | Finite word-probability and inverse measurements |
| `experiments/forge/word_joint_policy_contracts.py` | 31–34, 136–145 | Exact thresholds, full horizon, sample count and terminal checks |

The passive reproducer requires the original local raw files and snapshot. This directory contains only metadata and code, with no arrays, checkpoints or logs. It does not hydrate missing evidence.

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /ml2/hypergan/.venvs/particlegan-develop-integration/bin/python \
  check_retained_codes.py --report review.json
```

The relative command runs from this report directory. Any Python environment with compatible NumPy can perform the saved-array comparisons; the environment does not supply scientific runtime qualification. Missing or changed pinned inputs fail closed.
