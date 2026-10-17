# Generator-step pair: both complete image protocols fail

Atlas and E22 each completed the original 600 updates for `image-develop-img_intensity2-source-transpose12`. Both original gates and the added first-acquisition/hold gate are **FAIL**. None of the 25 captured observations passed, so neither run acquired the required five-check window. The seven later cases per family remain **UNKNOWN**: two failed cells and fourteen unknown cells across the full sixteen-cell denominator, with zero qualified whole configurations. The sixteen separately supported capacity witnesses are not training passes.

The scientific source is `488b792e2fb875894f017cf7043420f2bb66190f`; execution snapshot digest is `ac779e01f99f725951778c5000443ae9c5c215f0fd17e74e9c6be53e6c929bd9`. The explicit shared tuple was `lr=.00265625`, `prior_lr_mult=3`, `d_lr_mult=4.5`. It halves nominal G/output-noise learning rate relative to the preceding critic-balance experiment while retaining nominal prior and critic rates. The image host, initialization, target, primary sampling law, fixed numeric gates and original horizon remained unchanged. Common public package/provider/scorer files have identical hashes across these two source cohorts; their helper manifests and declared Recipe rates differ.

The target is an equal mixture of two 8×8 grayscale images, with a central 4×4 patch at .35 or .85 and zero elsewhere. Primary observations sample 1,024 draws from the public selected serving law with additive output noise omitted. DV12 latent perturbation remains active. All captured image observations selected fast weights; an enabled `serve_average=4` does not mean an average was selected. The .06 image-neighborhood radius is a per-image classification cutoff, not a separate bound on aggregate mean RMSE.

| Final gate | Observed, both families | Required |
|---|---:|---:|
| High-quality fraction | 0 | ≥ .9 |
| Represented modes | 0 | 2, with each accepted mode mass ≥ .25 |
| Nearest-template distribution TV | .5 | ≤ .1 |
| Finite-template TV, including rejected mass | 1 | ≤ .1 |
| Rejected mass | 1 | Enters finite-template TV |

Every captured observation fails the same four named bounds: `distribution_tv`, `finite_template_tv`, `hq`, and `modes`. Final mean RMSE is .17500434815883636, a diagnostic. The learned training output-noise standard deviation is .053762443363666534; it is not added to these primary evaluation draws.

The saved arrays show an early dark-output collapse. Mean patch intensity falls from .5064423 initially to .0000265159 at update 25, .000000358692 at 50, and .0000000000465174 at 600. At update 600 the maximum saved patch value is .0000000699051; 89.5004% of all saved pixels are at most .0001. These summaries use retained arrays only, without model sampling or rescoring. Relative to the predecessor, patch intensity and local sigmoid sensitivity are less suppressed and terminal G gradient norm is larger, but both cohorts miss every primary image gate. A numerical improvement within a failing dark cloud is not recovery of either target mode.

The final retained G gradient norm is .0004195018846075982. Its output-bias gradient is positive .00004680943311541341, so a local minimizing step points darker at this endpoint. The terminal effective critic rate is .0003848624664329677, approximately 3.219% of the nominal .011953125, alongside a smoothed payoff-error value of 5.489766837193128. Public policy applies reciprocal-square critic payoff damping, but the rate was assigned before the final controller observation; these two endpoint values are not one contemporaneous rate/error equation. Optimizer moments and prior positions differ from the predecessor, so preserving nominal prior/D rates did not preserve their realized dynamics. No saved training-loss stream or intermediate gradients establish which owner initiated the collapse.

No realized birth/move, isolation, row reset, controller reopen, surprise fire or anchor event is recorded for this run; Atlas also records no settled-guard epoch rebase. Those facts exclude an observed such event as the immediate change in this recorded cohort. They do not identify why learned G/prior dynamics entered the dark region or prove that payoff damping, optimizer memory, prior movement or saturation caused it.

The two families have exact equality in all 25 retained observation records, 275 numeric metric scalars and 50 NPZ arrays containing 1,641,600 scalar values. The NPZ and GIF files are byte-identical. Final G, D, prior, EMA-G and EMA-prior tensors, optimizer state, named training/evaluation streams, controller and row-evidence state are identical. Whole checkpoints are **not** identical: Recipe/backend/settled-guard metadata and saved global CPU/CUDA RNG states differ. Captured outputs and endpoint owners do not prove equality at unrecorded update boundaries or complete policy equivalence. Atlas selects the small-population reference-kNN fallback on this 32-row host; its distinct 20,000-row native feature-cell question, and possible settled-guard behavior, remain unattempted. These are reasons to retain the named families without borrowing native success.

New scientific supervisor costs are 28.03618986881338 seconds for Atlas and 26.841381517937407 seconds for E22, totaling 54.87757138675079. The earlier engineering debit of 4.757908704923466 and earlier scientific debits of 27.0030500178691/27.3557217749767 remain charged once. Cumulative campaign cost is **113.99425188452005 seconds** under the unchanged 15,360-second ceiling. There is no timing rank or new reservation.

The pair JSON records a slower-critic tuple as **PROPOSED_NOT_EXECUTED**. It is a conditional sensitivity question, not a causal conclusion or accepted acquisition. Any later named KA2/K3P serving-family proposal belongs to a separate prospective report and receives no success or source credit from this diagnosis.

This analysis constructs/restores no models, draws no new samples, calls no scorer, performs no optimizer update and opens no CUDA context. Original scientific grades and artifacts remain unchanged.

Bound evidence and compact numeric details:

- [Pair evidence JSON](pair-intensity-analysis.json), SHA256 `a9a89c87b9ac1a0d6e659c4e6947d6de20a8be76080b15807529d911fe7af60e`.
- [Atlas descriptive comparison](atlas-intensity-analysis.json), SHA256 `3c878f9333d951643306ba557d2700113d281309f8af2d169e624c1999094fb7`; its pending-E22 field describes the earlier capture time and is superseded by the pair report.
- [Atlas receipt](/ml2/hypergan/forge-generator-step-20261003/atlas/atlas--d2e6767ae639367ca480c4271d72f20700a2d03e95242f4dd1ba75bfc01ed3f3/image-develop-img_intensity2-source-transpose12/receipt.json), SHA256 `3faae2709afbea454b747ecde87169015792bf1f9eb2f2561b6e090f8d8a40ad`.
- [E22 receipt](/ml2/hypergan/forge-generator-step-20261003/e22/e22--accb716670730414f49af2a9bce2cd4c373d7df69bdcebfbf495637349e928a1/image-develop-img_intensity2-source-transpose12/receipt.json), SHA256 `d20dd05b23c10a2ecaac760da441a6f53a44ef67e5041246ab1dd17f92682e90`.
- Public source boundaries: `particlegan/continuous.py` payoff damping and smoothed error; `particlegan/policy.py` rate assignment and serving selection; `particlegan/training.py` ordered critic/G updates; `benchmarks/toy_audit/api_images.py` primary finite-template gate; `benchmarks/toy_audit/api_run.py` retained observation/final-state scope. Exact source-file hashes are bound in the JSON receipts.
