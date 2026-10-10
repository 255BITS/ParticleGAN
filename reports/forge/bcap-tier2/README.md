# BCAP Tier 2 quality and retention results

The selected BCAP recipe keeps **6/6 required Tier 1 passes**, but does not
qualify for Tier 2. Its 21 required Tier 2 tasks finish with **6 PASS, 11 FAIL
and 4 INCOMPLETE**. Gaussian acquisition is transient, the word inverse map
loses accuracy during continued training, and the image hosts cannot construct
DualNorm optimizers for their convolution tensors. The single
[current technique inventory](../technique-inventory.md) contains the updated
whole configuration. [Final metrics and verification](readout.json) retain every
task, including unsuccessful and unsupported executions.

The recipe is the [Tier 1 winner](../bcap-six/README.md), with global
`optimizer_smoothing=1e-5`, zero momentum and constant G/E, D and sampled-prior
rates `.012`, `.018` and `.03`. **No learning-rate annealing** is used. No
hyperparameter, seed, architecture, prior, sampling law, task budget, cadence
or threshold changes in this stage. Seed `0`, public deterministic initialization
and checkpointed named RNG streams retain each task's declared contracts and
fixture exceptions. The [ready study](../../../configs/forge/studies/bcap-tier2-smoothing-v1.json)
and [frozen plan](plan.json) bind this one whole recipe through Tier 2.

## What passed

Four behavior tasks pass: `unipolar`, `cover_leftover`, `mid_scale_identity`
and `mode_hold`. Mode hold retains all eight modes, quality `0.998535` and
effective modes `7.218385`. Two distribution tasks also pass:
`vector_two_broad`, with quality `0.979736` and mass TV `0.042236`, and
`vector_spiral`, with normalized sliced Wasserstein distance `0.051822`.
Each PASS is the declared complete-curve and terminal gate, rather than an
endpoint substituted for sustained evidence.

## Why retention failed

Gaussian stability restores the exact own smoke **endpoint at update 1,000**,
then trains to update 6,000. It preserves optimizer history and RNG state; it
does not restore the earlier passing draw at update 959. Only **1/72 stationary
checks** passes, and the first new check at update 1,042 already fails. After
the target mean changes from `2` to `3`, the learner misses the reacquisition
deadline and passes **0/24 shifted hold checks**. The fixed-state control passes
0/48 shifted checks. At update 6,000, the learned mean is `3.010940`, standard
deviation `0.417360`, and KS `0.053048`, above the `0.05` limit. Movement toward
the new mean is insufficient to establish distribution quality or retention.
All 169 saved live/control sample sets reproduce exactly under the public scorer.

Word hold restores its own **earliest confirmed checkpoint at update 4,167**
and performs exactly 4,000 additional updates, retaining the original schedule
horizon and all optimizer histories. Only **13/25 paired checks** pass; the
first failed check is update 4,501. Generation still covers all five words
with quality `1` at update 8,167 and mass TV `0.022656`. Paired reconstruction
is no longer exact, and minimum correct-token probability is `5.095e-9`, below
the required `0.9`. Thus good generated words coexist with a failing inverse
map. All 25 saved primary/confirmation pairs have identical training-state hashes
and preserve their independent evaluation streams.

## Other failures and setup limits

`trajectory` fails identity MSE: `0.247542` against `<=0.02`.
`residual_student` also fails identity MSE (`0.062671`), reaches success rate
`0.416667` instead of `1`, and retains wrong-pad rate `0.583333` instead of `0`.

Four vector tasks fail: `vector_unequal_mass`, `vector_unequal_width`,
`vector_anisotropic` and `vector_overlap`. Unequal-mass training loses a
component and ends with mass TV `0.316338`; unequal-width mass TV is `0.240234`.
Anisotropic component covariance error reaches `9.323815`; overlap normalized
sliced Wasserstein distance is `0.164992`. The receipt summaries retain the
full gate verdicts separately from these illustrative endpoint metrics.

All three 100-mode native hosts finish **7,000 updates** and fail coverage,
sustained live accuracy or independent holdout. Holdout precision is `0.18098`
for `grid100`, `0.22317` for `rotated100` and `0.34645` for `staggered100`.
Unavailable component accuracy metrics remain null when sufficient covered
components are absent. Oracle scorer controls pass; they do not qualify the
trained models.

`img_stripes2`, `img_bars4`, `img_blobs4` and `img_intensity2` remain
**INCOMPLETE** after setup errors, with no training updates. The frozen public
DualNorm implementation rejects parameter tensors with more than two dimensions:
`dualnorm supports matrix weights and vector/scalar biases only`. Their
convolution architectures were accepted by preflight, but optimizer construction
refused them. These are implementation-support gaps, with no numerical quality
verdict and no training GIF. Supporting convolutions requires an explicit
optimizer adaptation and a separate source identity. The subsequent
[convolution adaptation and four-image rerun](../bcap-convolution/README.md)
now completes all four tasks on a new source, with four sustained FAIL gates.
Its results do not replace these original setup-error receipts.

## Cost and retained evidence

All six required Tier 1 receipts and the clock diagnostic are reused under
identical scientific source, recipe, task, protocol and runtime compatibility
keys. No Tier 1 updates repeat. The 21 new attempts comprise **17 completed
training runs and four setup errors**, with **zero scientific retries**, for
**1,121.805491 paid worker-seconds**. All reservations are released. The declared
through-Tier-2 administrative cap is 43,020 seconds; reuse reduces the new
reservation ceiling to 40,500 seconds. Both A6000 GPUs run one worker each.
The complete-current-tier policy finishes independent peers after failures;
Tier 3 is outside the requested scope and remains unmeasured.

[Seventeen actual-training GIFs and renderer receipts](media/index.json) use
the same selected recipe. They render saved samples or numerical learning curves
under guards that disable model construction, forward calls, training and
sampling. Native media use certified saved snapshots; spiral media show its
numerical curve because the procedural target has no mixture centers. No images
are inspected to select or grade results.

The immutable local archive `artifacts/bcap-tier2-smoothing-v1.tar.gz` contains
byte-verified raw logs, events, checkpoints, saved observations, original receipts,
queue state and frozen source. Its SHA-256, member digest and file count are in
[readout.json](readout.json). Bulk artifacts remain ignored by Git. Compact
metrics, receipt proofs, media and reproduction sources are committed.

The auxiliary study signature names `ks`, whereas the actual scorer field is
`cdf_ks`. The original declaration and its incomplete decision are retained;
the readout explicitly records the reporting correction and actual endpoint
value. This clerical error changes no task criterion, receipt or qualification.

Publication also repairs the family renderer's handling of structured error
reasons. It now displays the image `ValueError` message while preserving the
original reason object and INCOMPLETE grade. Regression tests cover both inline
and catalogued errors. This reporting change is outside the executed scientific
source; it does not qualify a new source cohort or require another training run.
All 58 family-report tests and 155 current-inventory tests pass. Forge validation
passes, and compiled memory reports CURRENT with 666 records and no missing inputs.

## Next comparison

Keep the selected Tier 1 recipe and these Tier 2 outcomes in the inventory.
Inspect the saved Gaussian and word trajectories before another bounded,
global recipe comparison. Smaller constant rates or stronger fixed smoothing
are plausible retention hypotheses, with no evidence here assigning causality
or selecting either change. Convolution support needs its own explicit
adaptation. Tier 3 cannot qualify this failing Tier 2 recipe; the provisional
screen grants no public-default adoption.

## Reproduction

Execution uses scientific source digest
`801d07b11ff269f441d7960dbc445939aa770dbceebdca5e315432c76b46b97a`,
with Tier 2 declaration commit `f96819ddb`. It is scientifically identical to
the Tier 1 producer; original receipt origins are preserved. Use a checkout of
the executed declaration commit `f96819ddb`, restore the archive's ignored
execution paths at their recorded root, and use the recorded
Python/Torch/NumPy/SciPy and A6000 runtime before checking reuse:

```sh
python reports/forge/bcap-tier2/run.py --stage plan
# Explicit execution; all seven Tier 1 receipts must already be reusable.
python reports/forge/bcap-tier2/run.py --stage enqueue
python reports/forge/bcap-tier2/run.py --stage drain --gpus 0,1
tail -F runs/forge/events.jsonl
```

The completed archive and existing queue need no further training. Register
original evidence with `refresh.py`; subsequent publication refreshes use
`python reports/forge/regenerate_technique_inventory.py`. The compact readout
is a display artifact; independent regrading uses byte-exact original receipts.
