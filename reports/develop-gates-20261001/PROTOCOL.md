# Develop gate investigation, 2026-10-01

The user authorized diagnosing and resolving the post-merge failures while
keeping changes on develop. This study investigates test identity and replays
the existing Atlas qualification; it does not search seeds or tune a recipe.

The merged tree is `fa2c378d5ea4dd8f66eabeb17973818a174732ef`. Software CI
passed, while Actions run `36829031177` recorded 19/22 on the legacy BCap
candidate. Its native samples were clean; its transfer samples included the
declared output noise. PR #209 introduced the native clean-sampling change
before the #155/#223/#221 integration. The current CI job does not select E22
or Atlas and does not run their original 19-task qualification.

## Saved-draw diagnostic

`compare_sampling.py` compares each original native clean cloud with the same
cloud plus the recipe's fixed .029 output noise, using the original evaluation
noise seeds. It measures all five saved terminal observations and the separate
100k holdout under the unchanged coverage and accuracy limits. Source sample
hashes are retained. This is an explanatory paired-draw diagnostic, with zero
training updates and no fresh qualification credit. It leaves the original
clean FAIL results and artifact unchanged.

## Fresh Atlas replay

`replay_atlas.py` freezes all current package files and the byte-identical
`configs/100gaussians/atlas.json`, SHA256
`a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4`.
It uses the retained RA15 construction adapters, original CUDA host fixtures,
scorers, seeds, budgets, observation schedules and primary noisy sampling law.
All external research sources are hash-bound before and after execution.

The denominator is fixed before execution: three native 7,000-update gates,
13 portability gates (mode hold, four images, six vectors, stationary and ring
shift), and three moving 1,500-update gates with two original 30-degree turns.
Native gates require all five terminal 20k-draw checks and their independent
100k holdout. Moving gates retain the original 95-mode and relative-quality
criteria. No shorter prefix, clean diagnostic or historical endpoint can fill
a required cell.

Execution uses original physical GPU0 / RTX A6000, float32, serialized
backward, one CPU thread, and at most .2 of GPU memory. It respects the retained
serial numerical-phase lock and does not signal other workloads. Timeouts are
2,400 seconds per native gate, 1,800 per other gate, and 10,800 for the whole
verification. Timing is recorded under current contention, without speed claims.

Source snapshots, raw logs, sample clouds and checkpoints remain in
`/tmp/particlegan-develop-atlas-qualification-20261001-attempt2/`. The first
attempt remains archived separately: a missing harness import stopped it
before training, at zero updates. Only compact results,
source identities, reproduction scripts and interpreted readouts belong in Git.
New quality evidence applies only to the original Atlas cloud/noisy cohort;
it does not qualify Forge's learned-MoG/clean-live tasks or a public-default
promotion. E22's original evidence remains separately labelled.

## CPU numerical diagnosis

The restored served-law BCap gate passed locally but Actions run
`36886525044` measured **19/22 FAIL**: grid100 failed coverage and fidelity,
and vector_unequal_mass and vector_unequal_width failed their retained gates.
This corrects the initial native-only count after independently grading all
19 transfer artifacts; no acceptance rule or recorded result changes. These
results remain real failures in their executed runtime. Its
initial samples match the earlier clean run exactly; paired sample differences
begin at float32 rounding scale on update 1 and amplify thereafter. Both jobs
used the same CPU PyTorch build, ISA caps, recipe, seed and training budget.

`probe_cpu_sampling.py` compares clean/served scoring under that exact CPU
wheel at native resource sizes. It stops each branch at 40 updates while
preserving the 7k schedule and original observations. It compares complete
checkpoints and RNG states. The first execution rejected a mismatched prefix
budget before training; the corrected diagnostic costs 80 updates. A second
80-update diagnostic adds only `MKL_CBWR=COMPATIBLE`. Neither earns quality
credit or changes a seed.

The portable CPU profile, revision 2, pins this explicit BLAS arithmetic
branch in addition to the existing ISA caps. Run one local complete common-22
replay and the resulting automatic CI job, at the original budgets and limits.
Keep their receipts distinct from the earlier runtime; do not relabel its
grid100 failure. The local full-suite execution is bounded to 2,400 seconds,
one CPU thread and no GPU. No further recipe or numeric-profile search is
authorized by this protocol.

The wall-time envelope is amended after measuring the new profile: its first
two local native gates required 957 and 833 seconds, so a full suite exceeds
the old 40-minute local envelope and approaches the old 45-minute CI limit.
Increase future CI executions to a bounded 75 minutes, retaining every update
budget, criterion and numeric setting. The active 45-minute job keeps its own
limit and receipts. Only an execution that times out may be repeated under
the amended envelope; cancel the duplicate if the active job finishes. The
local timed attempt is retained as incomplete rather than relabelled as a
failure or a complete qualification.

Intel documents that ISA selection, array alignment and arithmetic branches
affect reproducibility, and that the compatible CNR branch supports non-Intel
CPUs. These explain a plausible source of the observed sensitivity; they are
not proof of the particular hosted CPU used by the earlier job, whose CPU
identity was not captured. The new workflow archives its CPU/runtime profile.
See [Intel's code-branch documentation](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-0/specifying-code-branches.html)
and [reproducibility conditions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2024-1/reproducibility-conditions.html).

The native grader's manifest change receives an explicit evaluator revision
in all 14 affected Forge tasks. Their execution declarations, sampling laws,
thresholds and budgets are unchanged. Their evaluation fingerprints change;
previous receipts retain their frozen identities and receive no implicit new
qualification.

## Concluded result

The independent original Atlas replay is **19/19 PASS**, with all three static
native endpoints exactly matching the original qualified metrics. The public
formulation, Atlas configuration, original gates and update budgets are unchanged.

The extended portable CPU execution, Actions run `36897240453`, completed all
required budgets and remains **19/22 FAIL**: native 3/3 and transfer 16/19.
Trajectory, mode hold and unequal-width vectors fail their retained live gates.
The independent grader agrees with their stored verdicts and validates the
captured executable source and common sampling law. The timed local/45-minute
attempts remain incomplete; they are not aggregate passes or quality failures.
See the [final readout](README.md) and hash-bound compact receipts.

This study closes without further training, tuning, seed changes, continuation,
default promotion or release. Original clean diagnostics and the existing Forge
clean-MoG failures remain valid; future adoption needs its own justified,
policy-aware scientific contract matching the intended serving/prior scope.
