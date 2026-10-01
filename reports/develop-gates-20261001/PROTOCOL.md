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
`/tmp/particlegan-develop-atlas-qualification-20261001/`. Only compact results,
source identities, reproduction scripts and interpreted readouts belong in Git.
New quality evidence applies only to the original Atlas cloud/noisy cohort;
it does not qualify Forge's learned-MoG/clean-live tasks or a public-default
promotion. E22's original evidence remains separately labelled.
