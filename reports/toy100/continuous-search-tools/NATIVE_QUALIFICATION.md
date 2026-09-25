# Fixed native qualification matrix

Only a candidate that survives its own canonical and transfer gates advances to
this matrix. The user requires grid100, rotated100 and staggered100 each at seeds
1234–1237: **12 full 7000-update coverage AND accuracy passes**. The three seed1234
cells are already part of the 22 original toys. Reuse valid evidence for those
same cells when source/runtime identity is established; do not repeat them merely
to create a newer result. The other nine runs are fixed qualification, not seed
search. A failed cell stops advancement; preserve ERROR and incomplete results.

`native_seed_wrapper.py` keeps the candidate files unchanged and overrides only
the seed in the driver's input configuration. It enforces the declared seeds and
7000-update budget, checks an exact source pin before and after execution, and
saves the original driver, executed source, wrapper and optional observer hashes.
It rejects unfamiliar driver setup instead of guessing a patch. The frozen
native host still computes both canonical suites with their original limits.
The wrapper does not decide whether a candidate is eligible or whether the whole
matrix passes; a successful process exit alone is not a gate pass.

First inspect the qualifying candidate's pin:

```sh
python3 reports/toy100/continuous-search-tools/native_seed_wrapper.py --describe CANDIDATE
```

Then run one required cell in a fresh directory, using the approved runtime,
CUDA environment, one worker, and the reported `sha256`:

```sh
/tmp/pr38-default-env/bin/python -u reports/toy100/continuous-search-tools/native_seed_wrapper.py \
  --seed 1235 --source-sha256 SOURCE_SHA256 --observer CANDIDATE/native100.py \
  --candidate CANDIDATE --repo FROZEN_RUNTIME --task grid100 --output FRESH_OUTPUT
```

Use `--observer` only when the candidate's policy query is pure and the audited
same-update rate observer applies. That observer's source hash is fixed. It fixes
validation timing; it does not assign rates or change updates. A new installation
still needs review. Preserve the candidate's complete runtime provenance and
verify the frozen runtime separately from the candidate pin. Every seed/layout
must share the identical formulation and source pin; do not mix variant scores.

[Configuration validation](../continuous-round-3/native-seed-wrapper-validation.json)
checks all twelve configurations against the real resolver, rejects invalid
seeds/driver setup/short budgets, detects changed source, and confirms unchanged
CPU RNG and candidate files. **Zero training updates were run.** Syntax was checked
with and without the observer; these are harness checks, not candidate passes.
