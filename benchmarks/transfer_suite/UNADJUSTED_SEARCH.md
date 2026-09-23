# Find one shared ParticleGAN default

The [unadjusted leaderboard](../../reports/transfer_suite/unadjusted/README.md)
is the primary comparison. The winner is **19/19 with `shared_c6` and declared
discriminator choices**; its reference D profile scores 15/19. The initial
baselines are **8/19 for `gan_v2`** and **5/19 for `gan_v1` / `gan_legacy`**.
Public `get_recipe()` now selects `gan_v3`, the promoted `shared_c6` recipe.
The [installed-package verification](../../reports/transfer_suite/public_v3_promotion/README.md)
passes all 19 using the public trainer and the required custom-host routes.
The earlier 19/19 uses different optimizer settings per host and does not qualify
here. The objective is one unchanged recipe that passes all 19 live tests.

## Run a candidate

To reproduce the 19/19 shared recipe with its explicit discriminator
profile in one command:

```sh
python -u -m benchmarks.transfer_suite.shared_profile_search \
  --plan reports/transfer_suite/unadjusted/leading_profile.json \
  --output /tmp/shared-c6-profile > /tmp/shared-c6-profile.log 2>&1
tail -f /tmp/shared-c6-profile.log
```

The [full replay controls](../../reports/transfer_suite/unadjusted/runs/rare-profile-replays/README.md)
exactly reproduced both the earlier 18/19 profile and the winning 19/19 profile.
The unequal-mass winner uses batch-dependent neighbor-distance features;
see the [findings](../../reports/transfer_suite/unadjusted/FINDINGS.md).
Architecture cards are explicit;
recipe overrides remain global. Change the candidate name and global overrides
in a copy of the plan to compare another recipe using the same profile.

Create a JSON file, for example `/tmp/my-candidate.json`:

```json
{
  "candidates": [
    {
      "name": "my_shared_candidate",
      "overrides": {
        "lr": 0.00425,
        "d_lr_mult": 1.0,
        "prior_lr_mult": 1.0,
        "betas": [0.0, 0.99],
        "reg_coeff": 3.0,
        "reg_kappa": 1.25,
        "prior_reg": 0.05
      }
    }
  ]
}
```

These are example search settings, not a claim that this candidate passes.
This historical search runner pins unspecified fields to `get_recipe("gan_v2")`.
Use the explicit leading profile to reproduce v3; public `get_recipe()` selects v3. The resolved full recipe is
recorded for every run. The same card applies to all tests.

```bash
python -u -m benchmarks.transfer_suite.shared_default_search \
  --plan /tmp/my-candidate.json --output /tmp/my-shared-candidate \
  > /tmp/my-shared-candidate.log 2>&1
tail -f /tmp/my-shared-candidate.log
```

Use a new output directory. The runner defaults to all 19 tests on CPU with one
Torch thread and seed 0. It writes full live/EMA curves, actual optimizer-group
settings, action traces, source archives and an incrementally updated report.
Use separate output directories/processes for independent candidates.

The current runner supports shared `lr`, `d_lr_mult`, `prior_lr_mult`, `betas`,
`prior_betas`, `reg_coeff`, `reg_kappa` and `prior_reg` overrides. There are no
per-test options. A new generic optimizer or adaptation rule can be added as a
separate candidate mechanism, with integration/parity checks and its exact source
archived. Its rule must be the same everywhere; do not branch on task names or
feed evaluation metrics/target labels into training.

`shared_schedule_search` additionally accepts global `lr_anneal_start` and
`lr_floor` changes around `shared_c6`. It applies the declared cosine schedule
through the existing controller on every host and records actual multipliers.
The importer checks every recorded action against the recipe's schedule equation.
Schedule screens remain incomplete until all 19 cases run with that recipe.

## Rules for comparison

- One candidate means one complete recipe and update rule across all tests.
- Keep frozen data, initialization, particle counts, batch sizes,
  budgets, auxiliary objectives, behavioral metrics and thresholds unchanged.
- A PASS needs all metrics for at least five final observations of the complete
  24-checkpoint live curve. Overall PASS needs all 19 tests. EMA is separate.
- Do not pool different recipes' passing cases or select favorable checkpoints.
- No seed sweeps. Retain every failed attempt and its complete evidence.
- Optional top-level `"tasks": ["two_pole", "vector_unequal_mass"]` screens a
  subset. Mark these results INCOMPLETE. Run the remaining tests with the exact
  same recipe before claiming a full score.
- Architecture remains separate from formulation identity. Discriminator
  variants are allowed under the same unchanged recipe, with every architecture
  trial and failure recorded. The ordinary runner uses the reference profile;
  a research runner must declare a `discriminator_variant` card in each episode.
  The importer accepts explicit vector D variants and shows reference-profile
  performance separately. Changing the optimizer recipe per example is prohibited.
- A longer training budget is a separate toy and cannot replace a failed run at
  the original budget.

## Submit a result

1. Copy the complete output folder into a new unique folder beneath
   `reports/transfer_suite/unadjusted/runs/`; include source archives and failures.
2. Add an entry to `reports/transfer_suite/unadjusted/entries.json` with your
   candidate `name`, readable `label`, and repository-relative `indexes` paths.
   Multiple screening/completion indexes are allowed only for one identical
   recipe. Duplicate tests require distinct declared discriminator variants;
   rerunning an identical architecture does not supply another selection chance.
3. Run `python -m reports.transfer_suite.unadjusted.build`. It checks matching
   test setups, unchanged per-case recipes, actual LRs/betas, source hashes and
   behavioral verdicts, then regenerates the readable leaderboard.
4. Commit the runner changes, recipe, complete artifacts and regenerated report.
   Describe the mechanism and failing cases; report measured results without an
   all-pass claim unless every case passes.

The existing `compare_defaults` adapter has tests for mixed G/particle groups,
AE prior betas, direct particle optimizers and exact control replay. Run relevant
tests when changing that integration; do not silently bypass an ignored setting.

## Discriminator architecture trials

Keep `original_spec` equal to the frozen reference job. Declare the D change as:

```json
{"discriminator_variant": {"name": "plain_d128", "overrides": {
  "d_hidden": 128, "d_layers": 2, "fourier": 0,
  "research_discriminator": null
}}}
```

Use `shared_variants.architecture_spec(original_spec, discriminator_variant)`
before applying the unchanged recipe with `effective_spec`. Only `d_hidden`,
`d_layers`, `fourier` and `research_discriminator` may change. The null research
card restores the ordinary MLP; custom cards need their exact constructor in
the archived source. Set the episode's `architecture` to the variant name.
Keep all trials in the submitted indexes, including the failures. The importer
counts a test once if a declared architecture passes, while retaining the
original-profile score and links to every attempt. This measures architecture
support within one recipe, not a universal network architecture.

## Independently replay a discriminator result

The shared-cap6 architecture runner has a reusable exact replay command:

```sh
python -u -m benchmarks.transfer_suite.replay_shared_architecture \
  --reference reports/transfer_suite/unadjusted/runs/shared-discriminator-search/cross/episodes/shared_c6__raw_softplus96_l3__vector_overlap.json.gz \
  --output /tmp/shared-overlap-replay
```

For another architecture implementation, pass its Python module with
`--implementation benchmarks.transfer_suite.shared_pointnorm_research`, for
example. The module must expose the declared constructor and variant card.
The command uses the canonical host and shared recipe, freezes current source,
and compares all live/EMA observations, actions, optimizer receipts and verdicts
against the retained episode. Only timing fields are excluded. It retains
evidence before failing on any mismatch. Validation replays do not add selection
points or new architecture trials.
