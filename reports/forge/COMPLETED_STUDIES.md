# Completed API studies on the shared leaderboard

The [current trainer-family leaderboard](technique-inventory.md) includes the completed publications from PRs 266, 267 and 268. Its [JSON](technique-inventory.json) stores their exact gates, sampling laws, host recipes, source/runtime identities, costs, archive cards and original goal GIFs under `completed_api_studies`.

These publications answer different questions:

| Study | Question and recorded answer |
| --- | --- |
| Historical Atlas19 | Does the original Atlas recipe pass all 19 original host protocols? **19/19 PASS**, including native noisy coverage and accuracy with an independent 100k check. |
| C6 hold extension | Does the best original broad-distribution checkpoint stay converged for the required five later checks? **Both families FAIL, 3/5 later checks pass**; their original 1200-step PASS and incomplete study grades are preserved. |
| Critic-rate contrast | Does the unchanged higher-critic-rate recipe learn all eight required domains in each family? **2 FAIL / 14 UNKNOWN** after the first image gate fails; all 16 cold capacity checks are supported. |
| Generator-half contrast | Does halving the generator rate while preserving nominal critic/prior rates fix that image gate? **2 FAIL / 14 UNKNOWN**, with the same full denominator and 16 supported cold capacity checks. |

The ordinary family selections retain their complete results. Earlier word-only diagnostics remain historical motivation and reproduction evidence. Historical and current protocols have different hosts and observation laws; their cells cannot qualify one another. Cold capacity uses no optimizer updates. No complete shipping default or fair speed winner has been established by these additions.

From a checkout, rebuild the shared publication without training or raw-log hydration:

```sh
python reports/forge/regenerate_technique_inventory.py
```

The reducer verifies the [committed registry](completed-studies.json), three result JSONs, three archive cards, six readouts and all 31 original GIFs before writing either output. Missing or changed pinned inputs fail before publication. The projector source hash and input pins are recorded in the shared JSON; repeated regeneration is byte-identical. It does not open the raw paths recorded inside those reports, rescore samples, or independently certify an archive.

Read the expandable protocol details and follow the goal GIF/readout links to see what each test verifies. The [C6 baseline diagnosis](c6-baseline-debug-20261003/README.md) explains why mode and mass checks can pass while projected distribution shape fails. Its evidence is linked and hashed separately from qualification.

Paid supervised-child intervals and conservative reserves are separate fields. Generator cumulative cost already includes the critic study and startup once; historical Atlas19 and its hold extensions have a separate allowance. These are accounting totals, not comparable speed rankings. Bulk archives remain **LOCAL_ONLY**, with **NOT_PERFORMED** remote replication and no inferred retention guarantee; share the committed scores/GIFs and use each archive readout for the actual resolver.
