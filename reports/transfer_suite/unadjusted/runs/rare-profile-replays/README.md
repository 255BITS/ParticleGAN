# Complete-profile replay controls

The `shared_profile_search` command exactly reproduced both complete profiles:

| Profile | Cases checked | Live result | Numerical comparisons |
| --- | ---: | --- | --- |
| Earlier LayerNorm β4 unequal-mass model | 19 | 18 PASS, 1 FAIL | 190/190 exact |
| Winning batch-distance unequal-mass model | 19 | **19 PASS** | 190/190 exact |

These are fresh full-budget runs checked against the selected episodes.
They validate the convenience runner and assembled evidence, adding no selection
trials. The original [plan](initial/plan.json) remains unchanged; the
[winning plan](winner/plan.json) is the reproducible 19/19 baseline.

[Initial checks](initial/checks.json) and [winner checks](winner/checks.json)
compare ten mandatory fields per case, including
the recipe, canonical host, declared discriminator, actual optimizer groups,
all live/EMA observations, actions and verdicts. Only timing fields are ignored.
The complete [source manifest](initial/protocol.json), [source archive](initial/source.tar.gz)
and [raw episode index](initial/index.json) are retained. The winner folder
has the same evidence. Generated artifacts remain local under the repository
[artifact policy](../../../../README.md); source and reproduction plans stay in Git.

From the repository root:

```sh
python -m reports.transfer_suite.unadjusted.runs.rare-profile-replays.verify
```

The verifier requires all 19 canonical cases and all ten checks, validates
reference and artifact hashes, and checks the exact archived source manifest.
It performs no training.
