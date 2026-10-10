Both inspected develop CI runs failed only the maintained tier-report freshness check. The latest head `221f1c70447e7934f1aaf56ae30164c39f6e0aec`, run [37175573781](https://github.com/255BITS/ParticleGAN/actions/runs/37175573781), completed with 1 failed, 4,664 passed, 72 skipped and 1 xfailed. Python 3.10 and 3.11 passed their wheel smokes. Python 3.12 stopped at the test failure, so its separate renderer and wheel steps were skipped; release distributions were skipped.

The failing test is `tests/test_forge_tier_report.py::test_committed_tier_report_matches_current_declarations_and_artifacts`, line 524. The primary log shows 122,613 leading characters and 82 trailing characters unchanged, with only the published artifact input digest differing. Both committed heads retain `efe7547e…`; fresh rendering expects `44329581…` at the latest head and `6787e571…` at prior head `1f15b96b87da3638c5b49334457c2a7f9daca7cd`. The pytest minus/plus excerpt is expected/actual, so these values must not be reversed. The prior inspected PR run [37173231063](https://github.com/255BITS/ParticleGAN/actions/runs/37173231063) likewise had only this failure and 4,656 passing tests. Its parallel push run is recorded as failed in API metadata; I did not inspect a second primary job log for it.

The source explains the drift. `tier_report.build_report` includes `research_artifacts.build_artifacts`, which hashes the exact mainboard JSON at `research_artifacts.py:128–129`, explicit readouts and selected media. It returns a canonical input-map digest at lines 205–206, rendered in the footer by `tier_report.py:325`. A presentation or provenance change to `technique-inventory.json` can therefore stale this footer while preserving all scientific results. The task/view declaration digest is unchanged.

The smallest repair is to finish the new one-table mainboard output and declaration-only compile outputs, then run:

```sh
python -m experiments.forge experiments-by-tier --output reports/forge/EXPERIMENTS_BY_TIER.md
```

Validate with the unchanged focused freshness test and a second generation for idempotent bytes. Generate from the final inputs rather than pasting either old CI digest. No evaluator, test assertion, numerical gate or timeout change is indicated by this failure. The latest Python 3.12 job ran 16 minutes 56 seconds under a 25-minute limit and failed its numerical-test step normally.

These are PR #247 runs against master. Their primary logs bind actual merge checkouts `6b972d16…` (latest) and `3e03d1e2…` (prior); API top-level head SHAs remain the authority for the requested develop revisions. Nested PR head metadata can advance after an earlier run and is not used to identify historical execution.

My GitHub connection failed, so root supplied immutable primary API snapshots and job logs. This independent review made no repository changes, test or model runs, GPU/queue actions, dispatch or push. Exact primary/source hashes and the proposed validation are in [diagnosis.json](diagnosis.json). Root owns regeneration and the next CI.
