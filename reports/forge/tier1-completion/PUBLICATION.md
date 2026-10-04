# Publishing the completed Tier 1 round

The execution source stays frozen. After the driver writes final `results.json`
and exports `media.json`, stage the publication with that executed Git commit:

```sh
.venv/bin/python -m experiments.forge.tier1_publication stage \
  --source-commit 928b485ffbe6e17d79b41d307ae8b6275489b37a \
  --results-dir reports/forge/tier1-completion
```

The returned directory under ignored `runs/forge` contains independently
reconstructed main and policy reports plus `publication-stage.json`. Staging
does not update the selection or current leaderboard. It refreshes compact
receipt summaries from certified originals.

Staging requires all nine clean configurations to have PASS or FAIL for every
required main Tier 1 question and the declared clock diagnostic, with matching
execution, evaluation and timeout contracts. Atlas and E22 retain exact-source
blocked clean parent rows; their four runnable policy questions must be measured
and their three ownership blockers must remain explicit. Every final PASS/FAIL
must have its exact family/task/attempt GIF, matching GIF hash and provenance
receipt. The clock display summary preserves all four certified digest
comparisons, source hashes and unexplained dependencies, including comparisons
after a gate's first numerical failure. No command trains or samples.

Once the compact readout, media and artifact archive are complete, publish:

```sh
.venv/bin/python -m experiments.forge.tier1_publication publish \
  --source-commit 928b485ffbe6e17d79b41d307ae8b6275489b37a \
  --results-dir reports/forge/tier1-completion
```

Publication stages again before changing anything. It writes the new whole-row
selection pins first and then registers the frozen source with `publish_current`
and policy advancement when needed. This ordering avoids validating new policy
evidence against old selection fingerprints. Rejected registration restores the
original selection and scoped publication inputs. Historical selections remain
intact; numerical FAIL completes a measurement and supplies no PASS credit.

The existing current leaderboard and family pages remain the publication.
`scoped-evidence.json` preserves the separate policy rows, certified final
attempt identities and clock diagnostics. Its registered cells and media appear
on family pages, excluded from parent totals and parent qualification. Earlier
retry receipts remain historical; only the campaign's certified final attempt
supplies the displayed metrics and GIF for a measured cell.
