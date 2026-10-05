The revision 5 board at commit
`db5e4ceeb1360034399917ab53103f927661d3d1` rendered 142 unmeasured live
configuration alternatives without registering them as measured evidence.
They have zero attempts, paid time, passes and qualification credit. The same
board also contains five unselected declaration rows retained in its validated
full numerical snapshot. Its selected and measured rows remain registered.

Display refresh originally required every configuration alternative to match
a manifest-selected evidence row, so the byte-identical committed board could
not refresh. [The compact display receipt](legacy-display-roster.json) preserves
the 142 exact declaration identities, the original board's Git commit/blob,
file hash and input digest. It contains no leaderboard or numerical outcomes.
When the original Git object is available, regeneration verifies every identity
against it. Fresh shallow checkouts use the committed receipt. Full snapshot
declarations keep their independently verified snapshot identities.

The exception applies only to exact unselected zero-credit declarations under
the same recorded policy. A changed recipe/source, passing task, execution,
paid cost or qualification flag fails verification. Selected rows, evidence
rows and archived measurements retain their existing strict snapshot checks.
BCAP's new repair alternatives remain a separate display-only verification
snapshot and cannot enter this legacy roster or the incumbent selection.

Reproduce the compact receipt from the exact original Git publication:

```sh
python reports/forge/legacy_display_roster.py \
  --source-commit db5e4ceeb1360034399917ab53103f927661d3d1
```
