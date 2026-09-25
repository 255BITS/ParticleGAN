# Stress calibration evidence

This bundle preserves 16 fixed-reference development runs and their explicit v2
rescoring. The eight tasks comprise six ranking stresses and two diagnostics.
The existing nine required behavioral tests remain the eligibility blockers;
these stress results add no blockers. Every task keeps its original importance
tier, even when its reference solvability has not been demonstrated.

| Importance | Tasks | Reason |
| --- | --- | --- |
| Ranking | Fast/slow critic, small batch, larger critic, longer horizon, R1+R2 | Ordinary optimizer, capacity, budget and formulation variation |
| Diagnostic | Deliberately weak critic | An artificial architecture bottleneck must not determine selection |
| Diagnostic | Broad overlapping target components | Score the observable distribution; ambiguous component labels are unsuitable requirements |
| Reserved | Alternating critic update cadence | Unseen update dynamics; never evaluated in calibration |

Only the overlapping-data diagnostic demonstrated a sustained reference solution:
both fixed schedules passed a suffix of 23/24 observations and confirmed at update
300/1,200. Reference solvability remains **not demonstrated** for all six ranking
stresses and the weak-critic diagnostic. This does not establish impossibility.
All runs use seed 0, 24 fixed live observations, and a required passing suffix of
at least five. EMA remains separate. Total recorded episode time is 94.44 CPU
seconds; timings are individual observations on a shared host.

## Evidence and correction history

- [Original v1 results](v1/results.json.gz), [executed specs](v1/frozen_specs.json.gz),
  [source manifest](v1/source_manifest.json.gz), [exact source archive](v1/source.tar.gz),
  [reference runner](v1/run_references.py), [original run log](v1/run.log).
- [Explicit v2 rescoring](v2/results.json.gz), [corrected frozen specs](v2/frozen_specs.json.gz),
  [source manifest](v2/source_manifest.json.gz), [exact source archive](v2/source.tar.gz),
  [rescoring script](v2/rescore.py).

V1 averaged component covariance errors. A constructed counterexample with six
collapsed components and two healthy components passes that bound with error
.75, perfect HQ and correct occupancy. V2 adds
`component_min_eigen_ratio >= .15` to every identifiable stress case. The
counterexample now fails with minimum eigenvalue ratio zero. Nonidentifiable
overlapping data retains its global-distance-only gate.

The minimum eigenvalue metric was already present in every identifiable v1
observation. V2 therefore re-scores the recorded curves without retraining. All
16 final verdicts and convergence summaries remain unchanged. The original raw
JSON and its scoring source are preserved; tiers, budgets, architectures, data,
and the existing required gates were not changed. No learned-policy fitting or
reserved-family evaluation occurred in these calibration artifacts.

Each source archive contains the exact 25 files listed in its manifest. V1 and
v2 scripts preserve their original bytes and historical local paths; rerunning
them requires the matching repository checkout and environment. The v2 script
had no original disk log; its structured results retain the complete rescoring.

## Integrity

[inventory.json](inventory.json) records original/uncompressed and packaged
SHA-256 hashes, sizes, original paths, and each archived source member's hash.
[SHA256SUMS](SHA256SUMS) covers every packaged artifact except itself.
[verification.json](verification.json) records successful byte-for-byte gzip
round trips, both source manifests, and all 50 archived source members.

Run the relocation-safe verifier from this directory:

```sh
python verify.py
```

The original v1 results hash is
`df1cc9cc06c1dcef55ef398453d748a0f7eb6687b94728bb87c756539fd909d2`.
The explicitly rescored v2 results hash is
`666eb0c7f66759144b688531d7d0fd234c12b8eeaa95f3995bb427575f58bad8`.
