# Exact initialization-only source ports

The 46 experimental API configurations are retained here as complete compressed
source ports. Four additional bundles preserve preliminary clean-merge outputs;
their names end in `-unreviewed` and they are not separate candidates. The
provenance bundle retains reviewed resolutions, rejected merge attempts and the
unexecuted RP1 eager compatibility proposal.

Each primary bundle contains the original and merged package sources, raw merge
inputs, complete final package ZIP, exact tool version, initialization diff,
source declaration and port manifest. `index.json` binds every archive. Compact
storage keeps repeated generated source copies out of the review diff.

Restore one candidate before using the declared commands:

```sh
python3 reports/toy100/deterministic-init-retest/restore_port_sources.py --candidate api-dv16
```

Omit `--candidate` to restore all; add `--with-provenance` for the supplemental
resolution and rejection sources. Restoration verifies hashes and refuses to
overwrite changed files. Existing identical execution directories are preserved.

The adjacent CPU and independent source audits bind 45 runnable configurations.
RP1's earlier eager diagnostic still needs its exact external optimizer setup
integrated and reviewed. Source readiness and CPU construction checks are not
quality results; measured reruns are in `../screen-results.json`.
