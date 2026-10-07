# Saved inventory readout collector

`collect_gaussian_smoke_inventory.py` reads a stopped Forge queue and its original
durable request/evidence/result certificates. It writes a compact report, scalar
readout, file receipt and nonqualifying scientific recall record to an explicit
output directory. It does not run neural models, load checkpoints, grade samples,
modify the queue or regenerate the technique inventory.

The default round is `gaussian-smoke-inventory-v4`. The round/campaign and exact
executed source are parameterized; source identity comes from the saved launch
receipt unless supplied explicitly. The collector binds the round to that Git
commit and verifies the complete frozen source snapshot. Production collection
requires all 52 declarations, including blocked/refused candidates, the frozen
6/20/2 denominators, seed 0, CUDA-only execution and zero scientific retries.

Run after both workers and the coordinator have exited:

```sh
/usr/bin/python reports/forge/collect_gaussian_smoke_inventory.py --root . \
  --queue-root runs/forge/gaussian-smoke-inventory-v4 \
  --round configs/forge/rounds/gaussian-smoke-inventory-v4.json \
  --output runs/software/gaussian-smoke-inventory-v4-readout
```

Active/reserved work is refused before output. Each measured cell must match its
original durable attempt, exact candidate/source/job, saved independent evaluator
certificate and charge. Completed tasks need strict full-state file certificates:
the separate provenance-only descriptor, the existing Gaussian full-context tree,
or the existing clockfree complete branch-state tree. Provenance-only checkpoints
must bind every named stream, add zero updates/draws and remain ineligible for
continuation. This is a saved-byte/metadata audit; tensor contents and complete
state format come from the frozen producer, rather than a new restore test.

The readout retains whole candidates alphabetically, actual final numerical
metrics, unknown cell counts, costs, physical CUDA worker slots, prerequisite
identities, blockers and required-job eligibility. It creates no ranking and
imports no earlier source gates. It explicitly reports missing runnable Tier 1
or newly eligible Tier 2 jobs. The root uses its existing single technique
inventory and the separate family-pin helper for selection.

For a stopped source that lacks complete-state certificates, use
`--allow-interrupted`. This additionally requires its source-bound
`interruption-receipt.json`, zero active workers/reservations and unchanged
original verdicts. The readout has `finalized: false`, preserves each missing
checkpoint limitation and cannot supply another source's qualification. Pass
`--archive-receipt reports/forge/gaussian-smoke-inventory/provenance-interruption-v3.json`
to verify and link the root-owned archive without duplicating its receipt.

After review, the root copies the output's report/readout/receipt into the intended
publication directory and its `records/ROUND-readout.json` into
`reports/forge/records/`. Use `--publication-prefix` if the readout's eventual
publication path differs from `reports/forge/gaussian-smoke-inventory`; this binds
the recall record and relative archive link to the reviewed destination. The
interrupted v3 publication uses `reports/forge/gaussian-smoke-inventory/interrupted-v3`.

Metadata-only checks:

```sh
/usr/bin/python -m unittest reports.forge.test_collect_gaussian_smoke_inventory \
  reports.forge.test_prepare_inventory_family_pins
```

Those fixtures are JSON/plain bytes under temporary directories. They test
roster/certificate/source/device/retry refusals, eligibility, strict file and
named-stream certificates and partial-cut limits without CPU or GPU neural
execution. Bulk original receipts, stdout, curves and tensor states remain in
the local artifact archive.
