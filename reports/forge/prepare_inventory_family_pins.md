# Prepare the revised inventory's family pins

`prepare_inventory_family_pins.py` proposes display metadata from saved,
independently published rows. It never trains, samples, resolves new requests,
regrades tensor evidence, updates the evidence manifest, or writes the current
selection/leaderboard. The coordinator owns those publication steps.

After the complete authorized campaign has finished and its original receipts
are hydrated, independently stage the **actual executed source commit**, read
from the saved requests. Do not use an earlier refused preflight commit.
The staging API accepts an output prefix; the normal publication CLI does not.

```sh
/usr/bin/python - <<'PY'
from pathlib import Path
from reports.forge.regenerate_technique_inventory import regenerate
print(regenerate(Path.cwd(), source_commit="ACTUAL_EXECUTED_COMMIT",
                 execution_backend="cuda",
                 output_prefix="runs/software/inventory-publication/staged"))
PY

/usr/bin/python reports/forge/prepare_inventory_family_pins.py \
  --staged runs/software/inventory-publication/staged.json \
  --round configs/forge/rounds/gaussian-smoke-inventory-v2.json \
  --old-board runs/software/pre-run-publication/technique-inventory.json \
  --old-selection runs/software/pre-run-publication/family-current-v1.json \
  --manifest runs/software/pre-run-publication/manifest.json \
  --output runs/software/inventory-publication/pins
```

Review `pins/audit.json` before copying `pins/selection.json` to
`configs/forge/selections/family-current-v1.json`. The proposed card binds the
current view's fingerprint. Then the existing publication command is:

```sh
/usr/bin/python reports/forge/regenerate_technique_inventory.py \
  --source-commit ACTUAL_EXECUTED_COMMIT --device cuda --advance-policy
```

The helper keeps the round's preselected configuration IDs. For families
without one, it retains the old selected ID when that ID is in the round;
otherwise it uses the registered canonical candidate. It never chooses a
different recipe because that recipe scored better. Every pin binds one exact
scientific row, including all failures, blocked/unknown cells and denominators.

Complete required Tier 1 PASS/FAIL rows use `current_measurement`; an optional
`--configured-standard` labels an all-PASS qualified row `configured_standard`.
Neither setting grants calibration or default adoption. Partial source-bound
rows use the existing `historical_incumbent` display kind with an explicit
current-source/no-complete-measurement reason. A truly unresolved declaration
outside the frozen report's registrable rows is an audited unavailable choice;
the ordinary publisher retains its declaration-only canonical fallback.

Old selected rows must match exactly one receipt-verified registered snapshot.
The helper converts old measurement display pins to `historical_incumbent`
without measurement fields, because the public historical matcher does not
accept those fields. Their scientific hashes and original qualified tiers stay
unchanged. `--advance-policy` preserves their original policy and qualifications
in the evidence archive; the helper never rewrites them.

Run the saved-metadata checks without constructing a neural model:

```sh
/usr/bin/python -m unittest reports.forge.test_prepare_inventory_family_pins
```
