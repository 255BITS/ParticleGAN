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
python - <<'PY'
from pathlib import Path
from reports.forge.regenerate_technique_inventory import regenerate
print(regenerate(Path.cwd(), source_commit="ACTUAL_EXECUTED_COMMIT",
                 execution_backend="cuda",
                 output_prefix="runs/software/inventory-publication/staged"))
PY

python reports/forge/prepare_inventory_family_pins.py \
  --staged runs/software/inventory-publication/staged.json \
  --round configs/forge/rounds/gaussian-smoke-inventory-v4.json \
  --old-board runs/software/pre-run-publication/technique-inventory.json \
  --old-selection runs/software/pre-run-publication/family-current-v1.json \
  --manifest runs/software/pre-run-publication/manifest.json \
  --output runs/software/inventory-publication/pins
```

Review `pins/audit.json` before copying `pins/selection.json` to
`configs/forge/selections/family-current-v1.json`. The proposed card binds the
current view's fingerprint. Then the existing publication command is:

```sh
python reports/forge/regenerate_technique_inventory.py \
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

For a new executed source under the **same registered view**, explicitly pass
`--refresh-source`. The default still requires a later view revision. Source
refresh requires the identical view revision, policy fingerprint and ordered
tier requirements; it rejects a source commit or digest already represented by
registered evidence. It preserves the round's pre-run candidate choices even if
another recipe scored better. Existing rows remain exact historical display
pins, with their original scientific hashes and recorded qualifications.

The staged publication's saved receipt proofs, original request origins and
frozen source manifests must agree. The helper checks each distinct snapshot's
bytes against its original manifest, without loading models, restoring
checkpoints or grading samples. Keep the saved source snapshots hydrated when
preparing this proposal. The audit identifies `same_view_source_refresh`, binds
every input file and verified source manifest, and records that the evidence
manifest remains unchanged.

After the corrected V6 campaign finishes, the source refresh uses the same
staging procedure above and the unchanged view revision 7:

```sh
python reports/forge/prepare_inventory_family_pins.py \
  --staged runs/software/inventory-publication/staged.json \
  --round configs/forge/rounds/gaussian-smoke-inventory-v6.json \
  --old-board runs/software/pre-run-publication/technique-inventory.json \
  --old-selection runs/software/pre-run-publication/family-current-v1.json \
  --manifest runs/software/pre-run-publication/manifest.json \
  --refresh-source --output runs/software/inventory-publication/pins
```

After reviewing/copying the proposed selection card, register the completed
source with `regenerate_technique_inventory.py --source-commit
ACTUAL_EXECUTED_COMMIT --device cuda`. No view-policy advancement is required.
The existing publisher adds the new same-policy source cohort and retains old
cohorts; source refresh grants no automatic reuse of their task gates. This
helper does not perform either publication step.

When an unavailable choice leaves a family with only declarations, the
publisher can bind its canonical display to the source unanimously named by
the current family pins on that backend. This resolves old/new unrun-source
ambiguity without choosing a substitute trained recipe. It applies only to
rows with no attempts, paid cost, executed outcomes or qualified tiers.
Mixed-source pins and same-source runtime ties remain unresolved; measured
ties retain their original refusal. Excluded declarations stay in immutable
evidence and remain available to exact historical display pins. Cached
regeneration uses the same source consensus and preserves the display.

Run the saved-metadata checks without constructing a neural model:

```sh
python -m unittest reports.forge.test_prepare_inventory_family_pins
```
