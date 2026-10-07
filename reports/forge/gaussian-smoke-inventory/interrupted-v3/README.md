# Gaussian smoke inventory: exact gaussian-smoke-inventory-v3 readout

This is an explicitly closed partial source cut, not a completed campaign. Missing runnable work and checkpoint limitations remain visible; these gates cannot fill another source cohort.

The frozen roster retains all **52 candidates**, including **23 admitted** recipes and every blocked/refused declaration. Each recipe keeps its own six Tier 1, twenty Tier 2 and two Tier 3 required cells. This report does not select or rank recipes.

**0 whole recipes pass all six Tier 1 gates.** All runnable Tier 1 peers ran: **False**. All newly eligible Tier 2 jobs ran: **True**. Tier 3 is outside this campaign's cap.

The **62 unique certified attempts** cost **2897.206 seconds**. All actual workers used CUDA; process-local `cuda:0` can correspond to either physical GPU because the worker limits visible devices. No scientific retries, new seeds or cross-source gate pooling occurred.

Observed required Tier 1 non-passes by task: ae_gan_hold: 2, five_word_joint_acquisition: 7, gaussian1d_smoke: 1, ring16_acquisition: 9, two_pole: 2.

Newly eligible recipes: none.

Executed source `a4caa21d1039684d23e57be8e133b51b3b120775`, digest `82e38ced4add542a5b6af78be26a5eff1974a380346f2c1e21a84abcc63da05e`, protocol seed 0. Earlier source refusal/error cohorts and their paid cost remain separate; their gates do not fill these rows.

All completed task states have receipt-bound complete-state certificates: **False**. Uncertified completed cells: **45**. This collection checks strict artifact manifests, file hashes and declared state formats from the frozen producer. It does not load models or repeat evaluation.

[Every whole candidate, numerical metric, unknown cell count and eligibility audit](readout.json) · [Compact file receipt](receipt.json). Original request/evidence/result files, stdout, curves and tensors stay in the artifact archive.

Use the single regenerated technique inventory for family selection. A Tier 1 smoke pass establishes acquisition under its declared bounds; continuous stability retains its separate Tier 2 gate. No calibration or default-adoption claim follows.

Reproduce the saved-data collection after the campaign coordinator has exited:

```sh
/usr/bin/python reports/forge/collect_gaussian_smoke_inventory.py --root . \
  --queue-root runs/forge/gaussian-smoke-inventory-v3 \
  --round configs/forge/rounds/gaussian-smoke-inventory-v3.json \
  --source-commit a4caa21d1039684d23e57be8e133b51b3b120775 \
  --source-digest 82e38ced4add542a5b6af78be26a5eff1974a380346f2c1e21a84abcc63da05e \
  --archive-receipt reports/forge/gaussian-smoke-inventory/provenance-interruption-v3.json \
  --output runs/software/gaussian-smoke-inventory-v3-readout --allow-interrupted
```

The [root archive receipt](../provenance-interruption-v3.json) retains 2245 original files in `artifacts/gaussian-smoke-inventory-v3-provenance-interruption.tar.gz` (94909147 bytes, SHA-256 `dd0aaeca0f50fe1590d7d6b7a101d677610ca2966cd6544f2a697e059f1b907a`). The collector checks the archive bytes and exact attempt cohort; the root receipt supplies its member digest.
