# Gaussian smoke inventory: exact gaussian-smoke-inventory-v5 readout

This is an explicitly closed partial source cut, not a completed campaign. Missing runnable work and checkpoint limitations remain visible; these gates cannot fill another source cohort.

The frozen roster retains all **52 candidates**, including **23 admitted** recipes and every blocked/refused declaration. Each recipe keeps its own six Tier 1, twenty Tier 2 and two Tier 3 required cells. This report does not select or rank recipes.

**0 whole recipes pass all six Tier 1 gates.** All runnable Tier 1 peers ran: **False**. All newly eligible Tier 2 jobs ran: **True**. Tier 3 is outside this campaign's cap.

The **72 unique certified attempts** cost **785.797 seconds**. All actual workers used CUDA; process-local `cuda:0` can correspond to either physical GPU because the worker limits visible devices. No scientific retries, new seeds or cross-source gate pooling occurred.

Observed required Tier 1 non-passes by task: ae_gan_hold: 2, gaussian1d_smoke: 3, ring16_acquisition: 11, two_pole: 2.

Newly eligible recipes: none.

Executed source `df4539bbdda7bec5d20f95afd0bdb3fc283f41d2`, digest `3bf51eab2a80eef3645ca5c7df9fa0583cc7053332bc94b19ec874e2e19f98a0`, protocol seed 0. Earlier source refusal/error cohorts and their paid cost remain separate; their gates do not fill these rows.

All completed task states have receipt-bound complete-state certificates: **True**. Uncertified completed cells: **0**. This collection checks strict artifact manifests, file hashes and declared state formats from the frozen producer. It does not load models or repeat evaluation.

[Every whole candidate, numerical metric, unknown cell count and eligibility audit](readout.json) · [Compact file receipt](receipt.json). Original request/evidence/result files, stdout, curves and tensors stay in the artifact archive.

Use the single regenerated technique inventory for family selection. A Tier 1 smoke pass establishes acquisition under its declared bounds; continuous stability retains its separate Tier 2 gate. No calibration or default-adoption claim follows.

Reproduce the saved-data collection after the campaign coordinator has exited:

```sh
/usr/bin/python reports/forge/collect_gaussian_smoke_inventory.py --root . \
  --queue-root runs/forge/gaussian-smoke-inventory-v5 \
  --round configs/forge/rounds/gaussian-smoke-inventory-v5.json \
  --source-commit df4539bbdda7bec5d20f95afd0bdb3fc283f41d2 \
  --source-digest 3bf51eab2a80eef3645ca5c7df9fa0583cc7053332bc94b19ec874e2e19f98a0 \
  --archive-receipt reports/forge/gaussian-smoke-inventory/interrupted-v5/archive.json \
  --output runs/software/gaussian-smoke-inventory-v5-readout --allow-interrupted
```

The [root archive receipt](archive.json) retains 2480 original files in `artifacts/gaussian-smoke-inventory-v5-interruption.tar.gz` (105352732 bytes, SHA-256 `9897f0d1f5c873bbc1c78c53fed1ad853aa87757db11806c43504c04205d9d42`). The collector checks the archive bytes and exact attempt cohort; the root receipt supplies its member digest.
