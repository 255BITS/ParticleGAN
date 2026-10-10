# Gaussian smoke inventory: exact gaussian-smoke-inventory-v4 readout

The frozen roster retains all **52 candidates**, including **23 admitted** recipes and every blocked/refused declaration. Each recipe keeps its own six Tier 1, twenty Tier 2 and two Tier 3 required cells. This report does not select or rank recipes.

**0 whole recipes pass all six Tier 1 gates.** All runnable Tier 1 peers ran: **True**. All newly eligible Tier 2 jobs ran: **True**. Tier 3 is outside this campaign's cap.

The **161 unique certified attempts** cost **6136.618 seconds**. All actual workers used CUDA; process-local `cuda:0` can correspond to either physical GPU because the worker limits visible devices. No scientific retries, new seeds or cross-source gate pooling occurred.

Observed required Tier 1 non-passes by task: ae_gan_hold: 3, five_word_joint_acquisition: 17, gaussian1d_smoke: 4, ring16_acquisition: 23, two_pole: 11, unused_token_hold: 2.

Newly eligible recipes: none.

Executed source `79fdf16d2ed880a9db1873245f150375e3be31b0`, digest `d276c5a7344fab6ec5de7b314d3982af5b0ef8027c8b89c01376ae366844a9cb`, protocol seed 0. Earlier source refusal/error cohorts and their paid cost remain separate; their gates do not fill these rows.

All completed task states have receipt-bound complete-state certificates: **True**. Uncertified completed cells: **0**. This collection checks strict artifact manifests, file hashes and declared state formats from the frozen producer. It does not load models or repeat evaluation.

[Every whole candidate, numerical metric, unknown cell count and eligibility audit](readout.json) · [Compact file receipt](receipt.json). Original request/evidence/result files, stdout, curves and tensors stay in the artifact archive.

Use the single regenerated technique inventory for family selection. A Tier 1 smoke pass establishes acquisition under its declared bounds; continuous stability retains its separate Tier 2 gate. No calibration or default-adoption claim follows.

Reproduce the saved-data collection after the campaign coordinator has exited:

```sh
/usr/bin/python reports/forge/collect_gaussian_smoke_inventory.py --root . \
  --queue-root runs/forge/gaussian-smoke-inventory-v4 \
  --round configs/forge/rounds/gaussian-smoke-inventory-v4.json \
  --source-commit 79fdf16d2ed880a9db1873245f150375e3be31b0 \
  --source-digest d276c5a7344fab6ec5de7b314d3982af5b0ef8027c8b89c01376ae366844a9cb \
  --archive-receipt reports/forge/gaussian-smoke-inventory/archive-v4.json \
  --output runs/software/gaussian-smoke-inventory-v4-readout
```

The [root archive receipt](../archive-v4.json) retains 3981 original files in `artifacts/gaussian-smoke-inventory-v4-final.tar.gz` (289084194 bytes, SHA-256 `1531a5de3ee93e1bb216060eb25967010a2381c76ff0bc873267580851ba5c21`). The collector checks the archive bytes and exact attempt cohort; the root receipt supplies its member digest.
