# Gaussian smoke inventory: exact gaussian-smoke-inventory-v6 readout

The frozen roster retains all **52 candidates**, including **23 admitted** recipes and every blocked/refused declaration. Each recipe keeps its own six Tier 1, twenty Tier 2 and two Tier 3 required cells. This report does not select or rank recipes.

**0 whole recipes pass all six Tier 1 gates.** All runnable Tier 1 peers ran: **True**. All newly eligible Tier 2 jobs ran: **True**. Tier 3 is outside this campaign's cap.

The **161 unique certified attempts** cost **5910.478 seconds**. All actual workers used CUDA; process-local `cuda:0` can correspond to either physical GPU because the worker limits visible devices. No scientific retries, new seeds or cross-source gate pooling occurred.

Observed required Tier 1 non-passes by task: ae_gan_hold: 3, five_word_joint_acquisition: 18, gaussian1d_smoke: 5, ring16_acquisition: 22, two_pole: 11, unused_token_hold: 2.

Newly eligible recipes: none.

Executed source `45f056556503341bccf3ade0cd3365c5d0dadb91`, digest `6269a18ac4f82564cb16ba19afa4b3dd2f836a2b4085fbe5aeb81a35a453e895`, protocol seed 0. Earlier source refusal/error cohorts and their paid cost remain separate; their gates do not fill these rows.

All completed task states have receipt-bound complete-state certificates: **True**. Uncertified completed cells: **0**. This collection checks strict artifact manifests, file hashes and declared state formats from the frozen producer. It does not load models or repeat evaluation.

[Every whole candidate, numerical metric, unknown cell count and eligibility audit](readout.json) · [Compact file receipt](receipt.json). Original request/evidence/result files, stdout, curves and tensors stay in the artifact archive.

Use the single regenerated [technique inventory](../../technique-inventory.md) for family selection. Gaussian smoke measures acquisition; the original word task still requires a five-check passing terminal suffix, as discussed in [the findings](NOTES.md). All gates remain unchanged. No calibration or default-adoption claim follows.

Reproduce the saved-data collection after the campaign coordinator has exited:

```sh
python reports/forge/collect_gaussian_smoke_inventory.py --root . \
  --queue-root runs/forge/gaussian-smoke-inventory-v6 \
  --round configs/forge/rounds/gaussian-smoke-inventory-v6.json \
  --source-commit 45f056556503341bccf3ade0cd3365c5d0dadb91 \
  --source-digest 6269a18ac4f82564cb16ba19afa4b3dd2f836a2b4085fbe5aeb81a35a453e895 \
  --archive-receipt reports/forge/gaussian-smoke-inventory/archive-v6.json \
  --output runs/software/gaussian-smoke-inventory-v6-readout
```

The [root archive receipt](../archive-v6.json) retains 3987 original files in `artifacts/gaussian-smoke-inventory-v6-final.tar.gz` (288773439 bytes, SHA-256 `87e445f1b07e760a121056e104d4bbd72fd481a3d5aa31050c702b3c17773ead`). The collector checks the archive bytes and exact attempt cohort; the root receipt supplies its member digest.

[Selected whole-recipe findings](NOTES.md) · [Interrupted V5 evidence](../interrupted-v5/README.md) · [Actual-training GIFs and source receipts](../media-v6/README.md).

[Publication validation](validation.json): 148 metadata-only publication tests pass; Forge validation and compiled-memory checks pass, and historical coverage is 11,776/11,776. The merged implementation PRs separately record 103 CUDA optimizer checks and six CUDA scheduling checks. All 1,199 scientific files still match the executed source; original registered snapshot hashes and pre-run configuration identities are preserved.

The GIFs use certified retained training observations. Export added no inference, sampling draws, training updates or rescoring; numerical verdicts remain unchanged.
