# Fixed-formulation support and resource search

**The best pure resource change passes 4/6 valid data tasks versus 3/6 for the original setup:** 512 particles with batch 128 fixes unequal-width mixtures while preserving broad, anisotropic and spiral cases. It does not solve the unequal-mass or sustained-overlap failures. No shared all-pass data configuration was found.

The original resource grid is 12 shared cards: particle counts 512/1024/2048/4096 crossed with batches 128/256/512. Only particles and batch change in that grid. All loss terms remain Rp logistic, b_cap coefficient 3, kappa 1.25, prior regularization 0.05, no particle L2. The G/D networks, original LRs, Adam (0,.99), prior LR and budgets remain unchanged. Gaussian tasks keep 1200 steps; spiral keeps its original 1600. Evaluation uses live weights, all 24 observations and at least five final passing checks; EMA is separate.

There are 57 newly executed complete episodes and 9 byte-identical reused episodes (six original baselines and three prior 1024/128 hard-task runs). Reuse was accepted only after all relevant numerical source hashes matched. Targets and thresholds never changed; there are no seed sweeps, image reruns or imposed dynamics tests.

## Pure particle/batch results

| Shared particles / batch | Hard sustained /3 | Rare 2% mode | Unequal width | Overlap | Final shortfall | Seconds (3 hard) |
| --- | ---: | --- | --- | --- | ---: | ---: |
| 256 / 128 | 0/3 | FAIL (tail 0/24) | FAIL (tail 0/24) | FAIL (tail 4/24) | 0.2444 | 16.2 |
| 512 / 128 | 1/3 | FAIL (tail 0/24) | PASS (tail 10/24) | FAIL (tail 4/24) | 0.1111 | 18.7 |
| 512 / 512 | 1/3 | FAIL (tail 0/24) | FAIL (tail 3/24) | PASS (tail 11/24) | 0.1111 | 25.9 |
| 512 / 256 | 1/3 | FAIL (tail 0/24) | FAIL (tail 0/24) | PASS (tail 5/24) | 0.1374 | 19.9 |
| 4096 / 512 | 1/3 | FAIL (tail 0/24) | FAIL (tail 0/24) | PASS (tail 21/24) | 0.1881 | 33.3 |
| 2048 / 512 | 1/3 | FAIL (tail 0/24) | FAIL (tail 0/24) | PASS (tail 13/24) | 0.2132 | 32.1 |
| 1024 / 512 | 1/3 | FAIL (tail 0/24) | FAIL (tail 0/24) | PASS (tail 21/24) | 0.2444 | 26.0 |
| 4096 / 256 | 0/3 | FAIL (tail 0/24) | FAIL (tail 0/24) | FAIL (tail 0/24) | 0.1551 | 31.1 |
| 1024 / 256 | 0/3 | FAIL (tail 0/24) | FAIL (tail 0/24) | FAIL (tail 0/24) | 0.1702 | 19.7 |
| 1024 / 128 | 0/3 | FAIL (tail 0/24) | FAIL (tail 0/24) | FAIL (tail 1/24) | 0.2444 | 29.0 |
| 2048 / 256 | 0/3 | FAIL (tail 0/24) | FAIL (tail 0/24) | FAIL (tail 3/24) | 0.2444 | 24.9 |
| 4096 / 128 | 0/3 | FAIL (tail 0/24) | FAIL (tail 0/24) | FAIL (tail 2/24) | 0.2444 | 28.2 |
| 2048 / 128 | 0/3 | FAIL (tail 0/24) | FAIL (tail 0/24) | FAIL (tail 0/24) | 0.2998 | 22.3 |

A final snapshot can pass while its final suffix is too short; that remains FAIL. Selection first uses hard-task sustained count, then average final normalized shortfall, then lower resources. All twelve cards were declared before training. The 1024/128 card and baseline are reused byte-for-byte from prior runs after checking every numerical source hash.

| Shared particles / batch | Unequal-mass mean covariance ≤.85 | Unequal-mass min-eigen ≥.15 | Unequal-width mean covariance ≤.85 | Width min-eigen ≥.15 | Overlap mean error ≤.15 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 / 128 | 5.9630 | 1.0961 | 0.2111 | 0.7359 | 0.0505 |
| 512 / 512 | 3.5779 | 0.4380 | 0.8445 | 0.7383 | 0.0545 |
| 512 / 256 | 0.9129 | 0.7430 | 2.5519 | 0.7996 | 0.0333 |
| 4096 / 512 | 1.6879 | 0.8576 | 4.8602 | 0.8615 | 0.0929 |
| 2048 / 512 | 3.5496 | 0.4600 | 2.1515 | 0.9029 | 0.0418 |
| 1024 / 512 | 4.5910 | 0.4748 | 6.0859 | 0.9270 | 0.0120 |
| 4096 / 256 | 1.1689 | 0.9421 | 10.9373 | 0.8805 | 0.1513 |
| 1024 / 256 | 1.2500 | 0.1664 | 3.1546 | 0.8894 | 0.1645 |
| 1024 / 128 | 4.5577 | 0.7586 | 3.1725 | 0.8509 | 0.1182 |
| 2048 / 256 | 13.6166 | 1.0138 | 10.0910 | 0.9575 | 0.0112 |
| 4096 / 128 | 5.0256 | 0.7637 | 6.4972 | 0.6958 | 0.0620 |
| 2048 / 128 | 11.3110 | 0.9625 | 13.3475 | 1.1063 | 0.2202 |

## All six valid-data tasks

| Shared card | Sustained /6 | vector_two_broad | vector_unequal_mass | vector_unequal_width | vector_anisotropic | vector_overlap | vector_spiral |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| baseline_p256_b128 | 3/6 | PASS (tail 19/24) | FAIL (tail 0/24) | FAIL (tail 0/24) | PASS (tail 15/24) | FAIL (tail 4/24) | PASS (tail 23/24) |
| p512_b128 | 4/6 | PASS (tail 21/24) | FAIL (tail 0/24) | PASS (tail 10/24) | PASS (tail 13/24) | FAIL (tail 4/24) | PASS (tail 22/24) |
| p512_b512 | 3/6 | PASS (tail 20/24) | FAIL (tail 0/24) | FAIL (tail 3/24) | FAIL (tail 0/24) | PASS (tail 11/24) | PASS (tail 23/24) |

Covariance columns report the **mean of component-relative covariance errors** for the named dataset, not only the rare component. Some failures are caused by outliers in a more common component. No resource card sustains the unequal-mass task; the closest final mean covariance error is 0.9129 (512/256) against the unchanged ≤0.85 gate. More particles are not a monotonic fix.

## Separate optimizer and architecture combinations

After the resource grid was frozen, parent-requested combinations were declared as separate stages. The coordinated recipe uses G LR 0.00075, D LR 0.0015, prior LR 0.0225 and Adam (0,.999). The plain-beta stage changes only beta2 to .999 plus the stated resource count/batch. The architecture stage keeps original Adam/LRs and uses D width 128, three layers and four Fourier bands. The adversarial/regularization formulation and all gates remain fixed in every stage.

| Shared combination | Hard sustained /3 | Full-data sustained /6 | Unequal mass | Unequal width | Overlap |
| --- | ---: | --- | --- | --- | --- |
| recipe_p512_b128 | 2/3 | 4/6 | FAIL (tail 0) | PASS (tail 15) | PASS (tail 7) |
| recipe_p512_b256 | 1/3 | not promoted | FAIL (tail 0) | FAIL (tail 0) | PASS (tail 21) |
| b999_p512_b128 | 1/3 | not promoted | FAIL (tail 0) | FAIL (tail 0) | PASS (tail 9) |
| b999_p512_b256 | 1/3 | not promoted | FAIL (tail 0) | FAIL (tail 0) | PASS (tail 21) |
| p512_b128_d128_l3_f4 | 1/3 | not promoted | FAIL (tail 0) | FAIL (tail 0) | PASS (tail 7) |

The coordinated 512/128 card fixes unequal width and overlap but fails unequal mass and anisotropic, so it reaches 4/6 rather than improving the full-data count. Its unequal-mass mean covariance error is 3.3056: the rare component error is only 0.5134, while the 13% component error is 10.2514. The plain-beta combinations add no hard-task pass beyond overlap and were not promoted. The final D/resource combination also passes only overlap; unequal-width mean covariance error reaches 33.016. It fails the predeclared ≥2/3 promotion rule. The favorable effects of separate changes do not add reliably.

## Cost, validation and reproduction

New episode wall time totals 481.6s, with zero numerical errors. These are single CPU observations under shared load, not replicated speed comparisons. The particle sweep uses 2×–16× the original trainable support rows; batch 256/512 processes 2×/4× as many samples per update at the same update budget. Validation retains each original task budget.

All 57 new episode hashes and all source bundles were checked; each recorded verdict was recomputed from its complete live curve and unchanged thresholds. No repository code was edited: the existing `benchmarks.transfer_suite.solvability_search` runner at commit `a11c5304cde01c7fdc96e8a49a5a576b8cb8ebff` performed every new run. No new behavioral tests were necessary for this data-only research run.

Reproduce any stage by extracting its plan and using a fresh output directory:

```bash
python -u -m benchmarks.transfer_suite.solvability_search \
  --plan plan.json --output /tmp/new-valid-support-run
```

Every stage directory contains its exact plan, runtime/source protocol, source.tar.gz, per-episode curves/actions/live/EMA metrics and index. JSON payloads are gzip-compressed with original-byte SHA256s in archive_manifest.json. Original failures remain included. Reproduction scripts and logs are retained. All cases are inspected development data; no universal or production default is established.
