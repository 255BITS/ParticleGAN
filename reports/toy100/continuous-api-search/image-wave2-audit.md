Archived and verified ten new public-API image runs: seven passes and three failures. Source/result declarations and original hash receipts are unchanged; all metrics and per-update rate/noise rows are preserved losslessly.

| Candidate | Frozen task | Result | Passing / 24 | Final passing suffix | Final modes | Final HQ |
|---|---|---|---:|---:|---:|---:|
| API-DV6 | img_bars4 | FAIL | 0 | 0 | 3 | 1.00000 |
| API-DV6 | img_blobs4 | FAIL | 0 | 0 | 2 | 0.93750 |
| API-DV6 | img_intensity2 | PASS | 14 | 14 | 2 | 1.00000 |
| API-DV6 | img_stripes2 | PASS | 23 | 23 | 2 | 1.00000 |
| API-RP4 | img_intensity2 | FAIL | 0 | 0 | 2 | 0.84375 |
| API-RP5 | img_bars4 | PASS | 18 | 18 | 4 | 0.96875 |
| API-RP5 | img_blobs4 | PASS | 19 | 19 | 4 | 1.00000 |
| API-RP5 | img_intensity2 | PASS | 6 | 5 | 2 | 1.00000 |
| API-RP5 | img_stripes2 | PASS | 22 | 22 | 2 | 1.00000 |
| API-C7 | img_intensity2 | PASS | 6 | 5 | 2 | 1.00000 |

RP5 passes all four cards with final suffixes 5 (intensity), 22 (stripes), 18 (bars) and 19 (blobs). DV6 passes intensity and stripes; bars and blobs each pass zero observations. Its final bars sample has high-quality mass 1.0 but only three modes above the frozen minimum mass, while blobs retains only two modes. These are measured coverage failures. RP4 fails intensity. C7 passes intensity; its 600-update metric trace exactly matches C6 and precedes the first KA2 blend, so this screen does not establish a benefit from its changed critic memory.

Every frozen spec and all four fixture/metric/observation/noise-offset sources match pinned fa511ce010120b502f494d717d01b14b8551eed8 byte-for-byte, including bars and blobs. Every declared source and original artifact hash verifies. Each image uses the same public package as its ring; recipe differences are only z dimension, particle count and batch. Initial model and global data-RNG hashes match the archived public C6 image host. All 24 observation steps and 600 rate/noise rows are present for each run, and all retained gzip streams decompress to the exact original bytes.

Noise is part of each declared learner: DV6 uses input standard deviation 0 and output standard deviation .029 throughout. RP4, RP5 and C7 use absolute input decay over 360 updates and output warmup over 720. Evaluation applies actual output noise at completed update count, with seed 402 + update + 1901. The final RP/C7 evaluation noise is .0241666667; DV6 is .029. RP5 closes on stripes at 429, bars at 518 and blobs at 372; intensity remains open. No image run supplies reopening evidence.

The raw image card prior_weight=.05 is a learner regularization setting. Historical public comparison replaces it with recipe.prior_reg; pinned public KA2 and these candidates all use 0. Some immutable declarations incorrectly say that no auxiliary loss exists. Preserve those declarations and this clarification; no omitted task-required objective is established.

Each new evidence directory includes archive-manifest.json with retained-file hashes, original-file hashes and external checkpoint hashes. JSONL files are compressed losslessly. Original checkpoint tensors remain at their verified external paths, matching existing evidence conventions. Supplemental assessment files added after execution are preserved and labeled separately. image-wave2-audit.json contains complete entries ready for broader-results.json; shared results, README and state files were not edited.

This was a CPU/stdlib artifact audit without PyTorch import, training, GPU execution or checkpoint deserialization. Sample-level RMSE was not independently regenerated. Image passes do not substitute for the remaining task, stationary, shift and long-run qualification.
