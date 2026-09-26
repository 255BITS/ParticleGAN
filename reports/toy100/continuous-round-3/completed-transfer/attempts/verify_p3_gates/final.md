Neither candidate is promoted. K3P stays the selected base.

**RP1 stops on `img_intensity2`.** The final snapshot is 2 modes and HQ 1.0, but only 3 of 24 checks pass and the passing suffix is 2. The gate needs 5 consecutive stable checks. The first pass is at update 500, then update 525 falls to 1 mode and HQ 0.531. The controller never left full rate: generator and critic stayed at 0.00425, the prior at 0.0085, and the mixing weight at 1 for all 600 updates. The observation table is `reports/toy100/verify-rp1-attempt/traces/img_intensity2-observations.csv`.

Passes before that failure: `mode_hold`, `img_stripes2`, `ae_gan_hold`, `cover_leftover`, `img_bars4`, and `img_blobs4`. `vector_unequal_mass` and `vector_unequal_width` first aborted in the unchanged rate check (`lr_g diverged` when the gain moved to 0.99 while the applied learning rate was still 0.00425). Those ERROR rows are kept. Fresh adapter reruns later passed both, and that does not clear the intensity failure. The other 10 transfer gates and all 3 natives are NOT_RUN. No native, stress, or extra-seed work was started.

**P3 passed its four sensitive gates** (`mode_hold`, `vector_unequal_mass`, `vector_unequal_width`, `img_stripes2`), each ending locked at 10% of the initial rate with mixing weight 0. Its preserved shift is still FAIL 77/81, its noise still follows the task horizon, and the other 18 toys are NOT_RUN.

The full gate table, hashes, rates, and replay commands are in `result.md` next to `tests.jsonl`.
