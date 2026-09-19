# User-stopped small deconv continuation

Stopped at logged step86,600 to free GPU1 for SAGAN-style G/D attention. Latest complete saved checkpoint80k; 200k target was not completed.

| Step | FID50k |
|---|---:|
|40000 (parent)|26.2232|
| 50000 | 24.5681 |
| 60000 | 23.5844 |
| 70000 | 22.7604 |
| 80000 | 22.0022 |

The small generator continued improving through80k. No final capacity ceiling was established. Preserve checkpoint80k for future exact continuation. Updates after80k were not checkpointed; do not describe86.6k as a saved or FID-evaluated checkpoint.
