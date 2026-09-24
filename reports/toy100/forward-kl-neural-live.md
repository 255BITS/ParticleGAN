# Finite-GH9 live neural host — stopped before a ring pass

Host tag: **neural** for the runs below. The likelihood correction is an added density objective (donor search / EM / finite GH9), not a pure GAN-game fix. No production claim. Torch **2.14.0+cu130** on CPU, one thread. Pinned receipts were produced under 2.13.

## Wiring

`forward_kl_neural_live.py` calls the existing `forward_kl_neural_v2` adapter. After each native PR84 D-then-G Adam step the adapter fits a GH9 target from pre-G parameters and keeps it only when the actual finite-GH9 cost falls. D, Adam, RNG, and native gradients are unchanged. `trajectory` does not enable the correction (`task == 'mode_hold'` only).

## Pinned sources (this tree, byte match to neural44 archive)

| Script | SHA-256 |
| --- | --- |
| `python -m reports.toy100.forward_kl_neural_filter` | `b1743328ac5ec1d1131f40c1203d507f2c5baa2e48af3a1e40ec5f1ea731d870` |
| `forward_kl_neural_v2.py` | `44ea3bf3d50e32fa11231d779c6c7888f13dca4246f230f03c797c486b06a9ff` |
| `python -m reports.toy100.forward_kl_gh9_stress` | `adc5d6f6ded8e6cf297ee90aa0d540723becc632a601b93b5b9a4393cd344485` |
| `forward_kl_free_filter.py` | `0cc098d63efb2460dc038b897d5db532b9f87115ffbd5a78e42ee042ed21c873` |
| `forward_kl_gh9_remembered.py` | `86639dfefe0c11f82eccbd8dfe106e5797f36e75cf09a906de1768b02c28e1df` |
| `forward_kl_chunked.py` | `e0977838c7d5ae787f5194e653d1a8a09d5c7c0af7eb5f48b65af49a303b141f` |
| `forward_kl_remembered_donor_rescue.py` | `21828acbd6c823ae53fb0da343746442278eeeb4f68e8ece04e20fe85f346471` |

Capture-v2 SHA `37aa612bd3b3e1867a2d1508674158849ccf5940c907af76b1f8b061f4eb3e47` is the warm sidecar recorded by `sample_anchor_free1200.load_states`. The full capture directory is not in this checkout, so neural44 was not re-executed. Focused tests: `tests/test_forward_kl_neural_v2.py`, `test_forward_kl_gh9_remembered.py`, `test_forward_kl_gh9_stress.py` — 13 passed.

Bank stress re-run (`--first-bank` = archived v2 `result.json`) aborted on the first warm step. Native real-bank SHA-256 matched both `warm1324` and `cold1`. Fresh initial GH9 differed by about `1.1e-6` (warm) and `1.2e-5` (cold), outside the script's `1e-12` control. Archived `all_gates_pass: true` stands. This run does not replace it.

## Gate table

| Gate | Result |
| --- | --- |
| Neural warm 1324–1339, live Adam | **PASS** 16/16, min HQ `.999755859375`, min modes 8, 67.3 s. Every step `FINITE_GH9_FITTED_TARGET`. Fresh history at 1324, not a resumed likelihood memory. |
| Cold trajectory 400 | **PASS**, identity MSE `.000942668`, 18/24 suffix. **0** likelihood corrections. This is the PR84 host. |
| Cold ring | **Not passed.** Stopped at update 33. |
| Stay / own-hold | Not run. |

Cold prefix (diagnostic late-noise `.029`, not the host's post-240 grade): 8 modes from update 1; HQ rose to about `.830` by update 29 and sat at `.827–.830` through update 33. Step time grew from 1.1 s (1 bank) to 16.2 s (33 banks), about 0.5 s per added bank. A 1200-step cumulative ring at that slope is on the order of a day, not a fail-fast gate. Process stopped. Log: the `KL_UPDATE` lines from that run.

## Rank

Below board **#1** (pre-start anchors: neural ring 8/HQ 1, own hold). Below **PR84** as a GAN-dynamics reference: this lane did not finish a neural cold ring. Neural44 and the bank screens stay unranked until a full neural cold pass exists. This run does not create one.

## Keep / kill / next

**Kill** this promotion attempt. Live warm on a borrowed eight-mode state works. Cold trajectory does not exercise the likelihood update. The cold prefix covers eight modes and misses HQ `.9` while the all-history estimator's cost grows with the bank count.

**Keep** the adapter as a saved-window / warm-state filter only.

**Next single bet:** a history bound or incremental quadrature that is the same finite-GH9 acceptance test at bounded cost, then one cold ring. Do not retune memoryless anchors, frozen IDs, outside-only discovery, mean16, or local MMD.
