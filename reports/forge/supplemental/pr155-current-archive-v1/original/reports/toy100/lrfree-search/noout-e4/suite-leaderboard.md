# E4 frozen-host task suite (September 29)

Package digest `f69349eeda9679c6db04711be9eeebe9b42ed994bc544dd6df585be61e0f2188`;
original host `screen.py` digest
`ee8193adbdf09e93511befae7b6491143c26de88612eddf065cbb92eb2153c3c`.
The 21 non-native tasks ran under pool candidate `e4-noout-fullsuite`, config
hash `2f133e1568b05175`, with `eval_output_noise=true`. The three native tasks
use the exact-package direct 7k runs in [native-receipts](native-receipts/).

| Task | Verdict | Passing observations | Note |
|---|---|---:|---|
| mode_hold | PASS | 9/24 | |
| img_intensity2 | FAIL | 2/24 | also failed across the Round 6 lineage |
| img_blobs4 | PASS | 18/24 | |
| img_stripes2 | PASS | 22/24 | |
| img_bars4 | FAIL | 0/24 | also failed across the Round 6 lineage |
| vector_two_broad | PASS | 23/24 | |
| vector_unequal_mass | PASS | 14/24 | |
| vector_unequal_width | PASS | 20/24 | |
| vector_anisotropic | PASS | 22/24 | |
| vector_overlap | PASS | 21/24 | |
| vector_spiral | PASS | 24/24 | |
| two_pole | ERROR | 0 | parity engine refused `_stray_gate` and `_table_tester` hooks |
| trajectory | ERROR | 0 | same parity refusal |
| residual_student | ERROR | 0 | same parity refusal |
| unipolar | ERROR | 0 | same parity refusal |
| ae_gan_hold | ERROR | 0 | same parity refusal |
| cover_leftover | ERROR | 0 | same parity refusal |
| unused_token_hold | ERROR | 0 | same parity refusal |
| mid_scale_identity | ERROR | 0 | same parity refusal |
| grid100 | PASS | 23/34 | live noisy, 7k; 100k holdout PASS |
| rotated100 | PASS | 14/34 | live noisy, 7k; 100k holdout PASS |
| staggered100 | PASS | 18/34 | live noisy, 7k; 100k holdout PASS |
| **22-task total** | **12 PASS / 2 FAIL / 8 ERROR** | | 8 custom tasks are unscored, not model failures |

Supplemental frozen gates outside the 22-task preset:

| Task | Verdict | Passing observations |
|---|---|---:|
| ring_shift | PASS | 355/460 |
| stationary | PASS | 709/750 |

Including these two supplemental gates, the measured matrix is **14 PASS,
2 FAIL, 8 ERROR out of 24**. The original 22-task preset is **12 PASS,
2 FAIL, 8 ERROR**. Exact pool rows and per-task result files are in
[suite-receipts](suite-receipts/).

The native pass count does not make E4 a project solution: it violates A2,
fails the LR ×1.33 robustness check on all three native tasks, and carries
the unresolved base ledger. The custom host errors were left as parity
refusals, per the earlier instruction not to debug those hosts.
