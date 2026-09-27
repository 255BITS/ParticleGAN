# PR155 resumed new-initialization results

The 22 research screens marked **UNTESTED** in PR155 have now run with the merged `batch_feature_zero` network and prior initialization. Each kept its original learner, rates, noise policy, frozen 1,200-update host, 24 observations and sample stream. No seed sweep or mechanism tuning was used. Both RTX A6000 cards ran one case at a time.

**Result:** all 22 failed the strict quick screen with **0/24 passing observations**. Eleven ended at seven of eight modes; a high HQ score over the represented modes does not satisfy eight-mode coverage. This closes the 22-case `UNTESTED` list. The full [research leaderboard](research-leaderboard.md) now contains 97 scored configurations: 4 quick-screen passes and 93 failures. A research-host pass does not establish public API qualification.

| Previously untested case | Checks | Final modes | Final HQ | Evidence |
|---|---:|---:|---:|---|
| `di1-real-batch-innovation-63a67cee` | 0/24 | 7/8 | 99.98% | [source, run and audit](research-evidence/research-di1-real-batch-innovation-63a67cee-new-init/archive-manifest.json) |
| `di3-latch-then-anchor-snap-4e7feaf7` | 0/24 | 7/8 | 99.98% | [source, run and audit](research-evidence/research-di3-latch-then-anchor-snap-4e7feaf7-new-init/archive-manifest.json) |
| `ep2-8fe3d1db` | 0/24 | 7/8 | 99.98% | [source, run and audit](research-evidence/research-ep2-8fe3d1db-new-init/archive-manifest.json) |
| `px1-gap-contraction-b4b92eb2` | 0/24 | 7/8 | 99.98% | [source, run and audit](research-evidence/research-px1-gap-contraction-b4b92eb2-new-init/archive-manifest.json) |
| `px2-proportional-gap-6c17331e` | 0/24 | 7/8 | 99.98% | [source, run and audit](research-evidence/research-px2-proportional-gap-6c17331e-new-init/archive-manifest.json) |
| `rr1-innovation-reference-2c71f674` | 0/24 | 7/8 | 99.98% | [source, run and audit](research-evidence/research-rr1-innovation-reference-2c71f674-new-init/archive-manifest.json) |
| `rr2-prox-reference-51ecd753` | 0/24 | 7/8 | 99.98% | [source, run and audit](research-evidence/research-rr2-prox-reference-51ecd753-new-init/archive-manifest.json) |
| `rr3-prox-release-f0730455` | 0/24 | 7/8 | 99.98% | [source, run and audit](research-evidence/research-rr3-prox-release-f0730455-new-init/archive-manifest.json) |
| `cc1-halfsplit-coherence-894a403d` | 0/24 | 7/8 | 99.85% | [source, run and audit](research-evidence/research-cc1-halfsplit-coherence-894a403d-new-init/archive-manifest.json) |
| `td1-kernel-b6f35894` | 0/24 | 7/8 | 91.60% | [source, run and audit](research-evidence/research-td1-kernel-b6f35894-new-init/archive-manifest.json) |
| `pn1-motion-noise-814d29bb` | 0/24 | 7/8 | 91.48% | [source, run and audit](research-evidence/research-pn1-motion-noise-814d29bb-new-init/archive-manifest.json) |
| `ns3-fixed-schedule-shock-32b0ade5` | 0/24 | 6/8 | 94.87% | [source, run and audit](research-evidence/research-ns3-fixed-schedule-shock-32b0ade5-new-init/archive-manifest.json) |
| `td2-separate-constraint-a1291463` | 0/24 | 6/8 | 90.82% | [source, run and audit](research-evidence/research-td2-separate-constraint-a1291463-new-init/archive-manifest.json) |
| `pn2-rate-normalized-contraction-06b0914c` | 0/24 | 5/8 | 94.85% | [source, run and audit](research-evidence/research-pn2-rate-normalized-contraction-06b0914c-new-init/archive-manifest.json) |
| `ac1-persistence-549d59e6` | 0/24 | 5/8 | 73.34% | [source, run and audit](research-evidence/research-ac1-persistence-549d59e6-new-init/archive-manifest.json) |
| `cc3-frozen-baseline-quiet-399ef2aa` | 0/24 | 5/8 | 73.27% | [source, run and audit](research-evidence/research-cc3-frozen-baseline-quiet-399ef2aa-new-init/archive-manifest.json) |
| `sn3-settle-cosine-f26f985e` | 0/24 | 4/8 | 99.90% | [source, run and audit](research-evidence/research-sn3-settle-cosine-f26f985e-new-init/archive-manifest.json) |
| `ac3-gap-magnitude-8b2c2556` | 0/24 | 4/8 | 91.65% | [source, run and audit](research-evidence/research-ac3-gap-magnitude-8b2c2556-new-init/archive-manifest.json) |
| `cc2-margin-uncertainty-f8f9b004` | 0/24 | 3/8 | 100.00% | [source, run and audit](research-evidence/research-cc2-margin-uncertainty-f8f9b004-new-init/archive-manifest.json) |
| `ac2-advantage-2486972b` | 0/24 | 3/8 | 91.65% | [source, run and audit](research-evidence/research-ac2-advantage-2486972b-new-init/archive-manifest.json) |
| `pn3-frozen-raw-peak-957bfa9a` | 0/24 | 3/8 | 57.84% | [source, run and audit](research-evidence/research-pn3-frozen-raw-peak-957bfa9a-new-init/archive-manifest.json) |
| `sn1-gap-mobility-dbade7aa` | 0/24 | 0/8 | 0.00% | [source, run and audit](research-evidence/research-sn1-gap-mobility-dbade7aa-new-init/archive-manifest.json) |

The independently checked runtime audits verified original candidate source, the reviewed constructor proof, actual initial CUDA tensors and shared RNG cursor, final Adam clocks, all 24 observations, and the strict score. [Archive verification](archive-integrity.json) passed for all 97 research rows and the API ledger. The raw checkpoint locations remain recorded in each archive manifest.

## SN3 hard hold

SN3 retained its quick-screen PASS (11/24), and its first 1,200 long-run observations match the quick screen exactly. In the uninterrupted long hold it converged at update **1921** and passed **393** subsequent updates. At update **2315**, coverage fell to **7/8** and HQ to **75.15%**. The required hold is **1200** updates, so SN3 is disqualified for continued use. Its old-initialization run had only 156 good hold updates; the new initialization improved retention but did not meet the gate. The run continued for 300 diagnostic updates after failure and finished at eight modes, but the first departure still fails the uninterrupted hold. [Independent hold audit](sn3-long-hold-audit.json) · [raw result, log and source](sn3-long-hold-evidence/archive-manifest.json).

SN3 was not ported to the public `GANTrainer` or sent to broader image/native gates after this hard failure. The prior 49 API quick screens remain 4 PASS and 45 FAIL; all four passing API candidates already failed a separate image or unequal-mass gate. No default is qualified.

The historical [stop receipt](retest-closure/user-stop/stop-receipt.json) and [closeout](closeout.md) describe the earlier pause; this report records the authorized continuation. Another 89 historical definition rows still need reviewed adapters, exact bindings, or separate draft handling. They are outside the 22 prepared cases and retain `NOT_RETESTED` status, not a quality failure.

If research resumes, the most useful next experiment is to diagnose the missing eighth mode in the eleven 7/8 finalists without changing seeds, then screen a source-reviewed mechanism on the same frozen host. Any quick-screen survivor should clear its uninterrupted 1,200-update hold before public API porting or broader gates.
