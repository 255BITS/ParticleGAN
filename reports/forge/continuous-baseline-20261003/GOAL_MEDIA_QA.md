# Retained goal-media visual review

All six moving/native supplements are source-bound and readable. They illustrate the original noisy Atlas19 results without changing their gates or receipts. Both H2 hold movies correctly show a complete named-persistence **FAIL**, including the final instantaneous **PASS**.

| Media | Actual displayed updates | Frames | Retained full result |
| --- | --- | ---: | --- |
| Moving grid100 | 0, 500, 1000, 1500 | 4 | PASS |
| Moving rotated100 | 0, 500, 1000, 1500 | 4 | PASS; 98 terminal modes, original cutoff 95 |
| Moving staggered100 | 0, 500, 1000, 1500 | 4 | PASS |
| Native grid100 | 0, 50, 750, 1750, 2750, 3750, 4750, 5750, 7000 | 9 | Noisy coverage + accuracy PASS; clean both FAIL |
| Native rotated100 | Same nine native updates | 9 | Noisy coverage + accuracy PASS; clean both FAIL |
| Native staggered100 | Same nine native updates | 9 | Noisy coverage + accuracy PASS; clean both FAIL |
| H2 Atlas broad hold | Nine original states through 1200, then 1250, 1300, 1350 | 12 | Named hold FAIL; original PASS/study INCOMPLETE preserved |
| H2 E22 broad hold | Same retained/appended schedule | 12 | Named hold FAIL; original PASS/study INCOMPLETE preserved |

The moving targets turn at updates 501 and 1001. Their plots use 4,096 retained float16 samples and analytically rotated stored centers; the original gates use 20,000 samples. Static native plots overlay 4,096 retained float32 samples with the saved target cloud and fixed whole/central cameras. All 34 native observations, final five 20k checks, and independent 100k holdout remain bound to the original full joint coverage/accuracy verdict. Early missing fidelity statistics are labeled “unavailable.” Clean and EMA diagnostics are kept separate.

Initial, middle/turn, and final frames were visually inspected for each movie. All eight GIF hashes, byte sizes, decoded frame counts, and renderer Git blobs were verified, along with all 219 supplemental raw-artifact identities, both complete execution snapshots, and the H2 original parent receipts/artifacts. The two H2 GIFs are byte-identical; their distinct receipts identify the family. This media identity does not establish complete algorithm or state equivalence. Some compact H2 word/number labels are joined but legible.

The H2 extension passes only three of the five required checks after the first confirmation at update 1100: appended updates 1250 and 1300 fail; 1350 passes. The prominent full hold FAIL remains faithful to that result. Atlas19 now retains 19/19 complete original diagnostic results and metric GIFs; it supplies no ordinary clean-MoG family, default, or speed credit. This review performed zero training updates, model calls, draws, rescoring, or source/raw edits.

[Compact evidence, exact media paths and SHA-256 identities](GOAL_MEDIA_QA.json) binds every record. Moving renderer: `82c85cc3`; native renderer: `e872ea5a`; original Atlas scientific source: `a0d6d89f`; H2 source: `82c85cc3` (historical scientific modules: `8021a1c5`). The exporter controls passed 34 moving and 22 native software tests; no scientific rerun was performed for this review.
