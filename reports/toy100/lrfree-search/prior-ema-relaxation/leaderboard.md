# Native100 research leaderboard — frozen QR/noisy gates

One fixed seed per task; no seed sweep. A cell is PASS only when the frozen
coverage and accuracy verdicts pass, including all five terminal live checks
and the independent 100,000-sample holdout. `/34` counts passing observations
within that task's trajectory; it is not a replacement for the verdict.

| Formulation | grid100 | rotated100 | staggered100 | Native total | Scope |
|---|---|---|---|---:|---|
| **Prior EMA relaxation, final package** | **PASS 21/34** | **PASS 8/34** | **PASS 19/34** | **3/3** | Trained GAN; 22-task suite 7 PASS, 2 FAIL, 13 ERROR |
| Prior EMA block copies | NOT_RUN | NOT_RUN | FAIL 19/34, 6250 centre .20169σ | 0/1 | Diagnostic candidate; stopped after first native failure |
| Multiscale prior handoff | PASS 21/34 | PASS 11/34 | FAIL 19/34, 3/5 terminal misses | 2/3 | Direct parent candidate |
| Birth/death disabled | NOT_RUN | NOT_RUN | FAIL 1/34, final centre .2151σ | 0/1 | Full ablation of paired transport |
| Muse BD-counter D floor | PASS 22/34 | FAIL 0/34 | PASS 21/34 | 2/3 | Different parent; cumulative counter saturates by step 25 |

The [final verdict receipts](evidence/) share package hash
`64f82d9edba2a1422206b8474867cfdd35a793e42f24727c4e08fb79d0d532fc`.
The direct-parent and ablation [comparison receipts](comparisons/) preserve the
rejected runs. Muse's round-6 result is separate local research evidence and
is not part of this PR archive.

The previously reported row-EM sampler also reached 3/3 by changing the
sampled distribution after training. That result is retained as a diagnostic:
it has different semantics and scales with an output reservoir. The row here
changes the learner's prior during training and is evaluated through the
ordinary native generator path. Neither result qualifies a project default
from native100 alone.
