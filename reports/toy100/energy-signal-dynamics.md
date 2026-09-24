# Data-space proposal signal for continuous GAN training

No energy-signal candidate cleared both acquisition and sustained stability.
The experiment used the unchanged fixed-seed mode-hold and trajectory hosts,
constant nominal G/D learning rate .00425 and prior rate .0085, one Adam
moment update per optimizer per outer update, and the production noise clock.
The controller is scratch-only and is **not** a PR #60 production change.

The signal compares the already sampled real batch with clean generated
samples before and after the ordinary joint G/prior Adam proposal. It uses
`2 mean ||real-fake|| - mean ||fake-fake||`; the omitted real-real term is
identical before and after. Two disjoint half-batches must each improve. For
the conditional trajectory host, both samples contain the observed slow arc
alongside the fast output; this compares the joint data distribution without
using the host identity-MSE gate. D always takes its ordinary update. If the
proposal is rejected, G and prior parameters return to their prior values,
while their Adam moments retain the single ordinary update. Rolling back
both roles makes the criterion describe the actual generated sample after
the step. The source records each proposal score, acceptance, nominal rate,
and the exact run config.

Energy distance is a distribution discrepancy, as established by
[Székely and Rizzo (2013)](https://www.sciencedirect.com/science/article/pii/S0378375813000633).
[Cramér GAN](https://arxiv.org/abs/1705.10743) uses a related distance for
generator training. The proposal-acceptance rule here is an experimental
inference from those ideas, not a result proved by either paper. It can
reject useful GAN updates when finite-batch energy and the frozen task's
quality criterion disagree.

| Declared condition | Warm hold, checks passed / 200 | Cold trajectory | Cold mode hold | Decision |
| --- | ---: | --- | --- | --- |
| Ordinary constant observer | 6/200 | — | 0/5 terminal; 7 modes, HQ .3694 | Failing control |
| Marginal-output energy gate | 200/200; 0/200 proposals kept | FAIL, identity MSE .2764; 11/400 kept | Skipped | Reject |
| Conditional joint-data energy gate | 200/200; 0/200 kept | PASS, identity MSE .003682; 13/400 kept | FAIL, 0/5 terminal; 2 modes, HQ .09668; 46/1200 kept | Reject |
| Conditional gate with full/half/quarter/eighth backtracking | 198/200; failures at 1174–1175, minimum HQ .7993; 64/200 kept | Skipped | Skipped | Reject |

The observe-only controller exactly reproduced the known ordinary constant
mode-hold outcome, showing its measurement did not alter training. Only
394/1200 cold and 61/200 warm ordinary proposals improved energy on both
halves. The full-proposal gate protected the passing warm state by declining
every update, but did not acquire the ring from scratch. The conditional
refinement corrected the trajectory signal's missing conditioning and passed
that cold host, yet accepted too few ring proposals to acquire modes.

To distinguish overshoot from an unhelpful direction, a fixed set of rejected
cold-ring proposals was replayed on the same batch at fractions 1/2, 1/4,
and 1/8. Among 19 checked rejections, only 2, 2, and 3 respectively improved
both halves. The diagnostic reproduced the original failed ring run exactly.
Since a few smaller proposals helped, one backtracking rule retained the
largest improving fraction and otherwise rolled back. It passed only 198/200
strict warm checks; its final eight modes/HQ .9958 would hide the two earlier
failures if only the endpoint were inspected. No cold or shifted run followed
that failed gate.

The [fixed-cloud diagnostic](continuous-evidence/energy-signal/objective-conflict.json)
shows why the data-space signal cannot certify the frozen HQ gate. With 8,192
real samples stratified equally across the eight Gaussian modes, twelve
equal-weight generated particles initially at mode centers, and the declared
output noise, moving two duplicated particles off-center improves empirical
energy from 3.95568 to 3.85187 while HQ falls from 1.0 to .87036; all eight
modes remain represented. The actual warm backtracking trace shows the same
kind of conflict at update 1,174: both energy halves improved under an
accepted full proposal (4.17679→4.17277 and 3.98698→3.98080), while HQ fell
from .99902 to .79980. This establishes a mismatch between the acceptance
signal and the required quality measure in this representational setting; it
does not identify which two learned particles moved in that training update.

The [manifest](continuous-evidence/energy-signal/manifest.json) binds nine
compressed raw run archives, including the failing controls, source snapshots
at execution, effective configurations, per-step receipts, frozen-host
episodes, and source hashes. Each archive contains its own `summary.json` and
`declaration.json`; the two trajectory archives also contain the standard
suite source snapshot and full episode. All runs use seed 0, one CPU thread,
PyTorch 2.13.0, and AVX2. These results do not support a continuous-learning
claim or a shift-response test for this controller.
