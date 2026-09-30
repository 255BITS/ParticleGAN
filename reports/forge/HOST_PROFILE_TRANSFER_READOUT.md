# Host-profile transfer: two passes, one late accuracy failure

The [registered three-cell batch](HOST_PROFILE_TRANSFER_STUDY.md) completed for
**127.210083646 paid seconds**, with no execution errors or remaining reservations.
The explicit image and vector profiles passed. Native grid coverage passed, but
its distribution became too narrow late in training, so its full verdict is FAIL.

| Task | Verdict | Final evidence | Paid seconds |
| --- | --- | --- | ---: |
| bars4 residual16 | PASS | 4/4 modes; HQ 1.0; RMSE 0.003704; 18 final passing checks | 11.170117 |
| unequal mass, published critic | PASS | HQ 0.997559; mass TV 0.054736; SW1/scale 0.079294; minimum mass ratio 0.671387; 9 final passing checks | 24.002935 |
| native named affine grid100 | FAIL | 100 modes; HQ 0.99525; centre RMS 0.118938σ; absolute covariance-trace bias 0.273573; radial KS 0.126870 | 92.037031 |

The vector gate used all components, including the rare component; no finite-atom
exemption was applied. Bars4's first/stable pass was step 175, confirmed at 275.
Unequal mass first/stably passed at 800, confirmed at 1,000.

Native live accuracy passed at step 5,750, then failed **every final check from
6,000 through 7,000**. Covariance-trace bias became more negative, reaching
−0.273573 against the absolute limit 0.10; radial KS reached 0.126870 against
0.04. The independent 100k live holdout also failed (bias −0.265594, KS 0.120006).
The EMA holdout failed the same shape limits. The oracle control passed. An
earlier passing snapshot, full mode count, high HQ or EMA substitution therefore
cannot qualify this run. This records a late decline, without assigning a causal
mechanism from a single fixed-seed run.

All three receipts report finite state, complete intended optimizer updates and
**zero unintended RNG deviations**. Actual A2 applications were 0/600 on bars4,
1,199/1,200 on unequal mass and 6,999/7,000 on grid100; enabling a mechanism is
not evidence that it helps. Synchronized optimizer-phase times were 5.425,
17.642 and 73.124 seconds respectively. Peak PyTorch allocated/reserved bytes:
70,475,776/90,177,536; 73,220,608/75,497,472; 125,372,416/192,937,984. FLOP counts
remain unavailable; paid wall time includes setup and evaluation.

Immutable attempts:

- [bars4](attempts/680dcd337be343f69a1922cd3e89750a/result.json)
- [unequal mass](attempts/a1de5d14bd8b4640b95a826209ae3085/result.json)
- [grid100](attempts/dca8c7aa0eca484c9270d07125f7cb0b/result.json)

Request `89ee9af850e27926df7661de` binds source `5c9c929877c141ccf7352c16987d3a5aadf1aff3a0fedbfa78e7d9b8fe06fdb7`,
K3P revision `5e4a04632539a2ae1fa6d021e54f2ab5172245e7a028c795d45bfff9c2284426`,
and registration published at `b828711d` before execution. Learned-MoG sigma,
explicit image cloud exception, task-owned native initialization, clean live
sampling, and all gates match that registration.

The [57-cell matrix](calibration/host-profile-transfer-v1.md) has three measured
cells and 54 unknown cells, with no receipt issues. K3P now has an independent
negative reference. Its smoke predicate is still unknown, so no paired
false-accept/reject rate is available and adoption remains BLOCKED.

Next, complete only K3P's two missing cheap smoke cells under the separately
registered `host-profile-smoke-pair-v1` lane. This produces a useful pairing;
it does not require filling every remaining quality task of an already negative
candidate. Keep missing reference costs explicit. Finding an eligible positive
reference is still required before accepted calibration; inspect prior work
before proposing a substantive next candidate. No 14k extension is warranted
for this failed 7k parent.
