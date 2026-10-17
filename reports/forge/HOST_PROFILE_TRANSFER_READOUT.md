# Host-profile transfer: one correctly rejected negative

This readout describes the five completed K3P cells. The later
[existing-control readout](CONTROL_MODE_READOUT.md) adds two mode failures,
bringing the profile to 7/57 measured cells and 50 unknown. All three declared
lineages now fail smoke, making this exact profile unable to meet both the
required positive-reference count and zero false rejections. Stop filling it
solely for adoption; preserve all original baseline measurements below.

The [registered three-cell batch](HOST_PROFILE_TRANSFER_STUDY.md) completed for
**127.210083646 paid seconds**, with no execution errors or remaining reservations.
The explicit image and vector profiles passed. Native grid coverage passed, but
its distribution became too narrow late in training, so its full verdict is FAIL.
The separately registered two-cell smoke completion cost another 29.073682349
seconds. All five cells total **156.283765995 paid seconds**, with three passes,
two scientific failures, no execution errors and no remaining reservations.

| Task | Verdict | Final evidence | Paid seconds |
| --- | --- | --- | ---: |
| bars4 residual16 | PASS | 4/4 modes; HQ 1.0; RMSE 0.003704; 18 final passing checks | 11.170117 |
| unequal mass, published critic | PASS | HQ 0.997559; mass TV 0.054736; SW1/scale 0.079294; minimum mass ratio 0.671387; 9 final passing checks | 24.002935 |
| native named affine grid100 | FAIL | 100 modes; HQ 0.99525; centre RMS 0.118938σ; absolute covariance-trace bias 0.273573; radial KS 0.126870 | 92.037031 |
| mode hold | FAIL | 5/8 modes; HQ 0.998779; none of 24 checks passed | 17.930311 |
| intensity2 residual16 | PASS | 2/2 modes; HQ 0.90625; RMSE 0.025581; mass TV 0; 8 final passing checks | 11.143372 |

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
- [mode hold](attempts/49d41e284fde41698212677403ca5d67/result.json)
- [intensity2](attempts/5693d7b9c35844c78f4440ed15bad4a2/result.json)

Request `89ee9af850e27926df7661de` binds source `5c9c929877c141ccf7352c16987d3a5aadf1aff3a0fedbfa78e7d9b8fe06fdb7`,
K3P revision `5e4a04632539a2ae1fa6d021e54f2ab5172245e7a028c795d45bfff9c2284426`,
and registration published at `b828711d` before execution. Learned-MoG sigma,
explicit image cloud exception, task-owned native initialization, clean live
sampling, and all gates match that registration.

Request `5885f849de55d9603a1bdac7` completed the two smoke cells under
`host-profile-smoke-pair-v1`, published at `745f459c` before execution. Its
registration SHA is `82ddb3d1fd968c79b5aa91a0749ee69e6a2cf105f791b41abe1c62d8fe31f1a0`;
source and candidate match the first batch. Each recorded all intended updates
and 24 checks. Intensity first/stably passed at 425 and was confirmed at 525.
The earlier-source intensity pass remains separate evidence.

At completion of this baseline, the 57-cell matrix had five measured
cells and **52 unknown**, with no receipt issues. K3P's smoke FAIL agrees with
its independent reference FAIL: **one true rejection, zero false accepts out
of one paired negative**. There are no paired positives, so false rejection
remains unmeasured. This selected sample does not estimate population accuracy.
The complete smoke cost is 40.243799201 seconds; reference cost is incomplete
and no complete smoke/reference cost ratio is claimed. Adoption remains BLOCKED.

The [independent first-batch audit](HOST_PROFILE_INDEPENDENT_AUDIT.md) reproduced
all three grades and verified frozen source, 45 native artifacts and checkpoint.
The [smoke audit](HOST_PROFILE_SMOKE_AUDIT.md) reproduced both remaining grades
and the complete calibration reduction. It verified no active baseline work or
reservations; the five-attempt exact K3P revision is now concluded.
Preserve the successful hosts and stop filling quality tasks for this already
negative baseline. No 14k extension is warranted for its failed 7k parent.
Prior work supplies no exact-host learned-MoG positive. A separately registered
one-factor A2-off grid diagnostic can test whether active damping contributes to
the observed late contraction; it must preserve seed, initialization, width,
7k budget and full gates. A pass would justify further qualification, not establish
the full positive lineage. Do not launch a width or seed sweep.
