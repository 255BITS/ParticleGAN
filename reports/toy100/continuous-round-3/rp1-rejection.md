# RP1: ring success, rejected by transfer gates

K3P remains selected. RP1 is useful diagnostic evidence, but it is not eligible
for promotion or adoption as the shared search base. The unchanged formulation
fails two original gates. Further RP1 qualification is stopped.

| Measurement | Result |
|---|---|
| Own ring hold and extension | 1200/1200 and 300/300; minimum HQ .96851 and .96802 |
| Canonical live shift | Stationary 5/5, pre-hold 120/120, deadline 81/81; minimum deadline HQ .91382 |
| Frozen control | 0/81; saved reconstruction and later state capture are separate evidence |
| Transfer screen | 8 PASS, 1 FAIL, 10 NOT_RUN; two initial rate-observer errors subsequently repaired |
| img_intensity2 | FAIL: only 3/24 observations pass; no required five consecutive passes |
| Native grid100, seed1234, 7000 updates | Coverage PASS, accuracy FAIL |
| Rotated100 | Interrupted after training; no canonical coverage/accuracy verdict written |
| Staggered100 / extra seeds / long-term stress | NOT_RUN; no further qualification after rejection |

The image gate ends with two modes and HQ 1.0, but only updates 500, 575 and 600
pass. Update 525 collapses to one mode and HQ .53125; update 550 has HQ .875.
Its controller never leaves acquisition: full learning rates and early penalty
remain on for all 600 updates. The quiet counter never arms. A good final image
does not replace the sustained gate.

On native grid100 the anchor starts at update 518 and mixing is fully closed by
908. All five terminal accuracy checks fail the center criterion: center RMS is
.2614–.2891 sigma against a .20 limit. Independent 100000-sample holdout center RMS
is .2860. Coverage passes; mass, covariance and radial holdout limits also pass.
The late rate is near its floor. This result alone cannot establish whether the
cause is early closing, insufficient subsequent motion, or another interaction.
A new mechanism must measure that distinction instead of tuning to the gate.

A closer [offline trace inspection](rp1-native-drift-analysis.json) narrows the
native diagnosis. Live center RMS improves from .33112 at step 1000 to .24078 at
4000, .22061 at 5000 and .20631 at 5500. It has not yet passed .20. The controller
is still at the .01 network floor at the 5650 sample, but by 5700 it raises that
multiplier to .208 on the unchanged task, holds it through 5900, then closes.
Center error rises to .25022 at 5750 and .28688 at 7000. The guard clips at 5672;
these sampled traces do not prove the exact causal trigger or that suppressing
the restart would have passed. They support investigating false restarts instead
of assuming that slow floor updates alone caused the failure.

The short image never closes; the native continues improving at its floor, then
reopens on a stationary target and loses precision. A general training signal
must distinguish acquisition, ordinary fluctuation and real target change. No
new proposal may read these evaluation scores or hard-code these event times.

Source review also corrects a wording error in the frozen RP1 declaration:
its quiet counter is **not consecutive**. A low level increments it, an intermediate
level leaves it unchanged, and a reopen resets it. The original declaration said
“consecutive”; the source and raw results remain preserved unchanged. This is a
documentation correction, not a new formulation or repaired score.

The independent 1800-update horizon audit matches training tensors, optimizer,
controller, EMA, RNG, rates and noise. The raw whole-capture comparison fails only
because evaluation-call counters differ; the raw failure and the explicitly
scoped training-state pass are both preserved. Seven CUDA observer regressions
pass, including identical learner state/RNG and rejection of deliberately wrong
rates. These audits do not cancel the real quality failures.

[Ring/control report with witness limits](completed-curvature-rp1/attempts/k3p_responsive_precision/result.md),
[Transfer report](completed-transfer/attempts/verify_p3_gates/result.md),
[native raw evidence](rp1-rejection-evidence.json),
[independent audit](completed-codex/attempts/k3p_extragradient/result.md),
[observer and prefix artifacts](rp1-audit/manifest.json).
