# Independent first host-profile diagnostic audit

Source `5c9c929877c141ccf7352c16987d3a5aadf1aff3a0fedbfa78e7d9b8fe06fdb7`; registration `host-profile-first-cells-v1`; candidate K3P revision `5e4a04632539a2ae1fa6d021e54f2ab5172245e7a028c795d45bfff9c2284426`.

| Cell | Verdict | Charged seconds | Actual A2 applications | Main result |
|---|---:|---:|---:|---|
| img_bars4_residual16 | PASS | 11.170117 | 0 | 4/4 modes, HQ 1.0, RMSE .003704, mass TV .125 |
| vector_unequal_mass_published | PASS | 24.002935 | 1,199 | HQ .997559, mass TV .054736, SW1 .079294, minimum mass ratio .671387 |
| grid100_affine_square_named_v1 | FAIL | 92.037031 | 6,999 | 100/100 modes, HQ .99525, but terminal and holdout shape accuracy fail |

Total charged time is **127.210084 seconds**, exactly matching the canonical campaign ledger, with zero reserved seconds. These are measured supervised attempt costs, not FLOPs or exclusively training time.

Integrity checks: verified all 1,029 frozen source files, immutable diagnostic registration, job scientific compatibility hashes, durable result certificates and raw-result grading hashes. Recomputed all three exact recorded grades using the frozen source in read-only mode. Verified all 45 native artifacts (9,204,869 bytes), checkpoint file/content hashes, 7,000 optimizer updates and consistent trainer/named RNG checkpoint states. Canonical queue state and all nine durable receipt files remained byte-identical during the audit. No concrete integrity discrepancy found.

The image explicitly uses the declared sigma-zero learned-cloud exception and clean center enumeration. Vector and native use learned MoG locations, fixed positive sigma .025, no standardization, and clean sampling that retains prior kernel noise. Image/vector initializers remain deterministic orthogonal with named parameter streams. Native uses task-owned identity linear G, Xavier/zero-bias D, and Uniform[-5,5] prior locations; the checkpoint initialization receipt exactly matches the saved raw receipt. All three report finite guards and zero unintended RNG deviations. Image A2 was requested but never eligible/applied; its passing synthetic component probe is explicitly not training evidence.

The grid acquired full accuracy at steps 1,750–2,750 and again 5,250–5,750, then failed all five terminal checks. Live covariance-trace bias contracted from -.132836 at 6,000 to -.273573 at 7,000 while HQ rose. Independent 100k holdout confirms underdispersion: bias -.265594, radial KS .120006 (limit .040), center RMS .089885 sigma. This supports a late quality-retention failure on this cell, not a failure to acquire coverage or a substitute claim based on HQ alone. EMA also fails and remains diagnostic.

These are registered diagnostic cells, not candidate qualification or adoption. Preserve the two successful host results; the smallest relevant native investigation should address loss of shape quality after acquisition using this saved curve/checkpoint before authorizing further compute. No run was launched, no production file was edited, and no readout/conclusion command was issued; candidate remains available for the declared next-batch decision.
