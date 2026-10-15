# Independent audit of the initial B/T/W/S campaign

Completed 2026-10-15. The baseline is the only arm passing both full diagnostic gates. Neither history arm is a survivor. This is a completed negative experiment, not a partial success or an ordinary Tier1 qualification.

| Arm | Two-pole gate / terminal passes | Ring16 gate / terminal passes | Task execution cost |
|---|---|---|---|
| B | PASS / 17/24 | PASS / 26/96 | 114.801372 s |
| T | FAIL / 1/24 | PASS / 6/96 | 62.579858 s |
| W | FAIL / 1/24 | FAIL / 0/96 | 49.721141 s |
| S | FAIL / 1/24 | FAIL / 0/96 | 54.233658 s |

B is the current BCAP/DualNorm/direction-blend baseline; T adds global and local transport with weights1/1; W additionally attenuates the whole generator gradient from its history; S attenuates only the adversarial component from its own history.

All four two-pole endpoints individually satisfy mean_abs≥0.3 and grad_med≤1. The transport/history arms fail because their final passing suffix contains only one observation, below the required five. Baseline ends with17. On Ring16 the final covariance errors are B0.496567, T0.673336, W1.941778, S0.958363 (maximum0.85); final high-quality fractions are B0.960938, T0.912598, W0.880371, S0.770020 (minimum0.85). Ring16 requires all six bounds simultaneously and at least five final consecutive passes. W has zero passing observations; S has six transient passes but none at the endpoint.

Verification: all eight jobs completed their original80/1600 updates and exact24/96 observation schedules, with finite guards, hooks exercised, and zero unintended RNG deviations. The audit independently recomputed each full gate from the raw curve and checked each grader’s canonical raw hash. Source39fad3b03e5178bcdc68cc17ba012ac4083d5887147bef91f0bbeee1205baea6 has1224/1224 verified files, origin071a8d5614caf632b11ead731a6bb5fdd8a2afc8. All arms share runtime, compute profile, protocol, seed0, and named RNG configuration. Task declarations match after removing ownership annotations. Ring16 initial discriminator, generator and learned-prior tensor hashes match exactly; two-pole records an empty initialization receipt, so its identical fixed stored critic/zero-particle construction is supported by the shared task/source rather than claimed tensor hashes. Final named-stream states match across arms for each task. There are eight unique attempts and no scientific retry.

History checkpoint audit verifies the actual generator-only consumer, before normalization, with80/1600 packets consumed and pending=false. W Ring16 records1122 zero-rho steps and ends at rho0; S records116 and ends at rho0.0624252. These are estimator statistics, not a claim that every complete optimizer update was zero. On two-pole, W ends at rho0.711722 and S at0.997980. W’s final two-pole metrics nearly equal T’s, consistent with scalar scaling largely canceled by normalization.

Declared campaign budget:2400 s. Authoritative task execution costs sum to281.336028858001 s; current outstanding reservations:0. B Ring16’s charged108.680519 s includes the documented supervisor/compilation stall, while its adapter reports33.513416 s. The initial coordinator was interrupted only after B scientific workers terminated; the same queue continued under public Queue(on_completion=None), without retry or scientific source change. These costs should not be interpreted as an algorithm-only speed comparison.

All eight saved-output artifacts and provenance checkpoint hashes verify. All eight exported GIF hashes and their source tensor hashes verify against media/index.json. GIFs each show nine selected original scored observations, drawn from24 pole or96 Ring observations; receipts record zero added updates or sampling draws. These are explanatory artifacts and never qualification inputs. Root performs the visual review.

Reproducible audit: independent_result_audit.py (CPU tensor reads only; no model evaluation, gradients, updates or new sampling). Full raw metric/cost/identity rows and exact receipt hashes: independent-result-audit.json. Original attempt receipts remain unchanged under queue/bcap-selective-history-pole-ring-v2.

# Next-stage recommendation frozen before repair results

The selected repair comparison is primary `bcap-transport-penalty2-v1` (T plus reg_coeff2, history none) and secondary S with lr 0.012→0.006 and d_lr_mult 1.5→3 (halving generator/encoder/prior role rates while retaining discriminator rate 0.018 and transport weights 1/1). The withdrawn S+penalty2 draft is excluded. Each repair must pass BOTH complete original gates before any ordinary Tier1 admission. A final metric pass or a transient run of good observations is insufficient. The bounded repair screen has its own1200 s cap and must remain a separate cohort.

If both repairs survive, compare B plus both through the full ordinary Tier1 suite, rather than continue optimizing on the same two diagnostic tasks. If one survives, compare B plus that repair. If none survive, retain the negative findings and do not promote either repair. Ordinary Tier1 requires all six required tasks to pass their exact compatible live acquisition or sustained grading; no repair can advance to Tier2 without its own 6/6 ordinary passes.

| Ordinary Tier1 task | Original updates | Timeout budget |
|---|---:|---:|
| gaussian1d_smoke |1000|120 s|
| two_pole |80|300 s|
| unused_token_hold |200|300 s|
| ae_gan_hold |250|300 s|
| ring16_acquisition |1600|300 s|
| five_word_joint_smoke |20001|900 s|
| clockfree_audit_measurement_v1 (diagnostic) |24|300 s|

Required tasks total2220 s per arm; adding the clock diagnostic makes2520 s per arm. B plus one eligible repair requires a new5040 s cap; B plus two requires7560 s. The clock result does not alter the six-task scientific eligibility requirement. These are proposed next-stage reservations, not costs already spent. Root decides admission under the user’s broad execution authorization; this auditor never enqueues.

Reuse rule: public planning keys science by candidate revision (which binds source), execution/evaluation fingerprints, protocol/seed/RNG, initializer, runtime and compute, plus dependencies where applicable. Campaign identity alone does not prevent reuse. Terminal compatible records can be reused as their actual outcome; a prior failure cannot become a pass. Crucially research-diagnostic jobs carry `science.evidence_use=research_diagnostic` whereas ordinary qualification jobs do not, so these diagnostic pole/Ring receipts cannot satisfy ordinary Tier1 even for an identical recipe. Use the public planner’s exact compatibility decision and preserve original receipts. Source: experiments/forge/planning.py:239–276,325–337; configs/forge/views/bcap-selective-history-pole-ring-diagnostic-v1.json; configs/forge/views/discriminator_stability.json revision9.
