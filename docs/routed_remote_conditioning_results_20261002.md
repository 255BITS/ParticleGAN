# Fixed global-conditioning campaign: valid negative result

Increasing G's batch from16 to64 did not meet the frozen10% improvement requirement on this deterministic remote-conditioned fixture. All400 public GAN-only updates completed in16.41s with finite state, exact caller/source readback and active native controls. The scientific command returned2: a completed gate failure, without an exception or stall.

| Update | G16 held patch error | G64 held patch error | G64 relative benefit |
|---:|---:|---:|---:|
| 0 | 56.9847031 | 56.9847031 | 0% |
| 100 | 2.9455860 | 2.9461498 | −0.0191% |
| 200 | 1.5986965 | 1.5947275 | +0.2483% |

Both arms passed their initial-error reduction gates at100 and200. G64 failed the10% advantage gate at both checkpoints. Both arms also failed the requirement that200-step error be at most half the fixed marker-blind variance V=.000506165787. These four failures remain unchanged; no seed, teacher, normalization, threshold or budget was selected after the run.

The200-step total error is over3150×V. That fact alone does not isolate a failure to learn the remote marker: error in the paired midpoint or other ordinary output components can dominate total error. A separate zero-update readout of the exact saved200-step outputs gives:

| Arm | Paired midpoint error | Contrast error / V | Contrast alignment α | Predicted contrast power / V |
|---|---:|---:|---:|---:|
| G16 | 1.598190390 | 1.000345118 | .000300642 | .000946467 |
| G64 | 1.594223244 | .996276603 | .004269537 | .004815741 |

Write each pair as midpoint m plus/minus half-contrast h. Its error decomposes exactly as mean((prediction_m-target_m)²) + mean((prediction_h-target_h)²). Here α=mean(prediction_h×target_h)/V. Although midpoint error dominates total error, contrast error remains near the marker-blind reference V and predicted contrast power is tiny. Thus neither arm learned much remote-marker contrast within the fixed200 updates. The separate diagnostic used9 public predictor forwards to reconstruct/restore the frozen fixture and endpoints,0 gradients and0 optimizer steps, completed in1.83s, and failed its own fixed contrast≤.5V criterion. It changed none of the original campaign gates.

Each arm recorded200 KA2 applications and128 dense table rows per step. G16 recorded12 control evaluations,96 probes,6 accepted moves and3 splits; G64 recorded12 evaluations,96 probes,8 moves and4 splits. Move counts are descriptive and do not establish a restructuring advantage.

There is a limited real-task reason to exercise context-varying routing: on one saved Nova→Qwen raw64 cohort, replacing the full key/value bank with its mean code raised clean error from.145525515 to.154615760 (+6.2465%;49/64 rows worse). That two-forward counterfactual establishes useful bank dependence on that cohort. It does not isolate FiLM alone, establish a conditioning or GAN defect, demonstrate a convergence cause, or measure LPIPS improvement.

The raw pooled marker is±.003125: ±.2×4/256. The teacher deliberately reads it with coefficient320, producing±1 before SiLU; the student begins from public orthogonal weights. The clean teacher is representable by the student family, but that does not prove exact representation under native DV12 input noise. The real pretrained prefix can already carry global context, so the toy's direct receptive-field limitation is a chosen diagnostic rather than a demonstrated real bottleneck.

The publication driver reports the same descriptive algebra on its already-existing clean evaluation predictions, with no extra predictor forward and no additional campaign gate. A pure-tensor test verifies that a large shared offset can coexist with perfect contrast, while a repeated midpoint gives contrast error/V=1.

The [compact result](routed_remote_conditioning_results_20261002.json) preserves every fixed gate, calibration, teacher/fixture identity and organic control count. The [guide](routed_remote_conditioning_20261002.md) gives the runnable public command and equations. [The exact executed driver](routed_remote_conditioning_20261002_archive/scientific_driver.py.txt) and [original scientific protocol](routed_remote_conditioning_20261002_archive/scientific_protocol.json) preserves the executed V2 declaration. Publication changes source loading/provenance and adds descriptive algebra only; it does not rerun or replace the scientific result. No bulk checkpoints, per-step logs, host model dependencies or local recovery paths are required to run the public diagnostic.
