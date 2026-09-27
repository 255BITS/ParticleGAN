# RESEARCH-bcap_real_half_fake_one-new-init runtime audit

Retained-artifact audit PASS. Strict quality FAIL: 0/24, first arrival None, final streak 0; final 4/8, HQ 0.403564453125.

Source seal, own CPU proof, raw initial CUDA models/buffers/optimizer parameters, original construction randomness and all2400 rate actions/Adam calls match.

Original custom research loop and ordinary Adam wrapper; not public GANTrainer qualification.

The source performs ordinary lazy Adam initialization under its historical CUDA default-device context. No eager counter injection is present; actual scalar-clock placement was not serialized and is not inferred.

No final checkpoint, final sampling cursor, native step clocks or model/EMA state was retained by these original probes; no fresh-process continuation/replay claim.

Source and captured initial global/shared RNG plus the first17 original constructor random draws are verified. Full random-operation digest/prefix are retained, not independently replayed.

All2400 emitted scheduling actions and ordinary optimizer call counts are checked. No per-update real/latent batch table was emitted.

Inner host diagnostic labels do not override strict eight-mode/HQ0.9 final-five scoring.

Historical eligibility metadata is preserved, but this tested configuration demonstrably schedules its rates/noise against the1200-step host budget; no horizon-free qualification follows. Old quality is not inherited.
