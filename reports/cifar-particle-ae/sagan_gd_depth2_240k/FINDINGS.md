# Attention depth result

At 240k, expanded minus unchanged FID50k: -0.1953. Best expanded beats the 200k parent: True.

This is a joint G/D capacity intervention from the same full-state checkpoint. Existing weights, optimizer moments, EMA, prior and RNG were restored; new attention blocks began as identities with fresh Adam state. It does not separate G and D contributions or establish how the architecture would perform from scratch. The unchanged control was user-stopped after 260k; its matched 205k–240k evaluations and source hashes are preserved.

Best expanded FID50k was 12.2464 at210k, improving the parent by0.2881. The gain was not sustained: 215k–235k returned to roughly13.2–13.4, and final240k was12.9330. The matched240k improvement is modest (0.1953). Preserve210k as the best checkpoint; review its samples and coverage before choosing another continuation. This run supports a better sampled checkpoint, not a consistent new lower plateau.

No further training queued.
