# Round17: opposing inference/generation joints

Figure1 of GibbsNet (https://arxiv.org/pdf/1712.04120) exposes a mismatch in rounds15/16: both our pairs used post-write memories. This round tests actual opposing directions, conditioned on history at one physical time:

```
real joint = (E(M, x_real, t), x_real)
h = tanh(seed_projection(particle))
repeat total_decoder_calls:
    x = P(particle, M, h, t)
    if another decode remains: h = E(M, x, t)
fake joint = (h_that_produced_final_x, final_x)
K sees joint, detached M, and t; never particle.
```

E and P cooperate against K, including gradients through the real E branch. D's persistent writer retains its original objectives. The ephemeral latent resets for every physical event; inner refinement holds M, particle, and clock fixed. Only the final sample enters the existing runtime writer. Early inner iterations are detached; final E-to-P is connected. With one decode, the particle-to-h projection learns; with three it stays at initialization because warmup is detached. This is a conditional deterministic-particle adaptation, not a reproduction of GibbsNet's stochastic stationary chain.

Six scouts, 2000 updates, unchanged 10000-update schedule, latent dimension8, 500-update loss ramp: total decoder calls1 or3, each with joint weight0/.10/.25. Weight0 is a matched architecture control. Each P still uses the winning proposal adapter's two reads, so runtime costs are2 vs6 reads. Original winning 2k/5k checkpoints are reused; no seed repeats.

All original winner losses remain. No MSE objective, clipping, EMA, extra persistent G state, runtime expert, or additional generated temporal writes. Joint K uses public default exact B-cap. The existing maximum one generated temporal write per training branch remains unchanged. Old configurations remain default-off.

Run four-update full-panel GPU smokes, focused tests, then freeze sources and drain both GPUs through memory_dispatch. Stable tail: runs/memory_path/core_round1/train.log. Inspect results only after job completion. Diagnostics run after the whole queue finishes: information retention, temporal process, one-write response, and new opposing-joint/latent/refinement interventions. Diagnostic MSE is evaluation only.

Primary endpoint is full-circle pass count on 128 trajectories at1024 autonomous steps after prefixes8/32, also checking cold rollout. Continuous Q is diagnostic, not success probability. Promotion requires both warm pass counts improve, OR >=20% Q improvement at both prefixes with late Q no worse, radial RMSE <=5% worse, direction <=2 percentage points worse. Both routes require cold late stopping <=1pp worse. Compare against original2k; select at most2 exact-resume5k extensions. Judge loss-specific benefit against its matched architecture control as well. Do not relax gates.
