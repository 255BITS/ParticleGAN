# External Fourier clock scouts

Eight 2k-update scouts, fixed 10k schedule, both GPUs through the shared completion-driven queue. No seed sweeps, full generated training rollouts, clipping, EMA, or auxiliary losses. G reads D-owned M. One particle per real episode and per autonomous trajectory. Default API exact B-cap unchanged.

G receives sin/cos of dyadic frequencies times integer observation index. Six-band default frequencies: .03125, .0625, .125, .25, .5, 1 radians/step. Three-band variant: .125, .25, .5. No raw unbounded time, learned clock, or private recurrent state. Only G receives time unless clock_to_d=true; those scouts condition the paired real/fake scorer on identical time features, and retain candidate-only B-cap.

Training targets use their true sample index. Optional random integer origins in [0,2048] are independent of the real data and shared across each episode's sampled positions; they expose later clock phases without encoding or generating longer histories. Fixed-origin training covers t=0..63; its long evaluation tests temporal extrapolation. Cold evaluation starts at 0; prefix continuation starts at the prefix length.

| Scout | Change |
|---|---|
| clock_static6 | Six bands held constant; architecture-matched control for six-band G-only GRU |
| clock_fourier6 | G clock |
| clock_fourier6_shared | D also observes clock |
| clock_fourier6_offset | Random training origins |
| clock_fourier6_offset_shared | Random origins, D also observes clock |
| clock_fourier3_fast | Three faster bands |
| clock_fourier6_recent | Recent four-point memory + residual G |
| clock_fourier6_offset_recent | Recent/residual + random origins |

Historical dense4 and recent4_delta 2k runs are comparison-only controls. Primary decisions: full cold 256/1024 and original-orbit continuation at prefixes8/32, with direction coverage. Late-only circles are diagnostics. Extend a credible improvement to 5k under the same schedule; do not promote stopping reductions alone.

After completed results, intervene on clock and memory in promising models: preserve startup, then freeze clock or zero/shuffle/freeze memory, measuring full and late metrics plus output sensitivity. Dependence alone does not establish useful reliance; degradation in successful behavior is stronger evidence. If no model succeeds, report that joint beneficial use remains unestablished.

Central tail: `tail -F runs/memory_path/core_round1/train.log`.
