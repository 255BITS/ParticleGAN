# Round18 handoff

Latest user requested separate Mg/Md and internally advanced G state, then asked
about UCD for D clock. Completed seven2k scouts on both GPUs through the durable
queue. All post-training diagnostics completed. No qualifying extensions, no jobs
running/queued. Do not restart sealed state_round18. User requested a checkpoint
commit and is considering shelving the experiment. Pause further experiments until
requested; push was not requested.

Read assessment.md, comparison.md, state_comparison.md and formulation.md.
Current winner remains original round12 match_shuffle25. All full circle passes
remain0/128. Best new embedded8_dclock min warm Q .007584 vs baseline .010901.
D clock improves embedded49%, hybrid54%, intent6%, but worsens shared baseline46%.
Internal intent/hybrid updates lose80–86% Q against matched observation controls.

New Config.g_state_update defaults observation (old behavior); embedded/intent/hybrid
select InternalStateReader in memory_g_recurrent.py. All new modes require g_state_dim
and no D-memory access. Common 8-state GRU consumes encoded observations in prefix;
generated input is encoded point, final decoder64-features, or equal mixture. Same
architecture/initial weights. Point feedback mixes real/generated features uniformly
across modes. Pair branch trains one full generated transition, two outputs. No
full generated rollout/MSE training/clipping/EMA/B-cap override. D clock config existed;
no UCD added. Clock index discrete; time-class prediction lacks point-level
identifiability for random-phase circles.

Information diagnostic now supports recurrent G and --memory Mg (default M keeps
D-memory reports). Both state kinds are probed on the same held-out panels. Mg real
controls retain process info; all generated Mg/Md probes near chance by128. Crucially,
real-trained Mg probes fail after one intent/hybrid transition but re-fitted probes
still recover radius/speed. This supports early representation shift, not immediate
information erasure or proof of GAN decoder failure. D is exactly irrelevant to
separated runtime; Mg interventions affect outputs but zeroing often improves error.

Next possible investigation: saved-model one-write next-read compatibility, then
same internal state transition in teacher encoding and runtime with observations
as corrections. NOT implemented or queued. Don't launch another sweep automatically.
User previously questioned whether to shelve experiment; report negative results plainly.

130 focused tests,18 final transition rechecks,9 deployment/probe rechecks; four GPU
smokes (two final-source), actual Mg diagnostic smoke. Source/panel audits pass.
21.38min queue wall,37.71GPU training minutes. Exact resume and default-off behavior
covered. Source snapshots consistent across seven scouts. Branch feat/sequential-memory-path.
Rounds15/16/17 plus18 included in the checkpoint commit; preserve unrelated
.claude/,results/motion/,sparse-ucd.log.
Stable tail runs/memory_path/core_round1/train.log. No live PTY/wait session remains.
