# After two Gibbs-inspired rounds

The requested two adaptive rounds are complete: twelve fresh 2k scouts, four
4-step GPU smokes, 99 focused tests, and completed-model information, process,
and transition diagnostics. Both GPUs used, zero failures, no qualifying
extensions, nothing running or queued. Changes are uncommitted and unpushed.

Read assessment.md, two_rounds.md, comparison.md, and the round15 assessment.
Plans record how round16 followed completed round15 results. Stable log:
`tail -F runs/memory_path/core_round1/train.log`.

## Results

Winner unchanged: round12 match_shuffle25_5k is the nominal reference; saved2k
has stronger late Q. All full cold/warm256/1024 passes remain0/128. Saved2k
minimum warm Q is .010901. Best new: uncond_w10 .009286 (round15), read_w01
.009212 (round16), joint_match_g10 .009033. No gate relaxation or extensions.

Strong writer alignment improves clean information but harms G reads and
continuation. Weak read-space writer weight .01 helps relative to G-only
(.006940 -> .009212); weight .10 regresses to .004944. History negatives achieve
100% wrong-history ranking without solving continuation. All scouts lose tested
process decodability by128 generated writes; M+z does not rescue it. Neither
classifier accuracy nor clean information establishes successful continuation.

## Implementation

New helper: experiments/memory_transition.py, integrated into
memory_handoff_scout.py. Configuration fields:

- transition_weight, transition_writer_weight
- transition_condition, transition_include_x, transition_width
- transition_ramp_steps, transition_critic_only
- transition_space (memory/read), transition_mismatch_weight

All default off. K has a separate checkpoint and optimizer; public exact B-cap
acts in the actual candidate coordinates. K training detaches G/W branches.
G alignment freezes K/W. Writer alignment trains only the fake write, with
detached real reference, anchor and proposal. Read-space alignment freezes G
parameters and detaches z for successor reads while retaining the memory-input
derivative. No second generated write or persistent generated training state.
Runtime G/D loop unchanged.

New evaluation script: experiments/diagnose_memory_transition.py. It requires
completed runs and measures next-read behavior, raw memory gap, K hybrid,
anchor and history-negative tasks. Absolute anchor scores are confounded by
RpGAN offsets; use the paired-margin comparison. Existing information probes
provide full depth0/1/8/32/128 controls. Regression is evaluation-only.

Pipeline helper: reports/memory-handoff/gibbs_round15/run_round.py accepts
--round and --diagnostics. Do not rerun sealed queues. assess.py applies fixed
gates; audit.py checks archived sources and common panels. Sources were frozen
within each round. K initialization initially altered CUDA RNG; fixed before
all launches and regression-tested. Preserve unrelated .claude/, results/motion/
and sparse-ucd.log.

## Proposed next investigation, not queued

Measure reads before and after one actual D optimizer update on fixed held-out
histories, particles, clock and G. Compare cloned baseline versus alignment
updates using the same batch, then measure ordinary G catch-up. This separates
optimizer-time representation drift from static closed-loop dynamics. Do not
claim drift is already demonstrated. Check earlier frozen/slow-writer trials
before timescale scouts. If drift is small, prioritize analysis of the composite
map and selective preservation of process directions. Avoid another K-weight
sweep without a more discriminating hypothesis.

Constraints remain: no MSE training, full generated training rollouts, analytic
cursor, clipping, EMA, B-cap overrides or seed sweeps. D-owned memory, fixed
particle, expert-free runtime, metrics after completion. User authorized
subagents for this task. Commit/push was not requested in this turn.
