# Round14: diagnose memory information, then select local scouts

User authorized diagnosis followed by config-driven scouts on both GPUs, with
subagents as useful. All training constraints remain: D-owned memory, fixed
particle per trajectory, no geometry labels/cursor, no MSE GAN objectives,
no full generated training rollouts, unchanged public exact B-cap, no clipping
or EMA, no seed sweeps. Existing configurations/checkpoints remain supported.

## Diagnosis before selecting scouts

Freeze saved round12 match_shuffle25 at2k and5k. Encode a32-point real history,
then inspect memory after0/1/8/32/128 autonomous writes. Use independent diagnostic
train/validation/test episodes sampled within training process support. Probe
radius and signed speed (including sign accuracy), comparing real-history states
at the same depths. Circle labels/regression belong only to evaluation probes.

Fit linear and nonlinear probes with train-only standardization and validation
selection. Include real-trained probes applied to autonomous states, per-domain
probes, M+z and nuisance/chance controls where practical. Poor transfer alone
can indicate representation shift, not information destruction. Failure of a
finite probe is not proof that information is absent. Distinguish clean-state
information limitations from information lost after writes.

Read/use evidence also includes the completed winner audit: saved paths replay
exactly, runtime and training pair agree, but warm radial error grows markedly
within clocks32–63. Existing late state-restoration diagnostics supply clean
versus autonomous memory at identical clock/particle. No runtime expert is
introduced by these evaluation-only interventions.

Select training mechanisms only after reviewing completed diagnosis. If process
information decays or is weak even under real histories, prioritize a predictive
memory incentive. If information remains decodable but unused, prioritize its
readout/use. Record the decision and precise scouts below before launch.

## Fixed training and extension gates

Fresh2k scouts, same10k schedule, both GPUs via existing durable queue. Saved
round12 match_shuffle25 at2k is the matched-step baseline; saved5k is descriptive.
No control retraining. Inspect training results only after each run completes.

At most two exact2k→5k continuations on the same10k schedule. Qualify if BOTH
warm1024 pass fractions improve, OR quality improves at least20% at BOTH
prefixes8/32, late quality is no worse, radial error is at most5% worse and
direction agreement at most2percentage points worse. Both routes also require
cold late stopping no more than1percentage point worse. Rank qualifiers by
minimum warm pass fraction, then minimum warm quality. Do not relax gates.

Full cold/warm pass metrics remain primary. Q is a continuous diagnostic, not a
probability of success. Any extensions are assessed after completion. New
process-retention probes are diagnostic and do not replace the fixed gates.

Stable training log: `tail -F runs/memory_path/core_round1/train.log`.
Diagnosis output: `reports/memory-handoff/information_round14/diagnosis*`.

## Selected after completed diagnosis, before training

Both reference checkpoints lose decodable original-process information across
autonomous writes. For5k, M-only nonlinear radius/speed R² falls from .599/.954
at depth0 to .126/.436 at32 and approximately zero at128. Real-history memory
at128 retains .574/.946. M+z does not rescue the late states; nuisance controls
are at chance, oracle labels calibrate at R²≈1. Early real-to-generated transfer
is worse than fitting a generated-domain probe, indicating representation shift
as well as declining accessible information. See diagnosis report for limits.

Select a D-only future history-ranking objective. Keep winning immediate clean
mismatch weight.25, point feedback and pair GAN unchanged. The added branch
scores actual future observations at offsets4/12 against another episode's
observations at those offsets. It shares the exact candidate head G uses at
offset0, with an explicit horizon projection added to its first preactivation.
The projection is bias-free and zero-initialized; sin/cos-minus-one query features
are exactly zero at offset0. No future G queries or new persistent state.

For sampled target t, retain only4<=t and t+12<64 in BOTH anchor and donor pools.
Donor choice is the existing deterministic other-episode circular shuffle and
does not inspect future targets. Clean M encodes observations0..t-1. Explored M
encodes0..t-2 and writes a blend of observed[t-1] and detached G(z,M,t-1).
Targets x[t+4]/x[t+12] never enter that memory. One proposal/write is reused
across horizons; mixed averages clean/explored LOSSES .5/.5, not memory tensors.
Candidate scores and exact B-cap use identical cached memory and query horizon.

| Run | Future context | Replacement | Added weight | Explored future writer gradient |
|---|---|---:|---:|---|
| future_clean10 | clean | — | .10 | — |
| future_mixed10 | mixed | .25 | .10 | connected |
| future_full10 | mixed | 1 | .10 | connected |
| future_mixed25 | mixed | .25 | .25 | connected |
| future_mixed10_detachwrite | mixed | .25 | .10 | detached |

New future loss/penalty are averaged across horizons and contexts, then combined
as `(existing + weight*future)/(1+weight)`. Existing includes the unchanged
immediate mismatch normalization. G objectives are unchanged. The last control
detaches ONLY the explored new-future context, retaining clean-future and all
original mismatch writer gradients. It tests generated-context writer learning
more specifically than round13's control.

All scouts use bands4, a500-step write-strength ramp,2k updates/10k schedule,
batch128 and the existing1024-step evaluation panel. Clean adds real-prefix
encoding and future scoring; mixed also adds one detached D-phase G call and
one independent write. G calls D/G are4/4(clean) or5/4(mixed), internal reader
calls8/8 or10/8; max sequential generated writes remains1. No additional G loss.

Require bitwise offset0/default-off training equivalence instead of spending a
fresh scout on the mathematically inactive horizon projection. Require gradient,
causality, active B-cap, resume, and two-device smoke checks before launch.
Run existing process/ranking diagnostics after completion, plus the held-out
information probe on the best new scout to assess whether the targeted memory
information improved. Better probe or ranking scores alone cannot qualify an
extension; use the fixed autonomous gates above.
