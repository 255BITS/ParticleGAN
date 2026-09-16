# Next session: first principles before choosing more experiments

Latest user direction: commit round11 and compact, then return to theory and
look for explanations/solutions for persistent0/128. The negative GRU result
is useful evidence. Do not automatically launch another sweep or treat state
corruption/recovery as a selected solution.

## Facts to explain

- Clock removed stopping; motion is no longer the main missing behavior.
- Proposal repair remains part of the best recipe.
- Local pair judging improved continuous quality, but every full pass is zero.
- Longer training and some G GRUs improve short-horizon prediction/startup while
  worsening long autonomous fidelity.
- Separate G state did not solve it. GRU16 without D access beats the matched
  with-D model on long Q/radial, but neither beats the old stateless-G winner.
- Outputs respond weakly to the intended radius/speed late in continuation.
  This does not tell us whether information is erased or retained but unused.
- Zeroing or shuffling memory establishes sensitivity, not useful retention.
- Zero passes are accompanied by large continuous errors, not merely a narrowly
  missed threshold. Keep cold generic-circle success separate from warm fidelity.

## Candidate starting point for reasoning, not established diagnosis

Let h contain learned state (M and optional S), and let G produce x from h,z,t.
Runtime composes G with the writers to obtain a closed-loop transition F(h,z,t).
The current training objective mainly constrains outputs at real-history states
and bounded one-generated-write neighborhoods. It does not directly establish
that repeatedly applying F stays within a useful set of states or preserves the
identity of the process encoded by a prefix.

Local training is not inherently incapable of solving this: an accurate
transition with sufficient state and appropriate stability could work for long
sequences. We should identify which condition fails here rather than assume
full-rollout training is necessary or declare the adversarial objective incapable.

An important distinction: globally contracting all memory differences would
also erase legitimate differences between processes and phases. Useful recovery
would correct deviations away from valid dynamics while preserving distinctions
between valid histories. In this toy, radius, center, direction and phase expose
that distinction. They should remain evaluation concepts, not geometry labels
or analytic circle structure injected into a cross-domain training solution.
A second learned memory is subject to the same distinction; ownership alone does
not create the desired dynamics.

Questions to work through:
1. What information must the state preserve for the conditional next-sample
   distribution to describe a coherent process? Does our state/objective capture
   it, and does fixed z maintain coherent choices over time?
2. Does generated feedback erase process information, or does the reader cease
   using it? Could late restoration of real-history M/S distinguish these cases?
   Restoring state also changes its compatibility with the clock and other state;
   diagnostics must control those factors rather than overinterpret one rescue.
3. Is the failure primarily insufficient support for training states, wrong
   closed-loop dynamics near valid states, clock dependence, or optimization?
   These can coexist; do not call memory OOD a proven sole cause.
4. What local adversarial signal could reward both return toward useful states
   and preservation of process distinctions, without an MSE target, a trivial
   collapsed state solution, geometry assumptions, or a generated full rollout?
5. Which small evaluation intervention would most sharply separate competing
   explanations before selecting another configuration sweep?

Potential local state perturbation/recovery and late real-state restoration
remain hypotheses from the round11 assessment, not authorized queued designs.
Earlier slow/fast memory, clock-origin, memory-size, local stability and repair
scouts are documented in NEXT.md; check history before proposing them as new.

Constraints: cross-domain transferable mechanisms; no moving cursor or analytic
circle dynamics; no MSE training objective; no full generated trajectory training.
Existing two-output/one-generated-write branches and real-prefix BPTT are allowed
with their cost stated. D-owned memory remains central; G-owned state is now an
allowed experiment, with trained no-D controls needed to establish added value.
Keep API B-cap defaults, no clipping/EMA or seed sweeps, metrics-based decisions,
completed-only inspection, both-GPU queue and stable central log when runs resume.

Round11 is complete: five2k scouts, no qualified extensions,90 focused tests,
no running jobs. Keep proposal_mixed_pair25 at2k as the baseline. Read assessment.md
and reports/memory-path/NEXT.md before resuming. Commit requested; push not requested.
