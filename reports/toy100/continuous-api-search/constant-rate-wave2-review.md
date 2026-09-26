# Review before the next constant-rate attempt

The second attempt has no qualified winner. This review authorizes the next
three-proposal attempt after it exits; keep one worker, the pinned API base,
fixed declared seeds and all COMMON.md requirements.

Predecessor:
`/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T214250Z-1447906/constant_rate_stability/20260926T214250Z-1447914`

Read its final report and `repo/experiments/constant_game/` declarations, source
ZIPs and traces. Do not modify that checkout. Copy reviewed code with hashes/diffs.

## Retained evidence

- C4's fresh predictor/corrector never acquired either target through4600. Its
  predictor movements were much larger than accepted corrections; simply adding
  another Adam moment update repeats an already failed historical idea.
- C5's local secant approximation to an implicit game response acquired at560
  and held185/185; changed-target arrival+270, then194/194. Its implementation
  accidentally copied the critic reference exactly because it ran while critic
  gradients were disabled. These are real scores for the hard-copy version,
  not scores for its declared .99 averaged reference. No longer test was run.
- C6 fixes the reference update and explicitly uses serial backward. Initial
  arrival590,182/182; changed-target arrival+300,191/191. Its frozen image gate
  passes6/24 with final suffix5/5. Its true subprocess1600→1800 continuation and
  horizon-prefix checks pass; independent audits verify source and state.
- C6 nevertheless loses an unchanged target twice in its7500 stationary run:
  failed observations3390–3680 and6780–7010,54 failures total,638/692 since
  arrival, minimumHQ0 and zero modes. The final stable suffix7020–7500 is49
  checks. Long30000, remaining21 tasks and K3P are correctly NOT_RUN afterward.

Every nominal rate stays constant. Better short recovery and a passing image
task do not establish stationary stability. Inspect secant-field estimates,
accepted displacements, critic memory and optimizer state around both collapses.

## Mechanism directions

There is a useful causal question left by the C5 bug: does a deliberately fresh
one-step critic reference avoid the instability introduced by longer memory?
It may be worth a clearly specified successor that implements this behavior
directly, under the corrected serial runtime, and earns its own stationary and
image evidence. Do not implement it indirectly through frozen parameter flags,
rename C5's scores as new passes, or search an EMA coefficient grid.

Alternatively, use the collapse traces to repair the local implicit game model
itself—for example its ability to control a collective excursion—if that is
better supported. Select a coherent mechanism and explain the prediction before
execution. Keep nominal rates constant for this lane. The precision lane will
investigate combining autonomous retention control with the implicit update;
the data-drift lane owns independent real-data evidence. Do not duplicate them.

## Public API regression to repair before new execution

The final predecessor regression catches optimizer post-step hooks firing twice.
Its copied KA2CriticAdam.step calls the wrapped K3PCriticAdam.step when no limiter
is active, invoking both wrappers. The measured source remains unchanged; an
untested proposed-hook-repair.patch and hook-regression.json are saved under
experiments/constant_game. Review and repair this public contract before copying
the optimizer plumbing into a successor; verify ordinary math/counters and hook
count. No-hook quality collapses are not automatically explained by this bug.
Preserve the failed regression and earn new candidate results.

## Reusable verification

Retain the scoped serial_backward mode and exact checkpoint contract. It is an
execution correction, not an automatic quality pass. The independent final
audit and standalone patch are available from the data-drift predecessor.

The corrected generic image gate preserves the frozen plan, residual_upsample16
fixture, actual noise stream and all24/five-suffix scoring. Reuse it with source
receipts and unchanged candidate policy. Test it early before long qualification.
The root's CPU-only vector preparation scaffold is under
`/ml2/hypergan/gan-attempts/continuous-api-20260926/supervisor-support/c6-vector-harness`;
it is preparation support, not an executed or qualified broader suite. Review
its unresolved host initialization/stream details before using it. Eight custom
hosts still require faithful joint-update component integration.

The exact ordinary public K3P reference remains NOT_RUN. Preserve archived
kernels, lazy CPU scalar Adam counters and baseline schedules in the same
runtime; injecting eager state or replacing kernels changes the comparator.
Do not run an expensive comparator until a candidate survives retention and
sensitive broad gates. Root coordinates ownership to prevent duplication.

Use unique API-C7 onward names. Preserve all scores and failures, no seed sweeps,
no hidden task information, no endpoint selection, no PR merges.
