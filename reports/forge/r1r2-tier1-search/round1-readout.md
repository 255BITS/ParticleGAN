# Round 1: constant global R1/R2 policies

All seven whole recipes completed their six required Tier 1 tasks and separate clock measurement. None qualifies as a complete Tier 1 solution. The fresh scheduled baseline reproduces the archived Ring16 and pole endpoints exactly; archived source identities remain separate.

| Global policy | Gaussian | Pole | Token hold | AE hold | Ring16 | Word acquisition | Clock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Scheduled baseline | PASS | PASS | PASS | PASS | FAIL | PASS | FAIL |
| Constant beta2=0.99, gamma=1.0 | PASS | FAIL | PASS | PASS | FAIL | FAIL | PASS |
| Constant beta2=0.99, gamma=0.1 | PASS | PASS | PASS | PASS | FAIL | FAIL | PASS |
| Constant beta2=0.9, gamma=1.0 | PASS | FAIL | PASS | PASS | FAIL | PASS | PASS |
| Constant beta2=0.99, gamma=0.3 | PASS | PASS | PASS | PASS | FAIL | FAIL | PASS |
| Constant beta2=0.9, gamma=0.3 | PASS | PASS | PASS | PASS | FAIL | FAIL | PASS |
| Constant beta2=0.9, gamma=0.1 | FAIL | PASS | PASS | PASS | FAIL | PASS | PASS |

The complete [numerical publication](publication/results.json) binds exact recipes, task/source/runtime identities, checkpoints, costs, and 49 actual-training GIFs. The six ordinary tasks each have identical actual initialized tensors and ordered consumed target batches across all seven recipes. All named streams are checkpointed. Clock comparisons retain their own certified saved trajectories. Input-audit PASS means the comparison conditions match, not that a scientific gate passed.

All six constant laws pass the clock probe; all six fail Ring16. No Ring16 observation in any arm meets the full covariance bound. The best observed component covariance error is 1.568014, still above the unchanged 0.85 bound. Removing schedules alone does not repair the density failure. Strong constant gamma=1 fails pole transport; weaker constants can retain transport but also expose word or Gaussian acquisition failures.

The registered baseline study also predicted terminal Gaussian mean_error_sigma<=0.2. Its endpoint is 0.228405, so that auxiliary prediction is falsified; the actual Gaussian acquisition gate still passes its earlier independently confirmed checkpoint. The declaration and numerical verdict remain unchanged.

Charged work: 1259.1354882339656 seconds across 49 unique attempts, within the 3,600-second cap and 17,640-second full reservation. No execution errors, retries, Tier 2 jobs, seed changes, selection or defaults changes. Collection/rendering adds zero optimizer updates and sampling draws.

Saved observations show late high-rate bursts: beta2=0.99/gamma=0.3 loses 16-to-1 modes and HQ 0.7695-to-0.1094 between updates 1,584 and 1,600; beta2=0.99/gamma=0.1 also has a sharp late HQ dip. This supports testing a smaller fixed global update magnitude in a separately registered follow-up. Burst reduction alone cannot pass the original density gate.
