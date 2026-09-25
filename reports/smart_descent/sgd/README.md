# Raw-gradient descent research

No tested raw-gradient rule sustained full coverage and HQ≥90% on both development tasks. Common-LR SGD collapsed; per-tensor steps substantially improved coverage, and target decay produced a sustained ring4 pass. Shared performance across tasks remains unresolved. No new distribution or architecture was used to tune this study.

| Method | Ring4 live modes / HQ | Grid9 live modes / HQ | Sustained tasks | Objective ↓ |
| --- | ---: | ---: | ---: | ---: |
| Adam / cosine | 4/4 / 83.64% | 7/9 / 100.00% | 0/2 | 20.9395 |
| Tuned common-LR SGD / constant | 1/4 / 100.00% | 1/9 / 100.00% | 0/2 | 26.2143 |
| Tuned common-LR SGD / cosine | 0/4 / 0.00% | 1/9 / 100.00% | 0/2 | 27.4959 |
| Per-tensor current RMS / constant | 4/4 / 74.95% | 6/9 / 67.77% | 0/2 | 22.7123 |
| Per-tensor current RMS / cosine .6 | 4/4 / 100.00% | 3/9 / 75.78% | 1/2 | 12.1699 |
| Per-tensor running RMS .99 / constant | 4/4 / 100.00% | 7/9 / 83.42% | 0/2 | 21.1546 |

The objective includes a 20-point penalty for each task without sustained success, then final/late coverage and HQ deficits and a small SW1 term. Its mean therefore ranks the single-task cosine success highly despite worse grid coverage. This is not an overall PASS. All rankings use live weights; raw artifacts report EMA separately.

1. [Common-LR SGD and scalar feedback](common_lr/README.md): 16 G/D rate pairs under constant/cosine schedules, 64 episodes; 36 numerical failures. A 16-policy causal feedback search added 32 episodes but selected zero coefficients—the constant baseline.
2. [Per-tensor relative-step grid](per_tensor/README.md): 16 target-fraction pairs, 32 episodes, no numerical failures. The best fixed rule reaches 4/4 and 6/9 modes.
3. [RMS-memory and decay follow-up](rms_decay/README.md): 8 conditions, 16 episodes. Current RMS plus delayed cosine sustains ring4 from 950 (confirmed 1150); grid9 falls to 3/9 modes. RMS beta .99 gives more balanced final coverage but no sustained task.

Read-only layer diagnostics exactly reproduced all metrics. At initialization the common-LR SGD generator updates its first weight matrix by 0.0095% of its RMS, output weights by 1.24%, and output bias by 25.8%. A common scalar multiplier preserves this disparity for a given gradient; this motivated per-tensor rates. It is a hypothesis about the collapse, not proof of its cause.

Five analytical tests verify raw-gradient updates, absent momentum/optimizer state, unequal tensor LRs, and causal RMS memory. Source hashes match every archived phase. JSON archives use deterministic gzip (`mtime=0`); each phase includes a manifest with original-byte SHA256s and an exact source bundle.

Implementation and reproduction: [SGD study](../../../benchmarks/learned_lr/sgd_study.py), [per-tensor grid](../../../benchmarks/learned_lr/relative_sgd_study.py), [RMS/decay follow-up](../../../benchmarks/learned_lr/relative_sgd_followup.py), [method notes](../../../benchmarks/learned_lr/SGD_README.md).
