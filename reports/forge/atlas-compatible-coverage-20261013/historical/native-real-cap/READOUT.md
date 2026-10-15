# Native Atlas real-cap pilot: local repair supported

The separately preregistered fixed-seed-0 pair completed through the normal Forge scheduler. Both original 80-update runs and all 24 observations completed. The cap passed the unchanged full two-pole gate; native R1-on Atlas failed.

| Arm | Final mean absolute position (minimum 0.3) | Final median critic gradient (maximum 1) | Passing observations / suffix | Full gate |
| --- | ---: | ---: | ---: | --- |
| Native Atlas, real R1 on | 0.005321078 | 0.012204401 | 0/24; suffix 0 | FAIL |
| Atlas, real KA2-A RMS cap | 0.460719913 | 0.405106425 | 8/24; suffix 8 | PASS |

Only the real component of KA2-A changes. Native is E[||grad D(real)||²/d]; treatment is E[max(sqrt(||grad D(real)||²+1e-12)/sqrt(d)-1,0)²]. The expectation is outside the per-example squared hinge. Lambda remains 3; the fake cap, Adam/AMSGrad, DV12, noise, and late B/anchor/handover remain unchanged. This 80-update task cannot establish behavior at call800.

The formula is not novel: related K3P real caps and Atlas symmetric caps were recovered before this test. The local comparison tests its interaction with native Atlas settings at coefficient3. Historical ring failures remain negative evidence. No global winner, held-out transfer, or cross-seed robustness is claimed.

The initially proposed seed3/4 comparison was rejected by ordinary screening admission before any queue mutation or GPU worker. It remains preserved in ../astra-atlas-cap/ADMISSION-BLOCKED.json. The successor seed0 pair was preregistered separately; no guards or numerical gates were relaxed.

All48 saved observations were recomputed from stored particle positions and critic gradients; both independent full grades agree. Initial tensors and actual consumed target batches match. Saved checkpoints verify. Actual-training GIFs in actual-training/ visualize the retained live-table observations, with no new draws or updates.

Actual cost: 18.490690234 queue-seconds and 21.830803280 supervisor-wall seconds; no paid retry. Remaining inherited balance: 6257.010098606 queue-seconds and 4725.335402917 wall-seconds. Peak concurrency2. All owned processes are gone and all12 leases were free at terminal inspection.

Broader work remains blocked by five original-task API bindings. Gaussian/Ring require explicit live observation and sampling contracts; conditional unused-token, AE, and joint-word hosts require real ownership/routing semantics compatible with Atlas controls. The word task also pins a different policy.py. These are not honest fixes by changing declaration flags. See ../astra-atlas-bindings/FINAL-COMPATIBILITY-REVIEW.json. The conditional12-cell grid is two arms across Gaussian1D, two-pole, unused-token, AE, Ring16, and joint-word; none of those broader jobs was admitted. Its complete4680-wall-second reservation fits the remaining balance by45.335402917seconds, but source/binding readiness is still required.

The separate PR review reports PR373 at fed122ee, with6/7 scoped passes and failed Gaussian retention; 51ee4347 is PR395, not373. This is a potentially useful comparison reference, not an audited control for this pilot. Keep the historical Atlas19/19 noisy serving law separate from this clean/live parameter gate. PR401 energy/MMD, PR395 damping, and PR375 width results are retained as prior negative or unmeasured evidence; none was rerun here.

[Audit](ROUND-AUDIT.json) · [Preregistration](preregistration.json) · [Source/request bindings](PRELAUNCH-BINDING.json) · [Terminal processes/leases](TERMINAL-HOST-RECEIPT.json) · [Matrix receipt packet](INVENTORY-RECEIPT-PACKET.json).
