# 22-toy results and continuous stability

**Yes, there is a recorded 22/22 PASS.** The original
`constraints_simple_regularization` recipe still passes when its saved CPU
evidence is independently regraded. Its preserved-recipe CUDA control scores
**16/22**, with all three native 100-mode gates passing.

| Recipe / scope | Backend | Toy PASS / 22 | Native PASS / 3 | Post-convergence hold |
|---|---|---:|---:|---|
| Original recipe, decay and original auxiliary host terms | Recorded CPU, regraded | **22/22 PASS** | 3/3 | Not a constant-rate claim |
| Same original recipe | CUDA control | **16/22** | 3/3 | Not a constant-rate claim |
| Shared column RMS, strict continuous variant | CUDA | 11/22 | 0/3 | 131 good updates, then FAIL |
| Shared RMS, strict continuous variant | CUDA | 11/22 | 0/3 | 35 good updates, then FAIL |
| H, strict continuous variant | CUDA | 11/22 | 0/3 | Not confirmed by 6000 |
| Epsilon, strict continuous variant | CUDA | 9/22 | 0/3 | 19 good updates, then FAIL |

[Original 22/22 evidence](simpler22/README.md) ·
[Same-recipe GPU control and all 22 results](gpu-known-winner-control/README.md) ·
[Continuous-learning GPU matrix](gpu-leaderboard/LEADERBOARD.md)

The first GPU report omitted the original winner and evaluated only eight
continuous-learning variants. Its statement that no candidate qualified applied
to that cohort. It did not invalidate the earlier passing recipe. The new
control supplies the missing same-recipe CPU/GPU comparison.

The original recipe uses decaying learning rates and the frozen auxiliary
losses in the autoencoder and unused-token hosts. The strict H-family variants
keep rates active and disable those auxiliary terms. Compare these scopes
explicitly: a finite-budget pass with decay is not evidence of continual
stability with active rates, and the historical 22/22 is not a claim that every
host uses an exclusively adversarial objective.

The preserved-recipe GPU failures are `mode_hold`, `trajectory`, `img_bars4`,
`img_blobs4`, `img_intensity2`, and `vector_unequal_mass`. The other sixteen pass,
including native coverage and accuracy on all three 100-mode problems. All 22
GPU control verdicts and all 105 earlier GPU variant results were regraded.

The active work now starts from the **original scheduled CPU recipe**, resolving
its GPU failures first. [Twenty-four porting controls](cpu-recipe-gpu-port/README.md)
confirm 6/6 fresh CPU passes on the failed hosts. CPU initialization alone
recovers four on CUDA; ring and unequal mass remain failing. This partial
diagnostic is not a new 22-toy score.

Shared column RMS is retained as a historical **strict continuous-learning**
reference; it is not the active porting base. None of the
continuous variants passes the required 1,200-update hold after confirmation.
PR107/140 do not confirm by 6000; PR143 confirms at 1400 and fails after 11 good
hold checks. Their published adapters cover only two toys.

[GPU protocol and replay](gpu-leaderboard/README.md) ·
[Historical strict GPU reference](gpu-leaderboard/current-gpu-reference.json) ·
[Historical continuous-learning CPU board](continuous-practical-leaderboard-cpu-history.md)

[PyTorch 2.14 upgrade check](torch214-gpu-check/README.md): both remaining
blockers fail identically to 2.13 under native and CPU initialization. Four
full-budget CUDA runs; no change to the 16/22 full-suite result.

[Completed formulation round](formulation-round-20260924/README.md): 18 proposals,
36 primary GPU gates, 7 PASS / 29 FAIL. None passes both blockers. The full-suite
reference stays 16/22. [Measured follow-ups](formulation-round-20260924/FOLLOWUPS.md)
include two formulations that pass both blockers but fail trajectory; neither is
promoted. R1/R2 history is a reason to avoid unchanged repeats, not to reject a
candidate that passes its measured gates.
