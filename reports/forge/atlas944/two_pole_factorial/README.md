All four 320-update variants pass the declared escape/flatness test, while all four 80-update combinations fail. AMSGrad OFF with particle L2 zero confirms convergence earliest, at update 120 (first passing read 107, final 65 passing reads). The other confirmations are 237, 264 and 294. All five newly authorized cases completed; three prior 80-update baselines were reused without rerun.

This test checks whether particles escape a bank initialized at zero while the critic gradient remains small. Passing requires mean absolute particle position >= 0.30 and median absolute critic gradient <= 1.00 on the final five saved checks. All 80-update failures miss the escape gate. Full coverage of both target poles is not required; spread and pole proximity are ungated diagnostics. These are explicit optimizer/objective/horizon variants that preserve the original 80-update benchmark.

| Table AMSGrad | Particle L2 | Updates | Verdict | Confirmed update | Escape | Flatness | Full-run seconds |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| off | 0 | 320 | PASS | 120 | 0.91418856 | 0.01637712 | 18.904 |
| off | 0.02 | 320 | PASS | 237 | 0.82530332 | 0.04749019 | 29.095 |
| on | 0 | 320 | PASS | 264 | 0.87249970 | 0.04955092 | 16.597 |
| on | 0.02 | 320 | PASS | 294 | 0.64538652 | 0.09771565 | 31.326 |
| off | 0.0 | 80 | FAIL (reused) | — | 0.08783505 | 0.02218658 | 29.465 |
| off | 0.02 | 80 | FAIL (reused) | — | 0.01502315 | 0.00998517 | 20.862 |
| on | 0 | 80 | FAIL | — | 0.00328096 | 0.01282401 | 16.446 |
| on | 0.02 | 80 | FAIL (reused) | — | 0.00244565 | 0.01089700 | 19.655 |

The passing rows are sorted by confirmed convergence update. Full-run seconds include completing the declared horizon and are not wall-clock time to convergence. This single-seed comparison identifies the earliest-confirming variant within this diagnostic; it supplies no canonical original-task, family-wide or shipped-default winner.

Each 320-update run has 96 complete pure scored observations on the original clock lattice extended to 320. The final five clocks are 307, 310, 314, 317 and 320. Each run's first 24 ordered particles, targets, critic gradients and two metric values exactly match its paired 80-update case. Recipe.total_steps remains None. Thus extending only the external horizon lets each otherwise unchanged configuration pass after the original cap. The bounded optimizer trace remains partial UNKNOWN; full-state and optimizer-tail equality are not claimed.

Nine focused controls passed once, including an actual four-update prefix fixture matching protected932. Five fresh scientific cases completed sequentially on CPU1/gpus0 under the existing five 300-second allocations, counted once. The three-case continuation preserves the recovered communication timeout and historical halt under authenticated authority949, with no reset, new allocation or rerun. A launcher syntax error occurred before execution and is retained as UNKNOWN_NOT_ZERO overhead. Declared reserves are neither measured elapsed nor certified upper bounds. comparison.json contains a pre-publication cost cut; subsequent publication, Source authoring/model time and persistence/timeout tails do not constitute an exact all-inclusive measured cost.

Each measured cell folder contains its exact declaration, Task variant, software proof, raw result, grade, initialization, parent process receipt, saved curve and independently verified result. Numeric FAIL means a completed valid run. The archive contains the exact two Source leaves bound by the controls and declarations, based on repository commit9e55990ad3fbd58c081c9368950023a97a29f251. ROOT-only controls require the supplied retained-prefix bank; they are not installed into general test discovery.

To reproduce in an isolated copy of that commit, extract bound-source-leaves.tar.gz at its root. Run the control script with --checkout pointing there, --retained-prefix pointing to retained-932-numerical-prefix.json, its recorded SHA256, and a fresh --output file. Scientific reproduction requires a fresh bound declaration and owned attempt/output directory; never substitute a prior proof with different Source or credit this diagnostic as a canonical original-task pass.

The evidence attributes these 80-update failures to insufficient horizon for the tested seed and gates. Disabling AMSGrad or removing L2 brings confirmation earlier in this comparison, and disabling both gives the earliest confirmation. Robustness across seeds and other toy problems is outside this batch's scope.
