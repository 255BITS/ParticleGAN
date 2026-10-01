# Completed RA2 grid100 diagnosis

The frozen verdict is FAIL because only the last two of five terminal live
checks pass. `passing_checks=3` counts coverage passes across all34 observations
(2750,6750,7000), not3/5 terminal passes. Step2750 still fails fidelity because
center RMS=.28086σ. Coverage and fidelity both first pass at6750 in the terminal
window; the required final streak is5 and the observed streak is2.

| Step | Precision >=.97 | Max covariance eigen ratio <=1.7 | Center RMS <=.20σ | Radial KS <=.04 | Terminal |
|---:|---:|---:|---:|---:|---|
| 6000 | .96995 | 1.77539 | .23413 | .04303 | FAIL |
| 6250 | .97105 | 1.77794 | .22975 | .03989 | FAIL |
| 6500 | .97210 | 1.83027 | .23111 | .04078 | FAIL |
| 6750 | .97785 | 1.68483 | .19001 | .02540 | PASS |
| 7000 | .97785 | 1.59954 | .19122 | .02642 | PASS |

Every terminal cloud covers100 modes. Mass TV is.04285–.04330, below both
coverage and fidelity limits, and absolute covariance trace bias passes all
five checks. The persistent failures are center error and excessive width in
some modes, with additional precision/KS failures at6000 and KS at6500. The
precision miss at6000 is one of20000 points; removing that miss would leave
the other three blockers.

The prior tester retains its drift verdict from4104 with scale1 and block256
through6500. It declares stationarity at6664, halves its scale to.5 and block
to128, and the unchanged serving rule enables EMA. Live and EMA clouds are
identical at6750/7000. The saved EMA already passes at6250 and6500, when fast
weights are still served, so late stationarity/averaging explains the terminal
improvement. This is a consistent serving decision. EMA still fails the6000
covariance ceiling (1.72032), giving4/5 saved EMA terminal passes; choosing EMA
earlier alone would still fail the frozen5/5 rule.

Saved motion also supports residual prior movement within modes. During the
copy-free6250→6500 interval, the exact affine decomposition attributes.47167σ
RMS to prior movement and.01341σ to G, with.44097σ within-mode movement and
.16848σ mode-mean movement. Larger raw RMS at6000/6250 is dominated by40/20
ordinary copies, so those row jumps cannot be treated as typical unresolved
motion. No terminal isolation flags are present; ordinary cumulative moves
increase by40/20/0/0/42 over the five preceding250-update intervals. The count
blind spot in the diffuse25-mode learned toy is therefore not reproduced as
widespread terminal unsupported flags in this grid100 trajectory.

Final fidelity and the independent100k holdout pass: final precision=.97785,
center RMS=.19122σ; holdout precision=.97839, center RMS=.17462σ. They do not
replace the sustained-terminal requirement. Clean evaluation remains too
narrow (final trace bias=-.85570, KS=.62700); the primary noisy law supplies
the recipe's fixed-floor output sigma=.029 throughout the terminal window.

The full run takes1759.07s, about4.49 times root's archived392s comparison.
The saved timing is uneven:250-step intervals at5000/5250/5500 take159/159/138s,
then5750 takes41s and the final intervals take20–23s. Birth/death snapshot
duration falls from3.80/3.86/2.58s to.33–.36s despite the same recorded work
(10.244M cell distances,204.8M projection products and30064 exact count terms).
These JSONs do not attribute the slowdown to the sampler. Root's scheduled
profile should establish present cost separately from this observed transient.

`grid100-analysis.json` records the exact metrics, failures, controller/motion
evidence and hashes. All gate/scorer source hashes and before/after execution
integrity checks were verified. This analysis reads saved JSON only, imports
no Torch, and runs no trajectory, optimizer update, CUDA context or changed gate.
