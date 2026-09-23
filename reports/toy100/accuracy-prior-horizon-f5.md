# Noiseless F5 prior-horizon diagnostic

The two predeclared prior LR horizons, 4,000 and 5,000 updates, both failed
the original grid100 coverage gate and the accuracy gate. Each retained all
100 modes at step 7,000, but **zero of the five required terminal checks
passed**. Shortening the prior horizon did not repair the baseline's center
or width error in this fixed-seed test.

| Prior horizon | Final modes | Final HQ | Final center RMS / σ | Final covariance trace bias | Final radial KS | 100k holdout center RMS / σ | Original / accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 7,000, original F5 control | 100 | 0.9467 | 0.499 | −0.203 | 0.0437 | 0.490 | FAIL / FAIL |
| 4,000, scratch | 100 | 0.9276 | 0.546 | −0.205 | 0.0479 | 0.552 | FAIL / FAIL |
| 5,000, scratch | 100 | 0.9346 | 0.647 | −0.161 | 0.0715 | 0.653 | FAIL / FAIL |

The frozen accuracy limits are center RMS ≤ 0.20σ, absolute covariance trace
bias ≤ 0.10, radial KS ≤ 0.04, and mass TV ≤ 0.06, together with the
original coverage requirements. All three runs had final mass TV about 0.053;
the capped-prior rows still missed the center and covariance limits. The
5,000-step row had a smaller final covariance bias than the control, alongside
a larger center error and radial KS. Its 100k holdout failed as well.

The archived predeclared manifest froze the
exact F5 config, seed
1234, 7,000 updates, CPU/one thread, Fourier 5, zero input and output noise,
the G/D LR horizon of 1,600, all other optimizer fields, resources, data,
budgets, and thresholds. The only experimental change was the particle-prior
cosine horizon, applied by the archived scratch adapter
`reports/toy100/accuracy_prior_horizon_f5_probe.py`.
The adapter and native source were pinned at commit `f2c1c3e6efc008a1d49474b49a7a813a8af2b5c1`.
Every actual G, prior, and D optimizer rate across all 7,000 updates is
archived and checked against the declared schedule; final-five 20k clouds
and an independent 100k holdout are retained for each run.

Raw evidence is stored locally at
`artifacts/toy100-accuracy/prior-horizon-f5-v1-f2c1c3e` in the main
worktree. All 114 relocated files matched their RAM originals by SHA-256;
regrading from the durable copy reproduced both FAIL/FAIL verdicts, verified
the archived source, and recalculated every G/prior/D rate. This is **scratch
grid100 evidence only**. The varying prior horizon is not a production config
field, and no common 22-task compatibility claim follows from these results.
