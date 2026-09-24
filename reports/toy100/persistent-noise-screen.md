# Constant learning rate with persistent instance noise

This is a bounded scratch experiment, not a production shared recipe. The
predeclared eight candidates are old/simple regularization core × learning
rate {0.001, 0.0025} × persistent input-noise standard deviation {0.1, 0.5}.
All use the original alternating ordinary Adam, beta=(0, 0.999), D multiplier
1, prior multiplier 2, and one D/G gradient evaluation per outer step.
`lr_floor=1`, `lr_anneal_start=0`, and removal of the network horizon/floor
make every actual optimizer-group learning rate constant. No iterate averaging
or additional optimizer coefficient is introduced.

The earlier constant-rate screens annealed input noise to zero after the first
10% of the budget. This screen explicitly changes that rule to
`sigma(t) = input_noise_std` at every update. The stored ordinary
`input_noise_anneal_end` field is superseded by an explicitly bound scratch
policy; it is not interpreted as persistent noise. Input noise is still drawn
independently for critic calls on data coordinates, using the existing dedicated
stream. Conditioning coordinates, output-noise warmup from 0 to 0.029 in the
first 20%, output/global RNG conventions, and host evaluation scopes stay as
implemented in the frozen adapter. Persistent sigma is a tuning parameter in
native data units; this experiment does not remove scale dependence.

Mescheder, Geiger and Nowozin analyze Gaussian instance noise for the Dirac-GAN.
Their Lemma 3.2 gives the game-Jacobian eigenvalues
`f''(0) sigma^2 ± sqrt(f''(0)^2 sigma^4 - f'(0)^2)`. With `f''(0)<0` and
positive sigma, the eigenvalues have negative real parts, supporting local
gradient-descent convergence for sufficiently small learning rates. This is a
mechanistic motivation, not a convergence guarantee for our stochastic,
relativistic Adam GAN with b_cap regularization.
[Primary paper, section 3.2](https://arxiv.org/html/1801.04406v4#S3.SS2)

Before training, the source is committed and an immutable manifest binds all
eight generated configurations, implementation/driver sources, and original
host sources. The fixed seed is 0. Each row first receives all 1,200 mode_hold
updates and, only after a strict pass, all 400 trajectory updates. The original
24 live checkpoints, thresholds and five-check terminal requirement remain
unchanged. Both passes mean READY_EXPAND only. Full remaining transfer tests,
fresh old19/native3/common22, and longer continuation/shift tests would still
be required. There is no nearby row expansion if all eight fail.

The scoped adapter changes the noise-sigma function; it does not transform host
code. An observer calls the original PyTorch Adam step exactly once and logs
each group's actual learning rate, beta values, moment count, gradient RMS and
update RMS. It records every noise clock and input-noise application, including
the observed sigma and perturbation RMS. Unit tests check bitwise optimizer
parameter/state equality with ordinary Adam, unchanged global RNG, restoration
on exceptions, and rejection of altered sigma/rate/moment/coverage receipts.

The policy-specific independent verifier checks these actual receipts, then
runs the frozen source/budget/model/live-metric verifier with only the expected
input-sigma formula set to the declared constant rule. Saved receipts are not
rewritten to resemble linear annealing. All artifacts are explicitly marked
ineligible for the unchanged production common gate, which must reject them.

## Result

All eight candidates failed the full-budget mode_hold gate. There were no
training or audit errors. Each received 1,200 D and 1,200 G gradient
evaluations. Trajectory, the remaining transfer/native gates, continuation and
distribution-shift tests were skipped. This screen supplies no shared-recipe
or continual-learning success claim.

| Row | Core | LR | Constant input sigma | Final modes | Final HQ | Passing suffix / 5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pn000 | old | .001 | .1 | 8 | .955566 | 2 |
| pn001 | old | .001 | .5 | 4 | .203857 | 0 |
| pn002 | old | .0025 | .1 | 6 | .802979 | 0 |
| pn003 | old | .0025 | .5 | 5 | .374756 | 0 |
| pn004 | simple | .001 | .1 | 8 | .929199 | 2 |
| pn005 | simple | .001 | .5 | 0 | .000000 | 0 |
| pn006 | simple | .0025 | .1 | 5 | .539062 | 0 |
| pn007 | simple | .0025 | .5 | 1 | .019775 | 0 |

For pn000 the five terminal checkpoints were modes/HQ
`5/.575195, 6/.667725, 8/.774414, 8/1.000000, 8/.955566`.
For pn004 they were
`8/.917236, 7/.728271, 7/.506592, 8/.981445, 8/.929199`.
Both good endpoints are therefore failures of the sustained criterion.

Every recorded Adam-group rate matched the declared constant rate, and all
1,200 input-noise clocks per row matched the declared constant sigma. In pn000,
mean G update RMS decreased from .00150753 over the first 100 updates to
.000683182 over the last 200; the prior values were .00316352 and .00140069.
Their actual learning rates remained .001 and .002. Persistent smoothing did
not suppress the terminal swings sufficiently in this bounded comparison.
Do not widen this nearby sweep or promote a passing endpoint. These results
also do not establish that every possible constant-rate method must fail.

The run used committed implementation `660279039738e0379cbf23d1f5e0c36395c38b76`.
All eight compressed raw episodes were independently regraded after relocation.
All 117 original RAM files matched their durable copies byte for byte. The
final inventory includes the additional independent-regrade result (118 files).
The measured case-time sum was 68.950 seconds, including receipt observation
overhead. The 99 relevant tests passed before training.

- [Full leaderboard and actual-update diagnostics](persistent-noise-screen.json)
- Durable evidence: `artifacts/toy100-constraints/particlegan-persistent-noise-wave1-6602790`
- Manifest SHA256: `5a65f8708276ce169988d04b9f26f94e47c0b2efe9052438df80b4a8b599f3e4`
- Inventory SHA256: `4876a2b8ff060311b16e1e132959bd96fc849a54c70359ed881dca87802d9192`

Regrade the retained raw records with:

```sh
/tmp/pr38-default-env/bin/python reports/toy100/persistent_noise_probe.py regrade \
  --root artifacts/toy100-constraints/particlegan-persistent-noise-wave1-6602790
```

The earlier 79 OAdam/AMSGrad rows and 32 joint ExtraAdam/simultaneous-Adam rows
also failed mode_hold. Together these three committed screens contain 119
declared, strictly regraded failures; no additional seeds were explored.
The inherited reports retain their complete declarations and leaderboards:
[constant-game](constant-game-screen.md) and [ExtraAdam](extra-adam-screen.md).
