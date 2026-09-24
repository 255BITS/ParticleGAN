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

Results will be added after the declared run and durable raw-episode regrade.
