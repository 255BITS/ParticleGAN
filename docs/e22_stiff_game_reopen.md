The Supra editing-only continuation regressed after a native generator ladder
release. An exact replay cancelled only the generator tester's scale change
from `.5` to `1` after update 6222, preserving its decision/window history and
the training input, paired Gaussian and DV12 streams. At 6400 updates, the
240-context editing game improved from `.964123865` to `.920430676` with one
common critic, and from `.977455859` to `.931124359` with another. No structural
moves were accepted. The later optimizer-surprise event disappeared downstream.
Those application results establish a harmful release on that trajectory;
they do not establish that every learning-rate increase is harmful.

`examples/e22_stiff_game_reopen.py` isolates a possible mechanism into a
two-coordinate CPU unit fixture for ParticleGAN's native `SettleTest`, native
generator Adam/AMSGrad and public RpGAN loss. The fixed nonlinear feature-score
critic is **constructed**, not trained. The explicit valid settled optimizer
snapshot is specified analytically, not extracted from Supra. This is a
generator-controller unit reproduction; it excludes particles, routed models,
KA2, structural proposals, surprise actions and random minibatches.

The critic score is negative squared projected features. The generator trains
through the public paired GAN loss, with equal real/fake base inputs. There is
no output-MSE training loss, structural guard, stopping metric or test oracle.
The public `deterministic_orthogonal_` initializer runs before the specified
parameter snapshot is loaded. There are no seed or hyperparameter sweeps.

One feature direction has local curvature `1e6`; the other has `1e-12`.
Existing AMSGrad memory gives the contracted LR a dimensionless stiff-direction
step factor `1.6`, within the local stability interval `(0, 2)`. Restoring the
LR ceiling gives `3.2`, outside it. The stiff residual starts tiny and settles
further; a persistent weak direction dominates displacement cosines. After 48
updates, both native verdicts are positive and the ladder raises its scale.
The stiff direction then escapes its settled neighborhood. Every-two-step
blocks can also alias its alternating updates as positive coherence.

The predeclared horizon is 96 updates. The unmodified native arm's final paired
game is approximately `150.380`, with a peak near `4406.59`. Cancelling only
the first scale release keeps it near `log(2)`, approximately `.69314718056`.
Exact native optimizer/tester recovery from a CPU-mapped checkpoint immediately
before the release reproduces both arms. Initial ownership, parameter values,
memory units and spectral factors are validated explicitly.

The prepared moments have a physically consistent native Adam history:
999 identical gradients build the AMSGrad maximum, then a zero-gradient
update gives first moment zero and `exp_avg_sq = beta2 * max_exp_avg_sq`.
Both geometries replay this actual 1000-update gradient program in a focused
test and recover the specified moments within FP64 rounding. This validates
moment reachability; it does not claim that this gradient program came from
the fixed nonlinear score or was the generator's past dataset.

A second analytically specified geometry has contracted/full spectral factors
`.8`/`1.6`, both stable. Its native ladder also increases after 48 updates and
the game remains bounded. The suite explicitly requires this safe increase,
so disabling every release cannot pass all the tests. This is a specified
positive control, not a hyperparameter sweep. Exact checkpoint recovery also
reproduces its release boundary.

Run the standalone reproduction and focused tests with the repository on
`PYTHONPATH`:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' python -u examples/e22_stiff_game_reopen.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_e22_stiff_game_reopen.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_e22_stiff_game_reopen.py --runxfail
```

The normal test run records the known stability defect as a strict expected
failure. `--runxfail` exposes the red test. Its oracle checks bounded paired
GAN game on this stationary fixture; it permits any safe LR change and does
not require monotonic held-out performance. After a fix, an unexpected pass
requires removing the expected-failure marker. The contracted control and
exact recovery checks pass on the current implementation.

Verified against `develop` at `fa2c378d5ea4dd8f66eabeb17973818a174732ef`:
seven passes and one strict expected failure in 1.67 seconds. With
`--runxfail`, the intended game-stability assertion fails and the other
seven checks pass in 1.72 seconds. This adds a unit regression case; it is
not a Forge candidate, screen or scientific qualification claim.

No native fix is included. It remains necessary to verify which stiff/soft
geometry, aliasing or alternative direction-weighting mechanism applies to
the actual Supra release, and to preserve existing moving-target recovery
tests when changing generator ladder behavior. The earlier canonical public
two-site stationary probe did not reproduce the regression naturally: its
6400 updates contracted the generator once and had no later release or
surprise event. That negative result is distinct from this constructed unit
fixture.
