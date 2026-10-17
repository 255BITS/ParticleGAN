# E4 row-test portability probe

The [CPU probe](portability-sanity.py) runs the archived E4 row-gradient
estimator on synthetic gradient streams only. It does not train a GAN or
score another dataset. Its [output](portability-sanity.out) is a narrow check
of the estimator across units, rotation, particle count, and gradient
dimension.

Scaling gradients by `.5`, `2`, or `.001` and rotating them left the flags
identical in the two-dimensional probe. For particle counts from 12 to 20,000
and dimensions 2, 3, 4, and 8, this particular iid synthetic stream flagged
all planted one-standard-deviation drifting rows and none of the null rows.
Those results do not validate false-discovery control under anisotropy or
serial correlation; the independent review found inflated tail rates there.

At dimension 32, the 600-step probe supplied only about 60 touches per row.
The estimator requires an effective sample size of at least `3d=96`, so it
flagged none of the planted rows during that probe. Its fixed `W=50` window
caps effective sample size near 99, leaving very little margin above that
minimum even with more touches. For `N=12`, one flagged row is already more
than the hold's `Q=.05` fraction; the hold can engage on a single flag. These
are additional reasons not to treat E4 as a portable default.

The trained-task evidence is in [suite-leaderboard.md](suite-leaderboard.md):
the 22-task matrix is 12 PASS, 2 FAIL, 8 ERROR. The portability probe does
not change those verdicts or E4's A2 violation.
