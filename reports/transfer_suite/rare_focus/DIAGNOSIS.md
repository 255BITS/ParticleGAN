# Why the rare mode fails

**Update: the targeted search found a fix at the original budget.** A D96×2
Softplus5 critic with a two-parameter raw-coordinate linear skip passes the
final six checks. Two independent replays, including the reusable command,
match every numerical result. The diagnosis below describes the earlier failed
run that motivated the search; it does not claim a force decomposition of the
new winner. [Winning result and its full six-data profile](README.md).

The closest previous run learns the rare cluster's location and occupancy, but
its six output points become too flat. The covariance along their narrowest
direction ends at **13.34% of the target variance**, below the unchanged 15%
minimum. Evaluating through the usual 4,096 sampled outputs gives 14.10%; both
measurements fail. This is a geometric failure, not an unlucky evaluation draw.

Two instrumented replays exactly reproduce the archived Softplus5 D96×2 result,
including all 24 live/EMA measurements and actions. The instrumentation preserves
training RNG and gradients. These are diagnostic replays, not new candidates.

## Which updates flatten the cluster?

Measure the same final six particle identities along the final narrow axis.
For each actual Adam update, evaluate all combinations of old/new generator and
old/new latent positions. A symmetric decomposition attributes the observed
variance change to the two parameter blocks without changing training.

| Final 100 updates | Variance / target variance |
| --- | ---: |
| Before | .43319 |
| Contribution from particle-position updates | **−.31165** |
| Contribution from generator-weight updates | +.01191 |
| After | **.13344** |

Particle motion accounts for the contraction in this interval. Generator updates
slightly oppose it, and local generator Jacobians remain full rank. This does not
rule out every possible long-term generator-capacity issue.

The frozen discriminator gradient also contracts the cluster. Its π-frequency
features dominate that contraction; the raw-coordinate contribution expands it
slightly. The prior spread regularizer also weakly opposes the contraction.
This motivates changing how the discriminator represents smooth spatial shape.
It does not prove that a retrained architecture will solve the problem.

## Is the test achievable with six points?

Yes, for these final output metrics: a diagnostic affine transformation of the
same six rare outputs makes every final bound pass while leaving the other 250
outputs and all sample weights unchanged. This control uses component labels and
target covariance. **It is not a trained GAN, a sustained PASS, or a leaderboard
entry.** It establishes that six points are sufficient for the measured geometry.

The failure can also move between components: at update 1,050 the narrowest
component is the 13% one; at update 1,200 it is the 2% one. The actual gate checks
all four components, not merely whether rare points exist.

[Full diagnosis, component tables and source evidence](forensics/README.md) ·
[Actual optimizer update attribution](forensics/update_attribution.json.gz) ·
[New architecture experiments](README.md).
