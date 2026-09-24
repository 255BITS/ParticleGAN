# Does constant-rate drift persist when the target law is representable?

Yes, ordinary constant-rate Adam drifted from a constructed exactly matched
population law. The generator functional-metric step held that synthetic law
more closely for 200 updates, but neither method sustained recovery after a
+0.35 x translation. These are causal diagnostics with a changed target and
critic, **not** original mode-hold or PR #60 gate results. The exact
[declaration and raw receipts](continuous-evidence/matched-population-diagnostic/)
include the four live branches, original scheduled-prefix control, source
bytes and optimizer/noise state checks.

At the original scheduled update-1000 state, the diagnostic froze all 12
current **live** clean `G(prior.z)` outputs as equal-weight target centers.
It changed target sampling from eight radius-three Gaussians with standard
deviation .07 to twelve Gaussians centered at those outputs with standard
deviation .029, matching the generator's already fully warmed output noise.
The critic's final affine weight and bias were set to zero, so its output
started constant. Critic Adam second moments, generator and prior weights,
their Adam moments, EMA, and every training RNG were preserved. The same
post-intervention state was forked four ways: ordinary constant Adam,
functional-metric G damping, and one frozen-after-shift sibling for each.
All three nominal group rates remained exactly G/D .00425 and prior .0085.
The original 1,200-step noise horizon stayed fixed.

The primary metric was an **analytic population** Gaussian-kernel MMD²
between the equal-weight, two-dimensional mixtures, each convolved with
isotropic σ=.029 output noise. Its bandwidth .21 equals the original HQ
radius, and its closed form is in
[the diagnostic source](matched_population_diagnostic.py). It is zero at the
constructed initial match and invariant to particle permutations. The
predeclared diagnostic bound was MMD²≤.01, observed every ten updates
through the hold and 400-update response. As in the original continuous task,
a transient post-shift crossing does not count as sustained recovery. Indexed clean-center RMS
was recorded only to show physical movement; it is not a distribution score
because a particle permutation can leave the mixture unchanged. Neither
metric entered either learner's updates.

| Branch | Hold MMD²≤.01, updates 1010–1200 | Hold last MMD² | After shift MMD²≤.01, 1210–1600 | Final MMD² | Final centroid error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant Adam | 14/20 | .01234 | 3/40, transient | .12641 | .10623 |
| Constant, frozen after shift | 14/20 | .01234 | 0/40 | .17596 | .31953 |
| Functional metric | 20/20 | .00279 | 0/40 | .20521 | .72027 |
| Functional metric, frozen after shift | 20/20 | .00279 | 0/40 | .19188 | .34923 |

The two frozen siblings had exact model, optimizer, EMA, RNG hashes and
diagnostic curves matching their active sibling through update1200. At the
shift their target centers moved in place, with no model or optimizer reset.
Each frozen sibling made no subsequent Adam updates and retained constant
post-shift MMD². Constant Adam reduced final MMD² relative to its frozen
control and briefly crossed .01 near update1230, but failed sustained
recovery. Functional metric reduced stationary drift, then finished worse
than its own frozen control by this distribution metric. The metric's final
indexed-center RMS was 3.008; because the MMD kernel saturates for distant
points, this movement is reported separately rather than inferred from the
MMD alone.

This intervention rules out *unrepresentable target law alone* as a complete
explanation for the local constant-rate drift: the initial law matched
exactly, yet ordinary Adam exceeded the predeclared .01 discrepancy bound at
six hold checks. It does not identify whether finite-batch critic noise,
alternating updates, preserved stale D variance after the affine reset, or
another mechanism dominates. The critic reset and new twelve-center target
deliberately alter the original game, so the experiment says nothing about
original eight-mode acquisition. A quiet response to a matched target is
acceptable; the missing sustained translation response is the observed
limitation here.

The scheduled prefix hash matched a separate uninterrupted original host at
step1000. All branches inherited the same post-intervention hash; two
analytic metric tests passed. The [source archive and declaration](continuous-evidence/matched-population-diagnostic/declaration.json)
bind the exact runtime code and the bounds written before the four branches
were evaluated. The experiment ran with Python 3.12.13, PyTorch 2.13.0+cu126
on CPU and one PyTorch thread per process.
