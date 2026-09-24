# PR84 saved-state population-field comparison

At four exact fixed-target failure states, the unrestricted population
density-ratio critic points the generator toward the ring centers, while the
trained PR84 critic usually points it outward. This separates the observed
wrong-direction field from an unavoidable consequence of the 12-versus-eight
Gaussian mismatch. It does **not** identify the optimum of the actual finite
critic with its `b_cap` penalty, or establish a stable replacement update.

The [completed diagnostic and setup-failure archive](continuous-evidence/pr84-population-field/manifest.json)
contains the frozen sources, all four raw state results, declaration, summary,
test source and logs. The first script version stopped with `KeyError: 'values'`
before producing any state row: the prior capture stored latent vectors under
`latent`. The second version corrected that read-only capture check. Neither
attempt advanced training. Two analytic tests passed. The completed run uses
the saved `post_accepted_d` states at steps 1,325, 1,389, 1,530 and 1,540.
Each arm receives the *same* actual next G minibatch and 16 subsequent held-out
minibatches; all 17 batch hashes agree across sharp/stencil and learned/analytic
arms within each state. Real/fake sampling and the PR84 five-point G stencil
are replayed, with the stencil width frozen at the saved step's `.15`.

For real density `p` (eight equal Gaussians, σ `.07`) and fake density `q`
(12 equal Gaussians at current clean G outputs, σ `.029`), the diagnostic
critic is `D*(x) = log p(x) - log q(x)`. This is the unrestricted,
**unregularized** population optimum of paired logistic D loss. The 12 clean
supports inside `q` are detached when G samples are differentiated. Both
sharp D* and its five-point stencil are diagnostic foils; the trained D uses
the host's finite Fourier MLP and D-only `b_cap` objective. The
[structural note](rp-misspecification-stability.md) gives the derivation and
the theorem limits.

| Saved update | Trained-stencil raw G-network direction outward / 16 | Trained-stencil pre-update Adam-metric direction outward / 16 | Population sharp and stencil inward, both direction measures |
| --- | ---: | ---: | ---: |
| 1,325 | 16 | 16 | 16 / 16 each |
| 1,389 | 14 | 15 | 16 / 16 each |
| 1,530 | 9 | 14 | 16 / 16 each |
| 1,540 | 15 | 16 | 16 / 16 each |

“Outward” is positive directional cosine against the *offline* nearest-center
squared-distance gradient at the frozen clean supports. Target centers are
never read by training. Across these four states and 16 held-out batches, the
population score gives **64/64 inward** raw G-network directions and 64/64
inward saved-metric directions, with or without the stencil. The trained
stencil's raw outward count is 54/64. Its direct clean-output score gradients
also oppose the analytic score gradients. For states 1,325, 1,389 and 1,540,
the trained-stencil raw and saved-metric direction fields match the earlier
[held-out receipt](pr84-heldout-signal.md) exactly: maximum numerical
difference zero in every row, with identical support, width and raw-gradient
coherence. Step 1,530 was not in that earlier receipt.

The metric comparison multiplies the batch gradient by
`P_old = lr/(sqrt(v_old/(1-beta2^t))+epsilon)`. It is a **frozen pre-update
metric**, not the actual Adam proposal: real Adam updates its second moment
first, then PR84 may scale the parameter proposal by its own-curvature rule.
Likewise, the independent output-gradient diagnostic does not include the
shared G-network Jacobian coupling or final bounded proposal. The paired
parameter-space result still shows that the saved network pullback and
pre-update metric do not alone turn the analytic inward gradient outward.

The analytic critic is sharply different from a capped host critic. A further
read-only [cap receipt](continuous-evidence/pr84-population-field/cap_diagnostic/ideal-cap-actual-d.json.gz)
replays the actual first D-phase real and prior samples from saved pre-step
states. Real values, latent vectors and indices match the exact capture; fake
output-noise draws are reconstructed from its saved RNG. The learned D's
recomputed paired loss matches the archived `log(2) − critic_advantage`
bit-for-bit at all four steps. With the host's coefficient one and κ one,
the sampled `b_cap` values are:

| Step | Unscaled analytic D*: D loss / cap penalty | Saved learned D: D loss / cap penalty |
| --- | --- | --- |
| 1,325 | .14713 / 8,439 | .66332 / 0 |
| 1,389 | .10707 / 15,383 | .69232 / 0 |
| 1,530 | .10385 / 12,029 | .67843 / 0 |
| 1,540 | .13095 / 39,081 | .68126 / 0 |

All 128 real and 128 fake analytic input-gradient norms exceed κ on each
sampled bank. These are single-batch estimates of the expected penalty at
each state, not a measured global expectation. The cap is soft, so D* is not
forbidden, but its *penalized* objective on these exact banks is thousands
versus about `.66–.69` for the saved D. The inward analytic field therefore
cannot be treated as the host's penalized best response. A scaled or reshaped
critic could retain inward guidance; this comparison by itself does not tell
whether the trained D lags such a critic or whether the finite penalized
objective prefers a different field.

A separate [frozen-q D-only relaxation diagnostic](pr84-critic-relaxation-diagnosis.md)
provides a useful check. Its source is
`reports/toy100/pr84_critic_relaxation.py` at SHA-256
`9f5f0d630e818d5259081c333f55e667f8a619ae81b25ebdd1bb62d334d7639b`;
local artifacts are under
`artifacts/continuous-learning/stationary-opponent-prediction/critic-relaxation-v1`
and `critic-relaxation-heldout-v1` in the mechanism worktree. At steps
1,325/1,530/1,539, its reported held-out *penalized* D losses fall
`.66591→.62919`, `.65424→.61716`, `.65191→.55991`. On separate held-out G
batches the fitted critic points inward for 24/24 raw and accepted proposals,
versus the saved critic's 2/24 raw and 1/24 accepted. **All three fits remain
nonconverged** (reported gradient residual ratios `2.87×/4.83×/3.36×`), so
this shows a better local critic can supply inward feedback, not a certified
best response or a training solution. At 1,539 the run has already lost a mode,
and the inward proposal does not restore the pass gate in one update. Those
separate artifacts are retained in their
[own committed manifest](continuous-evidence/pr84-critic-relaxation/manifest.json).

The next causal question is whether D can track that better capped field
*during* the continuing game without spoiling cold acquisition. A frozen-q
fit alone cannot answer it; any candidate still needs the matched warm,
from-scratch acquisition, and dense longer hold gates.
