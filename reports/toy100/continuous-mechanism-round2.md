# Continuous-learning round 2: attribution and projected skew response

**No qualifying replacement.** This round first replayed the previous cross-only
near miss exactly, then tested one rotation-only mechanism. Its numerically
repaired implementation fails 3 of 200 warm checks. No cold run was launched
after that result. Production losses, thresholds, budgets and defaults remain
unchanged; every new result is marked `shared_gate_eligible=False`.

The [evidence manifest](continuous-evidence/round2-mechanism/manifest.json)
contains deterministic compressed raw results and exact source archives.
The [derived summary](continuous-evidence/round2-mechanism/summary.json)
retains per-row directional gradients and the late warm trace. All experiments
use the existing seed 0; there is no seed search.

## Exact local attribution

[`cross_response_diagnosis.py`](cross_response_diagnosis.py) observes clean
support, critic input gradients, generator/prior motion and actual per-query
training losses. It checks that observations preserve every training RNG
stream. The warm final full-state hash and dense metrics match the earlier
cross-only run exactly; the cold final result and all 24 production measurements
also match exactly. The observer's original runtime bytes are archived; its
current version additionally finds the committed compressed reference results
when local artifacts are absent.

The warm failure is **within-mode drift**, rather than particles touring
between mode identities. There are zero nearest-center label changes over all
200 updates. Particle 0 moves from radius .0506 at update 1180 to .2538 at
1194, outside the HQ radius .21, before returning to .1703 at 1200. Particle 10
also approaches the boundary. At 1194, total motion has cosine -.114 with the
direction toward the target centers and +.652 with the current critic input
gradient. G network motion dominates prior motion.

| Warm update | Particle 0 radius | Particle 10 radius | D own-loss improvement at new G | G own-loss improvement at new D |
| --- | ---: | ---: | ---: | ---: |
| 1194 | .25380 | .18896 | .00809 | .000697 |
| 1195 | .24518 | .19500 | .01828 | .001052 |
| 1196 | .23406 | .20484 | .01160 | .000472 |
| 1197 | .22955 | .19979 | .02254 | .000547 |

Both individual losses improve at every failing step. A proposed guard based
only on these opponent-conditioned loss improvements therefore cannot reject
the observed bad updates. That proposal was ruled out without a training run.

The exact failing step numbers apply to this environment and warm state
`6cc79b6e0d11eafae176b68e7d9d8c26c02c886866370134cd70c864fe882e21`.
They are not claimed to reproduce across different numerical backends or
passing-state hashes.

The cold trajectory's final identity MSE .020058 is concentrated in identities
2 and 3, which have swapped nearest targets by update 110. Their coordinate
MSEs are .11716 and .11930; the other ten rows are close to their own targets.
The critic input gradients at the two wrong rows point toward their correct
conditional targets, with cosines .98771 and .97236. However, the frozen
set-cover auxiliary term pulls them back toward the wrong assignment.

| Final row | Adversarial output-gradient norm | Cover output-gradient norm | Gradient cosine | Sum norm |
| --- | ---: | ---: | ---: | ---: |
| 2 | .0847973 | .0854998 | -.995520 | .0080908 |
| 3 | .0846052 | .0841346 | -.995389 | .0081159 |

These are clean support gradients, not a claim that every noisy training
minibatch has identical cancellation. Full vectors, errors and critic gradients
for all twelve rows are retained in the summary and raw archive. The evidence
identifies a bad assignment basin: a dynamics method that preserves the game
field's zeros must reach a better basin during acquisition. It does not justify
weakening the frozen cover loss or the identity gate.

## One projected rotation-only mechanism

The revised July 2026 [LRSGA paper](https://arxiv.org/abs/2510.25716v2)
separates the game Jacobian into symmetric and antisymmetric parts and develops
low-rank approximations to the latter. Its smooth deterministic assumptions
do not establish convergence for this detached, noisy neural game.
The September 22, 2026 [Polyak extragradient paper](https://arxiv.org/abs/2609.26581)
also warns that a nonvanishing stochastic step can fail to find the mean
operator's root without a common component solution. Neither paper is treated
as a guarantee for this experiment.

[`projected_skew_scratch.py`](projected_skew_scratch.py) implements a new
rank-two implicit skew response inspired by that decomposition. It is **not**
the published LRSGA recurrence. With fixed per-update Adam metric
`P = diag(role_lr / (sqrt(v_hat) + eps))`, let `q = sqrt(P) F` and
`v = q / ||q||`. Two same-batch central finite-difference cross-field products
give `w = normalize(orthogonal(J_cross v))` and
`a = (wᵀ J_cross v - vᵀ J_cross w) / 2`. Then

```text
A = a (w vᵀ - v wᵀ)
u = (I + A)^(-1) (-q)
  = -||q|| v / (1 + a²) + a ||q|| w / (1 + a²)
theta_new = theta_base + sqrt(P) u
```

The projected symmetric component cancels, exact zero fields stay still, and
the metric step norm cannot exceed the ordinary joint-gradient step. Own-player
Hessians are excluded. In a two-dimensional bilinear game the solve is exact;
in higher dimensions it omits unmeasured skew directions. No nonlinear quality
guarantee is claimed. The measured full-joint residual is observational only.

Each outer update advances Adam moments once. D and G/prior gradients are
captured at one joint point. Every additional field query restores all model
parameters, buffers and original global/host/noise RNG states, then executes
the actual detached training block. This avoids silently losing cross terms
through autograd's `detach` or temporary `requires_grad` flags. Typical cost
is ten full gradient evaluations per player per update. Nominal post-fork
rates stay G/D .00425 and prior .0085, with no training-age schedule.

## Results and the explicit diagnostic exception

| Version | Warm dense result | Original five tail checks | Cold acquisition |
| --- | --- | --- | --- |
| v1, common finite-difference epsilon | FAIL 199/200; only 1001 fails, HQ .63916 | 5/5 pass | Numerical execution error at update 8; no quality verdict |
| v2, epsilon sized separately per player | FAIL 197/200; 1001, 1142, 1143 fail | 5/5 pass | Not run |

Both warm runs retain exact scheduled identity parity and the same warm-state
hash. Ordinary alternating Adam and the simultaneous joint-gradient control
each pass only 6/200 warm checks. v2 finishes with eight modes/HQ .999756, but
the final recovery does not erase its dense failures.

v1's first new update moved clean outputs by .25810 (.23836 from G, .02072
from prior), despite its metric norm ratio being bounded at .83407. It then
passed all remaining 199 checks. One cold diagnostic was explicitly authorized
and predeclared as an exception to the conservative transplant filter:
`single transplant update failure followed by 199/199 stable; tests cold own
dynamics`. The warm FAIL remained unchanged and no promotion was implied.

That diagnostic stopped at update 8: one common finite-difference epsilon made
the D perturbation about .00113 but the G perturbation only .00000345, violating
the declared 5% float32 direction-error limit. This is an invalid execution,
not a cold-quality FAIL. v2 repairs only numerical differentiation: each
perturbed player receives its own representable physical displacement and
its own central-quotient denominator. The 5% guard is unchanged. Its warm run
adds two later failures, so the limited transplant exception no longer applies.
The cold driver enforces that condition and no v2 cold run was attempted.

Five tests pass in [`test_projected_skew_scratch.py`](../../tests/test_projected_skew_scratch.py):
exact bilinear response, symmetric potential-field invariance, exclusion of own
curvature, zero-field rest, and real-host inactive identity plus active
RNG/query/moment accounting. An independent read-only review checked the skew
signs, norm bound, cross-block quotient and numerical repair. The test runtime
is about two seconds on one CPU thread. No new variants are justified by these
results alone.

Reproduction uses the environment in the main
[handoff](continuous-learning-handoff.md). Run the warm controller in a fresh
directory; it archives exact source bytes before training:

```bash
python reports/toy100/projected_skew_warm_probe.py --output /tmp/skew-warm
python -m pytest -q tests/test_projected_skew_scratch.py
```

Do not pass the transplant exception flag for the current v2 result: its two
later failures deliberately make that command refuse acquisition.
