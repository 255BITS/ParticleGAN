# G-field counterfactual at the stopped cold ring update

At update 472, the original cold ring critic fit stopped on a nonfinite
L-BFGS trial. Its [exact failure capture](continuous-evidence/pr84-critic-refinement-failure-capture/manifest.json)
contains both the accepted D state and the already evaluated lowest finite
same-bank D fit. I compared those two critics with the analytic, unrestricted
population relativistic-pair critic `log p(x) − log q(x) + C`, using the
frozen current 12-particle Gaussian mixture `q` (sigma `.029`) and the eight
equal real Gaussian modes `p` (sigma `.07`). This is an **offline diagnostic**;
the analytic critic is neither the finite-MLP optimum nor the optimum of the
host's `b_cap`-penalized D objective. The best finite D was never used by
the original stopped run; its G step here is a counterfactual that a guarded
finite-point fallback could take.

The [source](pr84_cold472_field.py) draws exactly eight native G minibatches
from clones of the saved post-D data and global output-noise streams, in the
host order (prior indices, fake output noise, fresh real). At this update,
input noise is zero and output sigma is `.029`. The captured first D bank
gradient equals the saved Adam first moment bitwise. Every G arm uses the
same minibatch tensors and the candidate's **best-finite-D-derived stencil
width `.15`**; D* would independently choose the same width. G/prior and Adam
are restored before each one-step proposal. No D is refit, no persistent outer
update is made, and no random stream is advanced. The [receipt archive](continuous-evidence/pr84-cold472-field/manifest.json)
stores source/capture hashes, runtime, per-particle vectors, batch hashes and
all eight proposal factors.

The pre-step clean particles occupy modes 0, 1 and 7, four particles each;
modes 2–6 have no clean HQ support. The table uses the nearest **currently
missing** mode for each particle (2 or 6) only to grade directions after
they are computed. Counts are across eight batches × 12 particles. “Raw
network” is the clean-output JVP of the negative G-parameter gradient through
the shared network, before Adam. “Accepted” includes the saved Adam moments
and the candidate's own-curvature G bound. A negative distance change means
the accepted one-step support moved closer to its nearest missing center.

| Critic frozen during G | Same-bank sharp D loss / `b_cap` penalty | Local output toward missing | Raw network toward missing | Accepted toward missing | Mean accepted distance change | New clean assignments |
|---|---:|---:|---:|---:|---:|---:|
| Accepted D* | `.283956 / .004972` | 43/96 | 64/96 | 57/96 | `−.00255` | 0/96 |
| Saved best finite D | `.238263 / .000865` | 96/96 | 96/96 | 96/96 | `−.10807` | 0/96 |
| Unregularized population ratio | `6,458,991 / 6,458,991` | 78/96 | 96/96 | 96/96 | `−.02351` | 0/96 |

The population critic's *logistic* part is only `.027583`; the huge total
above is its actual `b_cap` penalty on the archived 1,024-pair bank. Its
raw shared-network direction is large, but the host's G curvature bound
reduces its one-step factors to `.0014–.0024`, compared with `.0776–.1473`
for the best finite penalized critic. Thus the population score is a useful
directional benchmark but a poor numerical surrogate for the actual capped
game. The fitted critic improves the same-bank D objective and supplies a
direction that the shared-network pullback and bounded Adam step do transmit
at this state. The local sampled-output force, raw network force, and
accepted step all point toward the nearest missing center in every paired
case. This argues against an **absent local D signal or blocked network
pullback at this one state**, without establishing that subsequent stochastic
updates maintain a coherent acquisition direction.

Toward-missing projection alone can coexist with attraction to the current
assigned mode. For the saved best fit, 73/96 accepted particle steps also
reduce distance to their original mode; mean assigned-mode distance falls
`0.05367`. None of the 96 one-step particles changes nearest-mode assignment,
and every posthoc fixed-draw grade still has only three modes. This is a
single-state response, not a forecast that those directions accumulate. The
guarded continuation and full cold ring result must decide acquisition.

Reproduce from the committed capture and cold-run archives with one CPU
thread:

```bash
/tmp/pr38-default-env/bin/python reports/toy100/pr84_cold472_field.py \
  --capture reports/toy100/continuous-evidence/pr84-critic-refinement-failure-capture \
  --original reports/toy100/continuous-evidence/round5-critic-refinement-cold \
  --output /tmp/pr84-cold472-field.json
```
