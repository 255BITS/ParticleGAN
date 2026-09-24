# What a critic-refinement failure would mean

The frozen [`pr84_critic_refinement.py`](pr84_critic_refinement.py) source
(SHA-256 `cfdf3d050da538e92e79d5a12d18a715c3a8ca7e59a08515014e8e389c598a11`)
first completes the original bounded D Adam proposal. It then chooses the
lowest finite **training-bank** loss from one bounded L-BFGS fit of the same
sharp Rp logistic plus `b_cap` objective, with G/prior fixed. The eight cloned
128-pair batches do not advance training RNG; the first batch's D gradient is
checked bitwise against the host gradient. G's two evaluations see the
materialized fitted critic and the same frozen stencil width. D Adam moments
still describe the proposal *before* refinement. This is an empirical extra
D solve, not a certified best response or a decay-free cold-start recipe:
active fitting currently requires zero input noise and the mode-hold host.

The [one-dimensional capped toy](capped-critic-tracking-toy.md) makes a
specific mechanism plausible. At a population critic optimum with unequal
real/fake variances, the generator's own fixed-D mean curvature is negative
but the field evaluated along a fully responding critic is restoring. It does
not establish that an MLP fitted on 1024 fixed pairs tracks a population
optimum. The actual warm receipt in
`artifacts/continuous-learning/round5/critic-refinement-warm/forks/refinement.json`
records 200/200 passing local checks, 10,674 fit gradient evaluations, and
10,930,176 fit sample-pairs evaluated. Mean fit cost is 53.37 closures and
0.595 seconds per active update on one CPU thread; the whole warm fork took
131.3 seconds versus 11.7 seconds for the original fork, both including the
same 1000-update prefix. This cost is material before extending to a cold or
production host.

## One saved-state diagnostic if a later hold fails

Capture the first failing update and one immediately preceding passing update
with exact pre-D, accepted-D, pre-G, and accepted-G model, prior, optimizer,
noise and RNG states. Check the original replay parity before interpreting
counterfactuals. At each state, hold G and its output-noise scale fixed and
construct three nonoverlapping 1024-pair banks from cloned streams:

* **A:** the adapter's actual eight-batch fit bank, including its bitwise
  original D first batch; reuse the materialized fit `D_A`.
* **B:** the next eight draws; make one independently selected fit `D_B` from
  the *same accepted-D starting point*, with the same cap, budget and
  minimum-training-loss rule.
* **C:** a further eight draws, used only for evaluation. It must not select
  either critic or any hyperparameter.

On C, record the sharp penalized D loss and eight paired minibatch loss
differences for accepted D, `D_A`, and `D_B`; raw and saved-metric D gradient
residuals; cap activity; and critic logit differences. For a common G
counterfactual, use C's eight D-style draws as paired G inputs and the
**same** frozen stencil width for all three D
variants and record both their G/prior *partial gradients* and cloned-Adam
clean-output proposals. This avoids conflating a changed critic with a changed
stencil or moment state. The known ring centers may grade the resulting clean
outputs afterward, but must not fit or select D.

Interpretation is deliberately asymmetric. If `D_A` and `D_B` both lower C
loss and yield concordant G functional directions while the saved critic
does not, critic lag is supported. If each fit improves its own bank but not
C, the 1024-pair fit is sample-sensitive; if C improves yet `D_A` and `D_B`
produce opposing G directions, an empirical D objective decrease is not a
reliable G signal. If fit residuals remain large, bank sensitivity cannot be
separated from numerical inner-solve error, so do not call either fit an
optimum. If both fits agree and still give a harmful G proposal, spending
more closures on the same critic objective alone lacks support; inspect the
G loss, shared-network pullback and optimizer metric instead. Retained D Adam
second moments should be compared with post-refinement fresh gradients before
blaming critic lag in the next outer step.

If A/B agree but the next-step failure still suggests delayed opponent
response, one conditional extension uses **one actual accepted G step**, not
an epsilon grid: regenerate A's fake samples at its endpoint with fixed
latent indices and output-noise tensors, refit D once at that endpoint, then
compare the G *partial field* there with D frozen versus D refitted. Project
both secants onto the recorded G step. This measures a finite-step opponent
response, not a total derivative or a Hessian theorem. Its result is
inconclusive when either fit is nonconverged or the finite step crosses a
different critic basin. The bank-swap stage costs one extra bounded fit
(roughly 0.6–0.9 seconds in the three previous saved-state fits) plus heldout
fields; the conditional endpoint stage costs one more fit, with no live
training mutation, new seed or target-center controller.

The reusable [three-bank script](pr84_critic_bank_swap.py) was first checked
on the **original PR84** saved state at update 1325, where A's fit was already
archived. This is a diagnostic sanity check, not an outcome for the live
refinement candidate. One new B fit (58 closures) used no new seed. The saved
critic's untouched C loss is `.659397`; A and B reduce it to `.625343` and
`.620506`, with lower penalized loss on each of C's eight batches for both
fits. Across C's eight counterfactual batches, A/B G partial-gradient
cosines are `.966–.978`, and their accepted clean-output proposal cosines are
`.969–.982`. The saved critic's partial gradients point mostly opposite
either fitted critic (`−.64` to `−.39` cosine). Posthoc, both fits yield
8/8 HQ passes on those batches, versus 6/8 for the saved critic. All exact
first-D-gradient, A/B-bank, original unbounded/bounded G-proposal, saved-state
and RNG controls pass. A central finite loss secant along the actual accepted
G step agrees with the G partial directional derivative within `4.93e-5`
(about `0.28%`).

The C raw D-gradient infinity norm **rises** from `.03294` for the saved
critic to `.07232/.05559` for A/B despite lower C loss. Thus the result
supports reproducible *improved guidance at this state*, not convergence of
either empirical D fit, a population optimum, or sustained stability. No
endpoint-refit response test was run; the cheaper bank-swap already answered
its first sampling-noise question. The [receipt and source archive](continuous-evidence/pr84-critic-bank-swap-1325/manifest.json)
bind this one state and all source hashes.

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES='' \
  /tmp/pr38-default-env/bin/python reports/toy100/pr84_critic_bank_swap.py \
  --states /path/to/capture-v2/selected-states.pt --step 1325 \
  --archived-a reports/toy100/continuous-evidence/pr84-critic-relaxation \
  --expect-original-g-parity --output /tmp/new-bank-swap
```

## Which gradient is actually updated

Let \(\theta\) denote G/prior parameters, \(\phi\) critic parameters,
\(d=\nabla_\phi L_D\), and \(g=\nabla_\theta L_G\). The host update uses
\(g(\theta,\phi)\), holding its current critic fixed. If a differentiable
local critic response \(\phi^*(\theta)\) with invertible D Hessian existed,
the *field* derivative along that response would be

\[
\frac{d}{d\theta}g(\theta,\phi^*(\theta))
=\nabla_{\theta}g-\nabla_{\phi}g
  (\nabla_{\phi}d)^{-1}\nabla_{\theta}d.
\]

This differs from the **total gradient of the scalar composite**
\(L_G(\theta,\phi^*(\theta))\), which also contains
\((\nabla_\phi L_G)^T d\phi^*/d\theta\). In this host,
\(L_G\ne-L_D\): G uses non-saturating Rp logistic and a spatial stencil,
while D alone pays the slope cap. The usual minimax envelope simplification
therefore does not remove that total-gradient term. Calling extra D fitting
"Stackelberg-gradient descent" would misdescribe the implemented update.

[Fiez, Chasnov and Ratliff (ICML 2020)](https://proceedings.mlr.press/v119/fiez20a.html)
derive implicit-response learning dynamics for zero-sum and general-sum games;
their distinction motivates measuring the response term, but our adapter
does not implement their total-gradient dynamics.
[Zeng and Doan (COLT 2024)](https://proceedings.mlr.press/v247/zeng24a.html)
analyze two-timescale stochastic methods whose lower-level root operator is
strongly monotone; the frozen nonlinear MLP critic has no such verified
property. [Lin, Jin and Jordan (ICML 2020)](https://proceedings.mlr.press/v119/lin20a.html)
analyze a single nonconvex-concave minimax objective, whereas this host has
two different losses and a D-only cap. None of these theorems certifies the
current fit or the full 12-particle host.
