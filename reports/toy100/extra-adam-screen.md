# Joint ExtraAdam: bounded constant-rate probe

**All 24 ExtraAdam rows and all eight simultaneous-Adam controls failed sustained mode-hold.** There were no harness errors. Each attempted episode completed the frozen 1,200 outer updates and 24 live evaluation checkpoints. Trajectory and all broader gates were skipped, so there is no full-19, native-three, shared-22, continuation, or shift-test claim.

The [32-row leaderboard](extra-adam-screen.json) contains every exact configuration hash and terminal curve. The [grid declaration](../../configs/toy100/extra_adam_probes.json) describes the shared candidate changes. The training source epoch is `4b00335`, with [implementation](extra_adam_scratch.py), [driver/regrader](extra_adam_probe.py), and [numerical tests](../../tests/test_extra_adam_scratch.py). No production code changed.

## Exact method

[Gidel et al., Algorithm 4, option 1](https://arxiv.org/html/1802.10551v5) updates Adam moments at both gradient evaluations. For outer iterate `x`, the implementation computes a joint game gradient at `x`, updates moments and bias correction using count `2t−1`, and forms `y = x − η u_first`. It evaluates a new joint gradient at `y`, advances moments and bias correction to `2t`, then returns **`x_new = x − η u_second`**. The final step starts from `x`; the lookahead moments remain. This practical Adam method has no convergence theorem in that paper.

No additional lookahead coefficient was introduced: the same constant group rate is used at both evaluations. No iterate average determines the reported output. D's parameter update is deferred until G's gradient is captured, so both gradients refer to the same point. Saved D gradients are restored before applying Adam, avoiding contamination from G's backward pass. The simultaneous-Adam control uses one joint gradient and one ordinary Adam update.

The frozen host functions are adapted by wrapping only their existing gradient block in a one- or two-evaluation loop. Noise clocks, initialization, models, auxiliary losses, EMA updates and evaluation remain outside that loop. Existing samplers execute at each evaluation: stochastic minibatches and training noise are redrawn, while deterministic full-batch inputs stay fixed. The generated function is archived; a structural inverse must recover the original archived function exactly. A rehashed extra assignment is explicitly rejected by tests.

## Declared comparisons and cost

ExtraAdam crossed two shared cores, four LRs {.0005, .001, .0025, .00425}, and β₂ {.9, .99, .999}. Simultaneous-Adam controls crossed both cores, LR {.001, .0025}, and β₂ {.99, .999}. β₁=0, D multiplier 1, and prior multiplier 2 stayed fixed. The old core used κ=1.176/coefficient 6/prior regularization .05; the simple core used 1/1/0. Every row retained the .029 output-noise mechanism and existing input-noise burn-in. All schedule multipliers were exactly 1, with the network horizon cap/floor removed.

ExtraAdam used **2,400 gradient evaluations per player** per episode; simultaneous Adam used **1,200 per player**. Their measured case times summed to 767.85 and 126.84 seconds, respectively. They share the frozen outer-update budget, not a matched gradient-computation budget. The first declared row served as a complete harness smoke and was reused only after a new independent regrade.

| Candidate | Method / core | LR; β₂ | Final modes / HQ | Terminal passing suffix |
| --- | --- | --- | --- | ---: |
| `eg028` | Simultaneous Adam / simple | .001; .99 | 8 / .9983 | 3, requires 5 |
| `eg029` | Simultaneous Adam / simple | .001; .999 | 8 / .8535 | 0 |
| `eg012` | ExtraAdam / simple | .0005; .9 | 5 / .1697 | 0 |
| `eg016` | ExtraAdam / simple | .001; .99 | 4 / .3281 | 0 |

`eg028` had terminal `(modes,HQ)` values **(5,.6328), (6,.7473), (8,.9534), (8,.9990), (8,.9983)**. Its final three checkpoints look promising, but the required five-check suffix is absent. ExtraAdam's best final coverage was only five modes. This tested second-gradient method gave no numerical basis for a nearby expansion. The simultaneous controls also show that the poor ExtraAdam result cannot simply be equated with every use of joint-gradient updates.

## Evidence

All **32 raw episodes independently regraded after durable relocation**, including:

- The frozen source archive, original task specifications, exact configurations and unchanged live thresholds.
- The archived generated-function inverse against the original host's AST.
- Every G/D/prior rate, group size, Adam β, and moment counter; 2,400 or 1,200 updates as declared.
- Verified same-point gradient capture and, for ExtraAdam, 1,200 restorations of the original weights for final updates.
- Actual gradient RMS and parameter displacement from the outer base at every gradient evaluation.
- Continued rejection of the scratch episode by the production common gate.

**99 tests passed**, including the independent bilinear formula, exact moment counts, deferred-update checks, transform invariance and tamper rejection, and the existing common-gate tests.

The archive is `artifacts/toy100-constraints/particlegan-extra-adam-wave1-4b00335` in the ExtraAdam research worktree. All **487 original RAM files** matched their durable copies byte-for-byte. Manifest SHA-256: `d18ce72e85a78f0bc275693be1abda4212bbb474ee8351f828433aee9c1cd83e`. Final inventory SHA-256: `fdc2a574516787902980a755c3d5fd3cd881a2884c49d892f932bc79be2d9117`. The earlier `5eb8b90` declaration contains no training episodes and is retained separately as `particlegan-extra-adam-declaration-only-5eb8b90`.

Regrade with `extra_adam_probe.py regrade --root <durable archive>`. To reproduce training, check out `4b00335`, run `prepare` against a new root, then `run`, using the documented CPU/AVX2 environment. This probe changes the optimization method and its computation cost; its results do not establish that constant-rate learning is impossible.

The next finite comparison changes a different assumption: persistent discriminator input noise. The previous input noise disappears after 10% of each budget, so these experiments have not tested sustained input smoothing with ordinary constant-rate Adam.
