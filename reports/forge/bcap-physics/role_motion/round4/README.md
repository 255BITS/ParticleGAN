# Round4: generator/prior finite motion attribution

One global role-aware control is preregistered against the exact winning BCAP
configuration. This is a bounded diagnostic comparison, with unchanged full
gates, and supplies no ordinary Tier2 qualification or default promotion.

The [five saved-source probes](saved-role-probes.json) restore original winner
and transport-v2 endpoint checkpoints using their exact archived sources. One
next-batch update is previewed on each disposable copy, then the complete model,
optimizer and consumed-stream state is restored exactly. The native network
output RMS is .245671, prior RMS .036805, joint RMS .268698. Across all five
probes, network RMS is4.4–6.8times prior RMS and network jitter deformation is
larger. These are finite next-update measurements, not the original preceding
training step, full density grades or causal explanations of every failure.
Posthoc target component/core/spill labels only describe diagnostics.

## Mechanism and falsifier

Let z=c+epsilon be the already consumed generator-phase latent batch, and let
(theta1,c1) be the actual ordinary momentum0 DualNorm proposal from(theta0,c0).
Measure p=RMS[G_theta0(c1+epsilon)-G_theta0(c0+epsilon)] and
g(s)=RMS[G_(theta0+s*(theta1-theta0))(c0+epsilon)-G_theta0(c0+epsilon)].
Retain the whole c1 update. Use s=1 if g(1)<=p; otherwise start at p/g(1)
and halve up to24times until the actual finite g(s)<=p. A zero response or
exhausted search restores G exactly. Unchanged s=1 never recomposes floating
parameters. The public `GANTrainer(role_motion_balance=True)` owns this path.
The prior remains learned with fixed task width/uniform weights, and receives
its full ordinary optimizer proposal. All original losses/rates are unchanged.

The constraint balances two degrees of freedom using the observed composition
G(c), without labels, target sigma, evaluator centers, extra batches or draws.
It measures center travel, within-kernel finite deformation and nonlinear cross
interaction separately. It bounds only network motion on consumed rows; the
joint nonlinear response and served density remain unconstrained. A small prior
gradient can suppress useful G progress. The cap can trade overshoot for stalled
learning or contraction. No density, curvature or convergence guarantee follows.

[Bolte, Sabach and Teboulle's PALM](https://bolte.perso.math.cnrs.fr/BST2013.pdf)
treats coupled variable blocks with block-specific proximal steps under explicit
minimization assumptions. It motivates distinguishing roles, but this normalized
alternating GAN rule is not PALM and does not inherit that theorem. The rule's
finite measured inequality is an implementation property; task fidelity is an
experimental question.

Two ready schema-v3 studies freeze one candidate and the exact winner as its
matched control on the same source/runtime: two-pole, Gaussian smoke and each
arm's own eligible stability continuation, nativegrid100, unequal-width and the
passing broad-vector guardrail. Candidate two-pole is explicitly unsupported
because its frozen public-component host lacks this controller. Both original
task/gate definitions remain unchanged. Smoke failure blocks its own dependent
continuation; other diagnostic peers still complete. Seed0, public deterministic
initializer, architecture/data/prior/sampling/update limits and scoring cadence
remain matched. The frozen vector/Gaussian adapters actually reuse one real
tensor for D/G; that law is preserved and disclosed.

Preregistered native forecast: precision>=.48; precision<.30 falsifies it.
Unequal-width complete full-covariance/suffix gates and broad PASS are additional
required review outcomes; endpoint bounds do not replace sustained grading.
Exactly one comparison is authorized; no seed runs, rate search, continuation,
automatic second candidate or unchanged merge rerun follows this study.

## Budget and execution

The full main ceiling is16440seconds(8220per arm). Five completed saved probes
plus one pretraining sampler refusal reserve720seconds; meaningful tiny public
API software checks reserve120seconds, leaving total17280 below the18000track
ceiling. These diagnostic updates never fill scientific grades. The initial
vector probe supplied a CUDA generator to the CPU sampler; it refused before
any update. The corrected probe binds the original checkpointed CPU stream.

The independent public Queue/drain runner uses one freely shared GPU worker,
with the full-compilation completion callback disabled. All arms are pushed
before enqueue. Bulk logs, JSONL streams and checkpoints remain outside Git.
Tail execution with:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/role_motion/logs/driver.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/role_motion/queue/role-motion-round4-v1/progress.jsonl
```

Use `/home/martyn/dev/ParticleGAN/.venv/bin/python` with `PYTHONPATH=.` from this
worktree. `run.py plan|enqueue|drain --queue-root QUEUE` reproduces the declared
workflow. The ready declarations are under `configs/forge/studies/role-motion-*`.
The control-side unsupported-cell study admission follows the earlier
transport-v2 fix, retaining known API refusals and every original task identity.
Actual-training GIFs and compact certified final metrics will be published after
the frozen runs complete; saved probes are contextual, not a third matched arm.
