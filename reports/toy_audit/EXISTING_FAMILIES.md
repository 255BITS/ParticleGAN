# Additional shipped problem families

The main catalog executes the declared transfer suite and new open-PR problem
proposals. This appendix prevents other shipped synthetic families from being
mistaken for those tasks. These rows are **source/evidence reviews**, not fresh
training or current Atlas qualification. Configuration/algorithm variants share
their underlying problem; seed/config counts are not independent families.
Real-image datasets and Gym environments are outside the synthetic toy catalog.

| Rating | Shipped family | Intended verification and limits | Source / existing evidence |
|---|---|---|---|
| 4 | Sparse mixed real/categorical, identity-symbol law | Match class-conditional modes, active-coordinate noise, **exact inactive zeros**, and the deterministic class symbol. Small W1 alone can hide leakage onto inactive coordinates. | [sampler](../../lib/sparse_toy.py), [metrics](../../lib/sparse_metrics.py), [original problem report](../sparse-conditional-with-categorical-and-real-output.md) |
| 4 | Sparse mixed, split-symbol law | Each class has two equiprobable symbols; distinguish matching the full conditional symbol distribution from always returning one valid symbol, while preserving continuous/symbol consistency. | Same sampler's `symbol_map='split'`; shared underlying support, different conditional categorical law. |
| 4 | Gaussian-grid denoising, one/four classes | Recover the analytic multimodal posterior `q(x0\|xt,c)`, not just a posterior mean; verify class consistency and repeated conditional draws against the exact mixture oracle. One-class and checkerboard four-class conditioning are distinct contracts. | [analytic posterior](../../lib/denoising_toy.py), [trainer](../../experiments/train_denoising.py) |
| 4 | Conditional two-route trajectories, discrete/continuous geometry | Match class-dependent upper-route probability (0.8/0.3), continuous route variation, endpoints and obstacle clearance on held-out geometry. Collision scoring examines segments between frames; route IDs are not model inputs. | [sampler/scorer](../../lib/trajectory.py), [existing reports](../trajectory) |
| 4 | Analytic route transitions | Generate a **joint** `(state, action, next_state)` conditional law with `next=state+action`, correct route probabilities and held-out geometry. Matching the three marginals separately does not verify their relationship; shuffled-block controls matter. | [definition](../../lib/transition.py), [existing leaderboard](../transition/leaderboard/README.md) |
| 4 | Paired 2D affine and radius-dependent swirl | Learn the specified input-output transport on distinct train/validation/test rows; paired correspondence distinguishes a correct mapping from matching only output marginals. Fixed/movable clouds are model controls, not extra data problems. Evaluation MSE is separate from the adversarial training objective. | [protocol and targets](../../benchmarks/paired_error_2d/task.py), [runner](../../benchmarks/paired_error_2d/run.py) |
| 3 | Ring-8 acquisition, hold, warm/cold continuation and shift | Acquire all modes and retain them over an uninterrupted own-state continuation; distinguish target adaptation from an old favorable checkpoint. Initialization and hold/shift protocol matter, but dozens of controller/objective PRs reuse this family. | [continuous probe](../../benchmarks/toy100/continuous_probe.py), [ring host](../../benchmarks/locked_shared/mode_hold.py) |
| 3 | YuE2-named paired sign/kinematic lander | Show that sign-flipping a symmetric `(state, action)` joint law can preserve marginal/joint distribution while breaking control, whereas paired-error alignment fixes the correspondence. Learns a scalar sign around a supplied expert law. It does **not** run YuE2 or demonstrate general controller learning. | [explicit scope and gate](../../lib/yue2_particle_toy.py) |
| 3 | Safe-fast kinematic lander | Compare landing safety and speed for a one-parameter sink controller, with adversarial identity plus rollout cost versus declared controls. A useful objective-composition fixture, not a learned full policy or real LunarLander qualification. | [definition/gates](../../lib/safe_fast_landing.py) |
| 3 | Native paired-action 2D particle fixture | Verify the declared paired-action/reference and particle/controller plumbing on a small constructed host. Do not transfer its result to full physical control. | [experiment](../../experiments/toy_particle_native_2d.py) |
| 3 | E22 routed 16-site/pair, support/width, moving and replay fixtures | Verify the named dense-bank routing, paired residual law, support/width behavior and own-state replay contracts. They are component fixtures, distinct from native independent-row Atlas. | [routed paired](../../examples/e22_routed_paired.py), [moving](../../examples/e22_routed_moving.py), [replay](../../examples/e22_routed_replay.py), [support](../../examples/e22_routed_support.py) |
| 2 | Five-word autoencoder/latent demo | Recover the five named words and reconstruction through a tiny representation; deterministic finite support. Attractive latent plots do not establish text generation or unseen-word generalization. | [example](../../examples/five_modes.py) |
| 2 | Single Gaussian quickstart | Demonstrate public training/sample plumbing and a basic unimodal law; too little multimodal/conditional structure to be a broad quality benchmark. | [quickstart](../../examples/quickstart_gan.py) |

The [no-particle grid example](../../examples/100gaussians_no_particle_prior.py)
is a model control on the already catalogued grid100 law, not a new problem.

No fresh training GIF is claimed for these source-reviewed families. Existing
endpoint viewers and historical scores should remain labelled with their actual
recipe, budget and sampling law. Current-host replay/visualization gaps remain
explicit; this appendix is not a current positive-reference certification.
