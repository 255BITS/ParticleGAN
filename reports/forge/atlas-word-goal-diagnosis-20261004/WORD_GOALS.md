# Retained N11 word goals and two prospective rate contrasts

The encoder was updated, but the saved run did not learn the complete inverse
question. This is not evidence that its rate vanished. The original family
remains **INVALID**, its certified numerical gate remains **UNAVAILABLE**, and
its 558.5739127129782 paid seconds remain charged. The separate audit-health
patch recognizes a source-defined dimension-skip sentinel; it does not repair
any of the word measurements or authorize an unchanged replay.

This report reads only the original `f380eed9…` source, its full 20,001-update
checkpoint dictionary, recorded scalar observations, and saved arrays. No
model was constructed or restored, no forward/sample/scorer was called, and
no CUDA context was created. All 43 consumed inputs were hashed before and
after inspection without changes. Full identities are in `analysis.json`.

## What the optimization and inverse measurement actually do

The original scaffolds are G `2→64→128→168` with a 28-character softmax at each
of six positions, free continuous E `168→128→64→2`, and joint D
`170→256→128→1`. The task has five canonical inputs including underscore
padding and eleven actual uniformly selected prior rows. It is the explicitly
named `atlas_word_joint_min11` / `word_joint_policy_min11_v1` adaptation, not
the original blocked N5 law or independent Atlas qualification.

The caller performs one D and one combined G/E/table update per step. During D's
update E and the generated joint are detached. During the combined update D's
parameters are frozen, but D remains differentiable with respect to its inputs.
E receives gradients through the real-joint term `(word,E(word))`; G and the
table receive gradients through the fake joint. The loss is exactly
`mean(softplus(D(real_joint)-D(fake_joint)))` plus `prior_reg×spread(table)`.
Here `prior_reg=0`, so the spread contribution is zero. There is **no**
reconstruction MSE/NLL training term, no supervised row-to-word assignment, and
no direct G(E(word)) gradient path. The Recipe's `reconstruction_weight=1`
does not add a loss that this caller never uses. These are source facts, not
proof the joint objective is wrong.

The generated joint carries the same effective DV12 code used by G. Learned
training output noise touches only the 168 word coordinates. The real joint
uses the one-hot word and unperturbed E(word), with input noise zero. At
evaluation the observer selects G/E/prior together and computes E on all five
known canonical inputs, then calls the actual selected public generator.
Additive output noise is off; DV12 still perturbs both prior draws and the
encoded reconstruction queries. All 24 selected snapshots were `fast`.

Consequently the inverse measurement is a stochastic public-serving question.
An ideal deterministic, noiseless BiGAN inverse argument does not by itself
prove that this noisy joint law meets all numerical inverse gates. It is also
not unseen-word generalization. Neither observation is a reason to weaken the
question.

Source locations in the frozen snapshot:

- `word_joint_policy_adapters.py:105` owns separate G/E/table groups and public
  UpdatePolicy; `:142` defines the loss; `:153` implements ordered updates;
  `:262` evaluates selected G/E/prior and retains actual effective codes.
- `api_images.py:1020`, `:1030`, `:1040` define the original three networks.
- `gan_loss.py:6` defines the paired logistic joint loss.
- `definition_quality.py:102` defines the original probability and inverse
  measurements, including padding.

## A zero all-five flag is not zero individual reconstructions

`reconstruction_exact` is the Boolean equality of the entire five-by-six
argmax token array with the canonical array. It is not a percentage or a count
of individually recovered words. Its value is zero at all 24 observations,
and the separate minimum correct-token probability is far below .90 throughout.

| Actual saved step | Generated modes | Recorded mass TV | Individual paired argmax words recovered | Reconstruction illustration |
| --- | --- | --- | --- | --- |
| 834 | 2 | .6 | 2/5 | grape and lemon correct; apple→grape, melon→lemon |
| 10,001 | 5 | .2169921875 | 1/5 | melon correct; apple/grape effectively swapped; others→melon |
| 15,835 | 1 | .8 | 1/5 | every reconstruction is grape |
| 20,001 | 2 | .6 | 2/5 | grape and berry correct; apple→`berre_`, lemon/melon→berry |

The endpoint quality fraction is .9248046875 versus required .95; three target
words have no accepted generated mass. Paired minimum token probability is
zero. Finite reconstruction NLL 11.05240844637142 is compatible with this:
the original diagnostic clips probability at 1e-12 before taking logs.
Confident wrong tokens and exact zero correct-token probabilities are retained
softmax outputs, not NaN model tensors. No new numerical grade was computed.

## What supports a dynamics hypothesis, and what is missing

All G, E, prior and D parameter step counters are 20,001. At the endpoint the
G/E/table/noise stationarity scales are all one. G and E group rates are
.0053125; table rate is .00796875. E's saved first-moment norm is .0526097 and
G's is .00154545. These are nonzero optimizer memories, not a measured
per-update displacement history. They exclude an omitted terminal E owner and
a permanently zero terminal E rate; they do not establish healthy E updates at
every earlier boundary.

D's saved applied rate is .0006453535110926051, .121478 of its .0053125 nominal
rate. Its stationarity scale is one, while public payoff damping acts separately.
The saved post-generator payoff error is 2.6933488648504684. The applied D rate
used the preceding payoff state, so the post-generator damping formula is not
an exact reconstruction of that already-applied rate. Endpoint loss G is
2.2668192386627197, loss D .2700580358505249 including penalty .0102450475.
This shows an unresolved joint game rather than an equilibrium; the two
updates use different fake draws, so these losses are not a paired causal
comparison.

Saved code geometry changes materially. From step 834 to 10,001 the two prior
axis standard deviations shrink from (.526297,1.174961) to (.214468,.133191),
then reach (.404529,.602379) at the endpoint. All 24 saved prior tables retain
eleven distinct rows. Five distinct encoded queries nevertheless produce
only one reconstructed word at 15,835. Thus neither a zero-learning-rate
encoder nor complete exact collapse of all latent rows explains the available
observations. Saturated wrong-token regions, imperfect inverse coupling,
moving latent support and critic balance remain competing explanations.

The terminal history contains 1,397 actual table moves/rebases and 8,431
row-evidence hold steps; surprise has six recorded fires, including 19,998.
These events can couple changes to support, averages and optimizer state.
They are not an identified cause of the sampled failures. There is no complete
per-update G/E/prior/D displacement or loss trace, earlier complete checkpoint,
or paired reconstruction with DV12 off on the same selected state. The stored
source-defined dimension skip explains the health rejection separately.

The scaffolds are not an obvious dimensional impossibility for five finite
words. Even deterministic eleven-row allocation (2,2,2,2,3) has population
mass TV .072727…, below .1. That arithmetic gives no certificate under the
actual DV12/noisy training and held-out sampling law; current full-law
representational capacity is unverified.

## At most two proposed rate-only tests

Neither candidate is executed, admitted, a predicted repair, or a default.
One contrast may be selected without committing to both.

| Proposed tuple (lr, prior multiplier, D multiplier) | Single declared change | Question/falsifier |
| --- | --- | --- |
| (.001328125, 1.5, 1) | quarter the shared base rate | Does lower nominal G/E/D/table/noise motion preserve five-mode coverage and paired token confidence? Continued poor inverse goals would weaken a simple oversized-step explanation. |
| (.0053125, .15, 1) | table nominal rate ten times slower | Does G/E recover the inverse while latent support moves more slowly? Continued inverse failure would weaken a nominal row-chasing explanation. |

The first preserves nominal role ratios, not identical endogenous trajectories.
The second leaves G/E/D/noise nominal rates unchanged, but birth/death and DV12
geometry can still change row dynamics. No mechanism is disabled. Compiled
memory contains slower-prior KA2 positives from separate original-word
diagnostics with different source/family/serving bindings. They are motivation
only, not capacity, success, or causal credit for this N11/DV12 family.

Each prospective attempt retains the original seed-zero named initialization
and streams, original architecture/objective, N11/free E, same-code DV12,
words-only learned training noise, state-selected observation law, 20,001
updates, 24 observations of 1,024 draws, exact original bounds and terminal
five checks. A new source-bound declaration/resolver is necessary: the frozen
word resolver accepts only the old C6 pair and would reject either proposed
tuple. It must not be bypassed. Fresh health/implementation/source identities
must remain separate from the old INVALID result.

The total proposed new GPU1 allowance is at most **1,800 seconds**, at most two
900-second complete single attempts, with no extra grace, retry, passing-case
rerun or truncated-budget success. The old named cumulative cost 910.239143
seconds and GPU1 cost 675.413062 seconds are retained once as the root-reported
prior accounting, not reset or added twice. Root owns the exact spec, source
freeze, admission and execution. This proposal supplies no ordinary/default,
speed or independent-Atlas credit.

`analysis.json` pins each original source, the resolved task/Recipe, checkpoint,
raw/grade/study and all 24 NPZ files. `inspect_retained.py` is a portable passive
reader, and `handoff.json` pins this report and that reader.
