# Gibbs-inspired round15: joint transitions

User authorizes two adaptive experiment rounds, subagents, config-driven code
changes, both GPUs and the existing queue pipeline. Round16 is selected only
after completed round15 results and diagnostics. No MSE training, full generated
training trajectories, analytic cursor, clipping, EMA, B-cap overrides, or seed
repeats. One fixed particle per trajectory; D-owned M; runtime expert-free.

Build on saved round12 match_shuffle25. Keep the existing point/pair/mismatch
recipe unchanged; add a separate training-only critic K. At clean real-prefix
anchors with t>=4, compare (M_t, x_t, W(M_t,x_t)) with
(M_t, G(z,M_t,t), W(M_t,G(z,M_t,t))). M_t is detached in the new objectives.
Use original noisy real observations and one full generated write per independent
branch. No branch follows another generated branch. Real successor references
are detached for alignment. Default exact public B-cap regularizes K in its
actual candidate coordinates (x,M_next), or M_next for state-only variants.

K has its own optimizer and learns from detached transitions before the D/G
phases. G learns through frozen K and W. Optional writer alignment learns only
the generated write through frozen K; proposal, anchor and real successor are
detached. W retains the original D objectives. New G/writer losses are additive,
with explicit coefficients and a500-step coefficient ramp. K itself trains at
full weight. No renormalization of existing losses. No added runtime module.

First scouts: conditioned joint G-only weights .10/.25; conditioned joint
G+writer weights .10/.10 and .25/.25; unconditioned joint .10/.10 as a process
identity control; conditioned successor-only .10/.10 as an observation shortcut
control. Default K width128, same optimizer defaults/schedule as D.

All fresh2k updates,10k schedule,batch128,eval128 trajectories at256/1024 steps,
warm prefixes8/32. Reuse saved2k baseline and descriptive5k reference. Verify
default-off and critic-only no-op equivalence in tests instead of retraining a
baseline. Test causality, gradient ownership, resume and both-GPU smokes before
launch. Append configs to a fresh durable queue segment because previous queues
are sealed; central train.log remains the existing stable log.

Assess full cold/warm passes first, Q/lateQ/radial/direction/stopping second.
After completion run existing process and information probes plus new local
transition/read diagnostics. Better K classification or memory decoding alone
does not establish a winner. New diagnostics use fresh held-out episodes but
saved learned particles, not held-out particles. Long autonomous paths and
regression probes are evaluation only.

Retain previous extension gates: improve BOTH warm1024 pass rates OR improve Q
at BOTH prefixes >=20%, with nonworsening lateQ, radial <=5% worse, direction
<=2percentage points worse. Both routes require cold late stopping <=1pp worse.
After the adaptive round, at most two qualifying2k runs may be exactly resumed
to5k on the same10k schedule. If none qualifies, do not spend on extensions.
Round16 can pursue mechanistic evidence even if no round15 scout qualifies,
but must clearly record its hypothesis and controls before launch.

Tail: `tail -F runs/memory_path/core_round1/train.log`.
