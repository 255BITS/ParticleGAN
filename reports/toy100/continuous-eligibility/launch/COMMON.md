# Objective and rules for all three approaches

Make ParticleGAN able to point at a target and keep learning indefinitely,
without a user switching acquisition and maintenance phases. This is a public
API implementation task with measured experiments. Work from the supplied
PR195 commit in your isolated checkout, using actual `get_recipe()` and
`GANTrainer.step()`. A frozen research-loop success alone cannot qualify it.

The user accepts constant learning rates OR autonomous reversible rate control.
An adaptive policy must lower rates when useful, raise them again when learning
is needed, and regain stability. It must discover this from ordinary training
signals; no target-shift notification, target centers, labels, quality metrics,
known convergence time, planned ending, caller reset, or manual phase change.
The full learner must be independent of the evaluator's horizon, including
learning rates, noise, critic memory/mixing, particle updates and API stopping
behavior. A positive terminal LR floor does not make a scheduled decay eligible.
Fixed absolute initialization counts or smoothing constants are not themselves
disqualifying: explain why any retained initialization rule works at arbitrary
training ages and does not require a user-operated phase switch.

Read the read-only evidence paths supplied below, then act in your own checkout.
Read the eligibility report's retained scores as research leads: disqualified
configurations may contain useful ideas. Preserve their measured strengths and
failures, address the exact disqualification, and re-earn the descendant's own
scores. Disqualification is not a request to erase evidence or avoid the idea.
Retain the adversarial learner, trainable particle prior and general-purpose
API. No target fitting, post-hoc sample translation, per-toy policy, seed sweep,
coefficient grid or borrowed passes. A focused mechanism may change schedules,
noise or update rules; this explicitly supersedes the historical driver's
instruction to preserve inherited schedules. Do not weaken evaluator scoring
or quality thresholds; separately declare the required longer, delayed and
repeated-change protocols. Keep failed variants and their actual results.

The existing public API evidence is decisive background, not a new baseline to
repeat unchanged: constant KA2 retains 61/120 pre-shift checks and reaches the
shifted ring after 120 updates, but fails 83 post-arrival observations. Decayed
KA2 retains 120/120 and arrives after 1,690 updates. KA2 surprise rises both on
real change and its own collapse. Neither variant is an eligible winner.

Use the existing public ring worker as the starting evaluator, preserving its
default dimensions (20,000 particles, latent dimension 2, batch size 2,048),
architecture, initialization, data streams, quality definition and isolated
observations. Its planned step budget and inherited noise schedule must not
become inputs to a new continuous learner. Declare and snapshot each candidate
and evaluation protocol before training, log actual applied rates/noise and
controller state, and hash source, initialization and artifacts. Preserve CPU
initialization followed by CUDA training; do not silently substitute the tiny
research-host configuration or a new Adam implementation.

Recovery means time to the new distribution followed by stability. There is
NO fixed 81/81 or 400-update deadline gate. Report first arrival, every later
departure, passing/total checks since arrival, final stable suffix, minimum
quality, and pre-change retention. The unchanged observation definition is all
eight modes and HQ >= 0.90. A favorable endpoint or short suffix cannot hide
earlier collapse. Historical reports with 81/81 language are source evidence,
not current promotion rules; old requests for seed variants are superseded.

For each meaningful mechanism first measure cold acquisition plus retention
and a target change through the actual API. Use one existing seed, then adapt
the mechanism only from measured failures. A failed early mechanism does not
need expensive full qualification. For a survivor, use the SAME unchanged
policy for a 7,500-update stationary run, delayed and repeated target changes,
and a longer continuation (30,000 updates). These are finite evaluator windows,
never learner schedules, and cannot prove literal infinite stability. Declare
target-change times only in the data/evaluator and report all transitions.
Keep a frozen no-update comparator when establishing adaptation. Test an
identical training prefix under differing evaluator budgets and exact checkpoint
continuation including policy, optimizer, EMA and RNG state.

Before proposing a release winner, compare K3P on the same runtime/protocol and
complete the candidate's own broader 22-task quality checks. Historical K3P
numbers on a different runtime are context only. Coordinate any expensive
shared comparator with the local supervisor so three lanes do not duplicate
it; if unavailable, say NOT_RUN. Never combine passes from multiple variants.
Label research-host-only or incomplete API evidence explicitly. A successful
agent exit is not qualification, and you may not merge or publish a winner.

At most three coherent proposals per attempt, one GPU worker and one CPU
thread. This cap creates a review point, not an overall search budget: the
supervisor reviews evidence, updates briefs, and replenishes idle lanes until
a qualified solution or user stop. No nested agents or extra model sessions.
No hard timeout or dollar/token budget was requested. Start a real candidate
test promptly using the retained worker; avoid rebuilding the benchmark.
Save concise `result.md` and `tests.jsonl` incrementally; make logs easy to tail.

If a previous attempt is listed below, inspect its mechanism, results and
failures before proposing a successor. Never restart an unchanged failure or
random seed merely because a fresh external Codex session has started. A
follow-up may copy a declared predecessor's local changes into this isolated
API checkout, with an exact source hash and diff. Source is not evidence of
qualification; the resulting candidate must earn its own results.
