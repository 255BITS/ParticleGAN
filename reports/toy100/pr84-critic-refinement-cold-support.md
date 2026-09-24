# Cold-host support for bounded critic refinement

This is an implementation-support change, **not a cold acquisition result**.
No full cold host was run while developing it. The tested warm adapter remains
byte-for-byte unchanged, SHA256
`cfdf3d050da538e92e79d5a12d18a715c3a8ca7e59a08515014e8e389c598a11`.

[`pr84_critic_refinement_cold.py`](pr84_critic_refinement_cold.py) inherits the
same critic fitting procedure, 40-iteration/80-closure hard budgets, selection
by lowest finite training D loss, accepted-critic placement, fixed G stencil
pair, and once-per-player Adam moments. Its context is
`pr84_critic_refinement_cold(task=..., start_step=0, refinement=True)`.
There is no new gain, objective, quality-based controller or schedule.

The added support is the actual input-noise objective and the trajectory
host's conditioning. Each cached native D batch contains **four independent
input-noise tensors** in original host call order: real logits, fake logits,
real gradient penalty, fake gradient penalty. Reusing a single perturbation
would change that objective. All four are fixed across inner fit closures,
and the gradient penalty still differentiates with respect to the original
data/fast coordinates through the additive perturbation. Original data,
global, input and optional isolated-output streams remain untouched by bank
construction and fitting; policy clocks and counters remain those of the
three original replayed host blocks.

The fixed rule is eight **native** D batches. Ring banks contain 8×128 = 1024
pairs. Trajectory banks contain eight noise realizations of the same 12
paired conditional rows, hence 96 entries, **not 96 independently sampled
targets**. Slow conditioning stays exact and unperturbed. The real D target
is the host's `paired` tensor, including nonidentity pairing; the G cover term
and identity metric remain in the original host and are not used by the fit.
Output noise is formed before D input perturbations, as in both hosts.
Trajectory's one-based penalty step is retained; ring's zero-based loop is
converted to its original one-based penalty call. The adapter is scoped to
the existing CPU models, fixed output scale, Rp logistic loss and constant
autograd `b_cap`, rather than silently supporting other regularizers or models.

With input noise zero, ring bank construction and its loss call delegate to
the frozen warm implementation. A full two-update test proves bitwise equality
of model/Adam/EMA/RNG state, policy receipt and every non-timing fitting record.
This establishes implementation continuity with the warm candidate; it does
not substitute for the required long-hold and cold acquisition gates.

Nine focused tests pass, independently rerun. They cover exact first-bank D
gradient parity under nonzero input noise on both hosts with global and
isolated output RNG; unchanged streams/policy receipts and one Adam moment
update per player; fixed materialized D across G queries; preserved trajectory
stranger pairing; disabled full-host parity; and frozen-warm identity. An
analytic quadratic critic with an active penalty distinguishes the four-noise
objective from incorrectly reusing the logit noises in the penalty, checking
both loss and parameter gradient while keeping conditioning unperturbed.

The additional receipt hashes include the condition tensor and all four
cached noise arrays. Gradient-cost accounting uses the actual bank size,
and separately reports eight clean G forwards per active outer update.
The selected fitted critic may remain nonstationary, and its retained Adam
moments still describe the pre-refinement Adam proposal; those limitations
are unchanged from the warm implementation.
