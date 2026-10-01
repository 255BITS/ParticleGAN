# Current PR155 E22 versus ParticleGAN Atlas

These are actual learned-model runs of current PR155 E22 at `cabe2084`:
the original Toy25 and MNIST fixtures, seed314159,1,024 latent particles,
latent width128, batch128, and2,000 updates. Initial generator/critic/table
weights, data batches,10 metric checkpoints, noisy primary sampling,
evaluators and thresholds are unchanged. E22 retains optimizer reopening
and critic-anchor release. The comparison is against Atlas's previously
completed original learned runs, with their current source/replay bridge.

| Original final metric | Current PR155 E22 | Atlas |
| --- | ---: | ---: |
| Toy noisy-sample precision, higher is better |71.5454%|96.5332%|
| Toy covered modes |25/25|25/25|
| Toy mass total variation, lower is better |0.284546|0.052114|
| Toy original acceptance |FAIL|PASS|
| MNIST active embedding Fréchet distance, lower is better |1.880969|0.544488|
| MNIST embedding precision |76.7578%|86.9141%|
| MNIST embedding recall |71.9238%|84.7168%|
| MNIST confident class coverage |10/10|10/10|

Toy precision increases24.99 percentage points; MNIST embedding precision
increases10.16 points and recall12.79 points. MNIST has no numerical
acceptance threshold in the original protocol. Its backend is kNN for both
recipes; this result does not demonstrate image-scale feature cells.

E22's MNIST detector records one reopen event, `[202,3.984]`, during early
training on the stationary target. Atlas records no fire. Toy records no
fires for either recipe. The exact event counter is the completed-update
cursor recorded by the detector; its action begins the following update.
The common optimizer-noise/control state and all10 learning-rate/score
records are preserved in the machine-readable comparison.

Recorded training time is89.46s for E22 and161.57s for Atlas on Toy;
106.99s and91.05s on MNIST. These are timings of separate runs using the
same physical GPU/cap/runtime policy. They do not establish a universal
speed claim or a controlled latency advantage. Toy quality is higher at the
common update budget while its Atlas run takes longer.

## Formulation and use

Current E22 already trains the latent particle table along with G, controls
rates from optimizer history, redistributes particles using nearest-neighbor
critic-feature evidence, tests support, learns output noise, uses an averaged
served model and reopens after an optimizer shock. These are shared controls.

Atlas explicitly adds automatic capability selection, regional count/support
tests, bounded real-anchor births, conditional output-moment mean repair,
population-aware table settling, bounded local sampling and a geometry check
on averaged serving. Its settled-game guard qualifies optimizer reopening
with an earlier contracted network rate and starts a new detector reference
when the applied critic penalty changes phase. Caller-owned shapes, models,
initialization, particle count, latent width, batch size and data remain explicit.

For eligible standard models with at most8 raw output coordinates and enough
particles, a temporary map is fitted from recent real critic features.
Generated outputs are tested in that same map. Cells are regions in this
learned representation; they are distinct from latent particles and output
samples. They need not coincide with true classes or modes. Counts and support
use critic features; group placement also uses the complete low-dimensional
raw output frame. Auto-selected cells apply an empirical one-quarter factor
to configured generator/noise base rates. Table/critic bases keep their input
values; actual rates remain controlled during training.

After an actual optimizer-shock fire, mean repair can act on the occupied
subset of frozen averaged-output groups. Missing groups have zero direction
and weight, without redistributing their mass. All calibration observations
remain in the witness. Count/birth controls handle missing support. This is
an Atlas recovery refinement; E22 has no feature-cell mean phase.

The added controls offer explicit decisions about regional mass, group means,
supported births and averaged serving. Their measured quality depends on the
recorded fixture and configuration. Higher-dimensional outputs and custom
representation/routed ownership retain existing controls; the settled guard
can still qualify reopening on these routes. Automatic feature selection
uses capabilities, without task names or quality scores. Ordinary Recipe and
named E22 defaults remain unchanged. Compatibility repairs preserve repeated
references across checkpoint transfer, explicit CPU planning, and validation
of initialized output shapes before mutation.

## Scope and provenance

The recipe comparison changes the full selected formulation; it does not
identify one cause for the Toy improvement. The MNIST path uses the same
neighbor backend and original base rates. Results cover one fixed seed and
these original architectures/budgets. They establish neither scaling laws nor
general superiority across tasks. E22 already passed its recorded portability,
static native and moving gates; these results cannot support a claim that it
lacked drift recovery or those prior passes.

The E22 package is20 exact Git modules from PR155 at
`cabe2084284db923d525918cbf3e18de6f20faac`, raw SHA
`c174ba0b805cc8e49ca40ebbea11785bb908f3ea32615afdf2ecb43e77221316`.
Its upstream JSON SHA is
`03a8c7a8d35512a4e06735c37c375895f6ecc24011c4411a606eddd92f4c202c`;
all resolved training fields match its named E22 preset, allowing only the
`ka2`/`e22` report name. Atlas uses the already-qualified source raw500ff
and configa3ee5. Its fresh learned executions retain their historical labels;
no fresh Atlas training was added or claimed here.

`comparison-closure/COMPARISON.json` and its119 pinned inputs record
actual timestamps, completions, endpoint and all10 checkpoint metrics,
learning rates, noise scales, detector events and the source identities.
`comparison-closure/FROZEN.json` closes their hashes. Historical E22 runs
with reopening disabled and all failed intermediate candidates remain
separate immutable records. No GPU/model/scorer calls were made by the closer.
