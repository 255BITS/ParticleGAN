# Two persistent memories; local adversarial training

Md has32 coordinates and belongs to D. Mg has8 coordinates and belongs to G.
The64-coordinate decoder feature is temporary, not a third recurrent memory.
The proposal adapter is also stateless. All six recurrent scouts deny G access
to Md, including through the proposal adapter. The baseline clock scout retains
the original shared-memory formulation.

For a real prefix, before target t:

```python
Md, Mg = zeros_D(), zeros_G()
for x in real_prefix:
    Mg = G.state_cell(G.observation_encoder(x), Mg)
    Md = D.writer(Md, x)
```

A generation read and explicit state transition:

```python
x, features = G.read(z_fixed, Mg, clock_t)   # final read; no persistent writes
u = {"embedded": G.observation_encoder(x),
     "intent": features,
     "hybrid": (features + G.observation_encoder(x)) / 2}[mode]
Mg_next = G.state_cell(u, Mg)
```

D scores real/fake candidates against the same pre-candidate Md and optional
clock_t. Candidate scores never silently advance either memory. Default exact
B-cap differentiates candidate coordinates with conditions cached. The optional
clock is the existing six-band Fourier embedding; it is a supplied condition,
not a target of a clock-classification/regression objective.

The independent local pair branch generates x_t, advances each memory once,
then generates x_(t+1). D judges both coordinates against the original prefix
Md. Gradients from the second output train G's state transition. In intent mode
that credit passes through internal features rather than through x_t. Only
this bounded generated transition is trained; there is no third temporal output.

Point feedback is a separate branch, never chained to the pair branch. Md consumes
(1-a)*real_point+a*generated_point. Mg's shared GRU consumes
(1-a)*encoded_real+a*u. The existing ramp caps a at.25. Pair training and runtime
use the full generated transition. Real-prefix BPTT trains G's observation encoder
and recurrent cell; D phase constructs G state without G gradients. G phase
freezes D parameters while preserving the required loss derivatives.

The existing mismatch loss trains D to recognize wrong-history candidates.
No new auxiliary objective, MSE training, clipping, EMA or B-cap override is added.
G's state encoder and decoder must learn a compatible GRU input representation
through GAN losses; there is no imposed equality between observed/internal features.
That interface is a remaining hypothesis, not an established stable state model.

Runtime after optional real-prefix initialization only needs G, fixed z and clock.
Evaluation still updates Md for instrumentation, but no generated output depends
on it in the six separated variants. State zero/shuffle interventions and probes
operate on stored Mg or read copies as documented in each diagnostic.

UCD consideration: particlegan/conditioning.py currently selects discrete class
or joint time/class logits and uses D-only CE on real/fake logits. Absolute clock
is not generally identifiable from a random-phase circle sample. Memory might
reveal elapsed prefix length, which is not the same as identifying process dynamics.
We therefore test ordinary D clock conditioning, without adding UCD time targets.
A continuous score-conditioning construction could be explored separately, but is
not implemented or claimed equivalent to the repository's UCD API in this round.
