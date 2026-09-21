# One transition at a time

Open index.html in a browser. It is self-contained, works offline, and includes
both encoder models, both classes, all saved train/test scenes and time slices,
real-input prediction, the architecture diagram, and results/limitations.

The controls change independent saved transition samples, not a rollout. Blue
arrows are physical displacement; orange points are generated next states; red
connectors expose consistency error. Plot axes match across comparison panels.
The enlarged view is explicitly labeled. The gray circle shows the reference
obstacle geometry. Gray points optionally show reference states for comparison.

Share index.html alone, or this bundle. No server, external libraries or private
data are required. toy_transitions.png / .pdf show one disclosed interpolation
case; use the viewer to inspect all cases, especially the extrapolation geometry.
architecture.png / .svg / .pdf explain the model. Nothing has been published.

Suggested description:

“We tried learning a distribution of individual state/action/next-state triples
with three generators sharing a 1,024-component MoG prior. An encoder maps
(state, action) back into that latent space, allowing a synthetic G1/G2 → E → G3
path. Sharing a time-conditioned state discriminator improves joint SW1 by 14.6%
on our fixed route toy benchmark. Coverage and off-distribution action response
remain weak; this adds paired supervision and is not yet a general world model.”

Rebuild from repo root:
.venv/bin/python experiments/render_transition_architecture.py
.venv/bin/python experiments/render_transition_demo.py
