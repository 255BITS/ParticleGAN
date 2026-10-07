Ring16's ordinary matrix polar update assigns unit weight to almost-null
singular directions. PR331 found that a `1.03e-7` relative gradient perturbation
at update 401 becomes a `.252` relative direction difference, while a restart
changes the later quality result.

This prospective diagnostic adds an experiment-scoped smooth damping rule,
`s / hypot(s, max(shape)*float32_epsilon*sigma_max)`, with two frozen schedules:
update 401 only after a fresh live 400 prefix, or every update from initialization.
Both retain the selected BCAP recipe, constant rates, public initializer, prior,
batch, sampling law and full numerical bounds. Candidates never load the
archived 400 checkpoint. A separate zero-update CUDA probe measures sensitivity
on the original saved gradients.

No CUDA experiment ran: the current host has no usable CUDA device. Static
verification covers syntax, source bindings, plan validation and refusal to
start CPU training. The finite allowance is 3,200 updates/600 training seconds,
plus one 30-second saved-gradient probe, with no retries. Production optimizer
defaults and qualifications remain unchanged. Actual-training GIFs and numeric
candidate outcomes are pending CUDA execution; GitHub publication is pending
network access.

Read `reports/forge/ring16-damping/README.md` for the hypothesis, provenance,
falsifiers, commands and distinction between smoke acquisition and tier2 hold.
