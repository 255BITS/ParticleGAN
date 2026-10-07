Ring16's SVD update assigns unit strength to almost-null gradient directions.
This adds an experiment-only random sign rule on singular components below the
float32 numerical rank threshold, with independent fair `±1` signs from an
isolated checkpointed CUDA generator. Strong signs stay `+1`; biases and prior
updates remain unchanged. Exact-zero components are explicitly treated as
perturbations of arbitrary normalized completions.

Two fresh public-API arms compare an intervention at update 401 only with flips
on every update. Architecture, target batches, prior, seed, constant rates and
full quality bounds are fixed. The protocol reserves 3,200 updates/600 seconds
plus a zero-update 30-second saved-gradient probe. Smoke requires a full passing
state and independent same-state confirmation; five-terminal quality and tier-2
retention remain separate. Production defaults and qualification are unchanged.

Validation covers syntax, frozen bindings, static RNG usage and unavailable-CUDA
guards. GPU training, algebra, candidate-training GIFs and retention remain
unmeasured because this host has no available CUDA device. Remote publication
awaits network access. See the
[prospective report](reports/forge/ring16-sign-flips/README.md) and verification
receipt for the exact scope and execution commands.
