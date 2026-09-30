# A shared E22 bank at sequential token-routing sites

[`examples/e22_routed_sites.py`](../examples/e22_routed_sites.py) is a complete
caller-owned paired-error training loop with two sequential routing sites.
Both sites use one particle table, one mass vector and one E22 policy. The
second site's queries depend on the first site's output, so a structural
proposal must rerun the full model before measuring its final error.

The example supplies source, time and spatial position conditioning. It runs
two frozen BF16 host layers alongside FP32 encoder, query modules, adapters,
table and critic. It trains RpGAN with KA2 on paired residuals; it evaluates
clean output RMSE on a third disjoint source/time grid. Output MSE is not a
training loss or an acceptance condition.

## Complete model callback

Use the full-model alternative to the [single-site contract](e22_routed.md):

```python
rows = RoutedRows(model_forward=model_forward, features=paired_features,
                  sites=("first", "second"))
```

The callback receives an ephemeral `RoutedExecution` helper:

```python
import math

def model_forward(models, context, candidate, routing):
    G, E, R = models["generator"], models["encoder"], models["router"]
    encoded = E(context)
    first_logits = R.first_query(encoded) @ candidate.table.T / math.sqrt(candidate.table.shape[1])
    first_codes = routing.mix("first", first_logits)
    hidden = G.first(context, encoded, first_codes)
    second_logits = R.second_query(hidden) @ candidate.table.T / math.sqrt(candidate.table.shape[1])
    second_codes = routing.mix("second", second_logits)
    return G.second(hidden, second_codes)
```

The runnable callback obtains every query and host/adapter from `models`;
it does not capture live training modules. Each `mix` call centrally adds
the candidate's `log_mass`, applies softmax and mixes the shared table. Do
not add the mass again in the logits. Both keys and values differentiate
through the same table. The helper optionally applies DV12 independently to
the mixed codes at each site, preserving the native controller's limited
recent perturbation records.

Call each declared site exactly once in the declared order. Missing, repeated
or out-of-order sites raise an error. The helper expires when the full model
call ends; do not retain it or reuse cached training activations for a
counterfactual. Site two must recompute its queries from the candidate-dependent
hidden output. Independent DV12 draws at the two sites retain their sequential
ordering.

## Evidence and guards use the final conditioned model

Every deletion probe and split proposal reruns both sites with the proposed
bank and row state. Its learned feature error comes from the final model
output and the corresponding paired target. The example extracts the critic's
learned token residual features and flattens them into one feature vector per
conditioning context. It does not compare isolated site codes or hypothetical
independent row outputs.

Routing usage averages tokens within each context, then averages the declared
sites with equal weights. Extra spatial tokens or sites do not increase the
effective-context count. The conditional evidence remains a weighted empirical
diagnostic, without the original independent-particle BH or conformal claims.
The [routed law](e22_routed.md#conditional-evidence-and-coupled-moves) describes
its deletion effects, gradient persistence, mass split and per-context bounds.

Fitting contexts drive gradients and deletion diagnostics. Separate guard
contexts protect coupled fast and averaged proposals. A third context grid
measures final clean outputs and never enters the policy. The empirical guards
do not guarantee improvement on that final grid or on every possible
conditioning input.

## Running, restoring and serving

```bash
python -u examples/e22_routed_sites.py --steps 60 --output /tmp/e22-sites.pt
python -u examples/e22_routed_sites.py --steps 2 --resume /tmp/e22-sites.pt
```

The default task uses 8 spatial tokens, a 16×2 bank and a batch of 8 contexts.
Inputs have shape `[batch, tokens, 4]` with columns source-x, source-y, time
and spatial position. Final outputs have shape `[batch, tokens, 2]`. The first
FP32 residual is cast at the next BF16 host layer; the final residual projection
and reported output remain FP32. Frozen parameters retain their BF16 dtype and
exact values throughout training, averaging, serving and recovery.

The lifecycle follows the [ordinary policy hooks](e22.md#caller-owned-updates).
The caller supplies `RoutedBatch` pairs and guards, generates with `sigma=0`
and adds shared noise in normalized paired-error coordinates. Real and fake
receive the same Gaussian base draw; the real reference is detached during
generator training. This caller-owned noise mapping differs from the generic
served helper's optional direct output noise.

The application checkpoint contains policy state, model shape/mode and two
application RNG states: batch selection and the shared paired-error base
stream. Policy state includes its own DV12 streams, row controls, full-model
site contract, controllers, noise, optimizer moments and coherent fast/averaged
weights. Restore at a completed update boundary. A resumed CLI invocation
rebuilds the stored shape and mode before loading.

```python
served = policy.served_model()
prediction = served.routed_forward(source_time_position)
```

This clean default uses frozen copies of both hosts, both query modules,
encoder, adapters and one consistent selected table/mass vector. All sites
use fast weights together or averaged weights together according to E22's
served-choice rule. Training modules remain fast and serving consumes no
training RNG. Optional `perturb=True` enables per-site DV12; `output_noise=True`
adds the stored sigma directly in prediction coordinates.

The fixed tiny CPU run reduced clean final-grid RMSE from **.188005 to
.015629** after 60 actual adversarial updates. All 16 rows received nonzero
gradients each time. The controller accepted four splits (eight moved rows),
rejected 43 guarded proposals, and its active evidence held table descent for
one update. This is a synthetic integration result, rather than a diffusion
model benchmark.

Conformance tests cover downstream query recomputation, full token-output
features, context counts, active evidence and accepted moves, exact
CPU-deserialized resume, and coherent fast/averaged snapshots. The CUDA test
uses actual BF16 host operations with FP32 trainables and replays a naturally
accepted move after `torch.load(..., map_location="cpu", weights_only=True)`.
Exact replay is checked on the same device with serialized backward execution.

## One matched spatial comparison

```bash
python -u examples/e22_routed_sites.py --compare --steps 40 --tokens 128 \
    --z-dim 4 --particles 128 --batch-size 8 --device cuda
```

This shape means a **128×4 bank and 128 spatial tokens**, with four-column
conditioning inputs and two-channel final outputs. It compares three modes:

| Mode | Bank | Row evidence and restructuring |
| --- | --- | --- |
| `frozen` | `requires_grad=False`; adapters and queries still train | Both off |
| `no_rows` | Movable bank | Both off; no row observations, probes or proposals |
| `full` | Movable bank | Conditional evidence and guarded birth/death active |

The script checks identical initial parameters, batch indices and paired
Gaussian base draws, plus the paired and DV12 RNG states after every update.
DV12 makes four ordered draws per update, each shaped `[batch * tokens, z_dim]`:
first/second sites during the critic half, then first/second sites during the
generator half. The paired base stream is separate from DV12 draws, so
controller activity cannot change which base noise the next batch receives.
Learned noise amplitudes and DV12 amplitudes can differ as the three games
evolve. This is one deterministic comparison, not a seed sweep or a search
for settings where `full` wins.

Each mode reports clean final-grid output RMSE, maximum token output error,
accepted moves and training wall-clock time. Timing includes backward,
optimizer work and every controller/probe/guard operation, and synchronizes
CUDA at each measured update. Each comparison arm first runs one untimed
warmup update (`--warmup 1`), then restores its complete initial checkpoint,
module modes and gradients. Model construction, warmup, final validation and
log printing are outside the measured interval. The report includes
`warmup_steps`; `--warmup 0` exposes first-use overhead. Compare the measured quality and cost directly:
passing a learned-feature guard alone establishes neither better output RMSE
nor a faster training loop.

Measured on one NVIDIA RTX A6000 with PyTorch 2.13.0+cu126, 40 updates per
mode, batch size 8 and one restored warmup update:

| Mode | Clean held-out RMSE | Maximum token L2 error | Training wall time | Accepted splits |
| --- | ---: | ---: | ---: | ---: |
| Frozen bank | .078747 | .179697 | 1.326 s | 0 |
| Movable bank, row controls off | **.023803** | **.068836** | 1.974 s | 0 |
| Full `e22_routed` | .031008 | .075403 | 6.064 s | 5 |

All three modes started at clean RMSE .148873 and served fast weights at the
end. Initialization, batches, paired-error base noise and the four per-update
DV12 draw schedules matched. The full mode made 320 deletion probes, accepted
five splits (ten moved rows) and rejected eight guarded proposals. Its evidence
refreshed each update, but no persistent-force flags fired in this short
spatial run; the smaller 60-update conformance task separately exercises a
real evidence hold.

The movable bank without row controls has the lowest output error at this
budget. Full row controls improve on the frozen bank while taking about three
times the movable baseline's training time. This comparison establishes the
measured output quality and cost for one synthetic task; it does not establish
a quality advantage for full row controls over the movable baseline. Keep that
baseline in subsequent Sliders validation. The
[machine-readable receipt](e22_routed_spatial_results.json) records the shapes,
matched-input checks, diagnostics and source hashes used for these numbers.
