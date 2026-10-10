# Atlas/E22 image parameter capacity

The original source-transpose12 intensity2 and bars4 hosts have **SUPPORTED
gate-tolerance capacity** under both public policies. These four cells carry
**zero ordinary GAN qualification credit**: they restore previously supervised
reference generator parameters and construct explicit learned-prior positions,
with zero new fitting, backward or optimizer updates.

The unchanged public sampler matters. The exact repeated reference atoms fail
DV12: intensity HQ .7001953125, finite-template TV .2998046875 and one detected
mode; bars HQ .63671875, TV .36328125 and three detected modes. These failed
construction observations remain in the external
`toy-policy-image-capacity-20261002-json-recovery/receipt.json` archive. They are
pre-shape, zero-update capacity diagnostics, not a representation impossibility
or a failed ordinary training attempt. Its earlier import/export preflight errors
and partial saved arrays also remain outside Git.

The successful construction preserves the generator, all original mode atoms
and balanced 32-row mass. It offsets coordinate 7 of row `i` by
`(i - 15.5) * 2^-20`. This parameter coordinate was zero in every reference row.
The actual nearby same-mode supports bound DV12 perturbations through its native
nearest nonzero-neighbor radius. No sampler, recipe, controller, threshold or seed
changes are used. Each fixture rebuilds its public GANTrainer after restoring
parameters, so both fast/EMA models and the controller geometry derive from the
same constructed state.

| Public policy / source host | HQ | Modes | Finite-template TV | Mean RMSE |
| --- | ---: | ---: | ---: | ---: |
| Atlas / intensity2 | 1 | 2 | .0029296875 | .000523163 |
| E22 / intensity2 | 1 | 2 | .0029296875 | .000523163 |
| Atlas / bars4 | 1 | 4 | .0146484375 | .000478913 |
| E22 / bars4 | 1 | 4 | .0146484375 | .000478913 |

Each cell uses its original 600-update execution horizon at **completed step 0**,
1024 actual served samples and the runner's fixed evaluation seed 34002. One
public `begin_step` on the caller's actual first real batch followed by
`abort_step` selects the backend. That prelude retains controller/reservoir
metadata observations; abort does not undo them. It performs no completed update,
forward/backward or optimizer step. Atlas selects `knn/controller_reference` with
reason `finite_resolution_infeasible` for 32 rows (one possible guard flag versus
38 required BH flags). E22 retains its reference DV12 path. This certificate
does not certify a feature-cell sampling law.

Full fixture state, first real batch and actual served samples are retained in
`/ml2/hypergan/toy-policy-image-capacity-neighbors-20261002`. Receipt SHA256 is
`e138a5db3d8c841f6555970cdee04610394760edf81a2bc906db97240f32cc99`.
It binds exact case/preset/sampling/source bytes, reference state hashes,
controller/serving metadata and raw state/sample hashes. Observation preserves
complete fixture/RNG state, and both optimizer state dictionaries remain empty.
Oracle passes; collapsed output, global mean and unequal-mass controls fail.

Base-family reuse is limited to this round's learning-rate/prior-rate-only grid:
those rate fields leave the zero-clock sampling law, actual model/prior and
selected backend unchanged. Different hosts, priors, controller state, sampling,
recipe semantics or source bytes require new compatible evidence. The single
capacity observation does not establish training acquisition, the required
terminal suffix, full-protocol convergence, robustness or default adoption.

```sh
python reports/forge/family-winner-round1/policy_image_capacity.py \
  --archive /path/to/retained/image-representation \
  --output /tmp/policy-image-capacity \
  --local-neighbors --lifecycle-prelude
```

The [helper](policy_image_capacity.py) is a report-only construction observer.
The combined policy-capacity report owns admission of the complete required
eight-question suite; these image observations cannot fill other host cells.

The later `api-stress-*` override guard changes protected API contract bytes from
`0c79eae1...` to `2d23aacfb6107f1535ed04ea0d5bb67d8f33a260ac1aca9e1d1734bfaf0f2e28`.
The [compatibility observer](policy_image_compatibility.py) restores all four
exact saved full-horizon zero-update states through public `load_state_dict`
and verifies current actual served targets/samples **bitwise equal** and all
metrics unchanged. Current bindings and original bindings remain separate.
No fitting, training, new parameter candidates or raw-file writes occur.
The new receipt at
`/ml2/hypergan/toy-policy-image-compatibility-json-recovery-20261002/receipt.json`
has SHA256 `d94830ecb9138f4999eb07d035a874dadf003412e2e814f1fb7e9451685244a3`.
The original `e138a5db...` receipt and every original state/sample digest remain
unchanged. A recipe tuple-versus-serialized-list preflight rejection is retained
separately; it failed before any sampler call.

After PR251 publishes WordFixture component/stream declarations in the shared
image provider, a final selected-image compatibility addendum binds provider
SHA256 `243190aba0c5870668a63289b771dc6f784c9460b890bc6a77e12cc2796d7184`
and the unchanged `2d23aacf...` contract guard. All four actual CPU restored
samplers again produce bitwise identical targets/samples and unchanged original
metrics, with complete fixture/RNG purity and raw byte identities preserved.
This four-cell replay takes 2.762 seconds; no training or new lifecycle prelude
occurs. Final addendum path is
`/ml2/hypergan/toy-policy-image-compatibility-pr251-20261002/receipt.json`,
SHA256 `18421efd0e624c1d245c6e46e1898a460875d3f66b0f8f004381a1b392c8b9a2`.
The earlier source-bound receipts remain intact. Compatibility covers the
declared guard/provider publication only and does not qualify WordFixture or
ordinary learned training on the new source.
