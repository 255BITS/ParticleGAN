# Independent continuation and clock-state review

Reviewed the public checkpoint/extension path, native continuation adapter,
clock-free probe, state identity, independent graders, prerequisite payloads and
planner compatibility identities. No full experiment or seed sweep ran. Checks
used isolated configuration fixtures, a 24-update tiny public-trainer probe,
a two-update prefix plus two-update continuation and existing short unit tests.
Runtime fixes remain owned by the coordinator.

## Findings

| Finding | Severity | Evidence and required correction | Status |
|---|---|---|---|
| Native continuation rejected by the generic checkpoint reducer | Blocking integration defect | `views.qualify` originally accepted only matching uninterrupted run IDs. Native continuation supplies verified prefix/checkpoint artifacts, so a valid child PASS was overwritten with BLOCKED. The native branch must bind its verified proof to the selected parent's checkpoint, manifest, key and attempt. | Fixed and verified: dedicated native branch binds the selected parent; reducer regression passes. |
| Unbound clock-audit recipe could conceal a delayed guard | Blocking proof defect | A real tiny probe with `d_guard_min_steps=200` graded BLOCKED. Changing only `comparisons.pt`'s top-level recipe to `0`, refreshing its manifest and source-audit declaration, graded PASS while actual saved trajectories/checkpoint recipes still contained `200`. Bind audit recipe/extensions to the measured initial state and require each trajectory's exact formulation except the declared horizon perturbation. | Fixed and verified: formulation binding and refreshed-manifest adversarial regression pass. |
| Continuation resets the elapsed-time axis | Blocking integration defect | `_native` copies parent `events.jsonl` but originally appended suffix events using only the new run's elapsed time. The frozen native gate rejects decreasing elapsed time. A long prefix followed by its first short suffix interval therefore becomes INVALID independent of quality. Preserve the original events and add their final elapsed offset to newly measured suffix elapsed time. | Fixed and verified: appended events include the prefix elapsed offset; few-step monotonicity and parity regression passes. |
| Clock and continuation checks trusted completed-step labels without checking Adam counters | Blocking proof gap | Clock trajectories can copy the initial optimizer/model state, change only labels and horizon metadata, and obtain equal comparison hashes. Verify measured optimizer counters at the initial warmup and every probe state; the label-offset branch must still make exactly the same optimizer updates. Prefix/final native checkpoints need the same measured-counter binding. | Fixed and verified: both Adam state tables require actual scalar counters at the expected update; forged clock counters are INVALID. |
| Dependency identity omitted parent compute identity; execution identity omitted actual thread count | Compatibility defect | In an isolated planner fixture, changing a GPU parent's `resources.gpus` from `1` to `0` changed the parent's key but left its CPU child's key unchanged. Changing an independent task's `cpu_threads` from `1` to `2` also left its key unchanged, although the worker consumes that setting. Bind prerequisite scientific keys/compute and actual execution thread count. | Fixed and verified: recursive full prerequisite keys and actual task thread counts change downstream identities; planning regressions pass. |
| Structurally different state objects had identical state digests | Blocking identity defect | The concrete example below collides because tensor metadata is encoded as an ordinary tuple, followed by bytes which can encode another ordinary item. Add a distinct tensor tag and unambiguous container framing before hashing. | Fixed and verified: distinct tensor/container/scalar tags, container lengths and hash schema prefix remove the demonstrated collision. |
| Native final checkpoint leaves static formulation and RNG metadata unbound | Blocking proof gap | After a real two-update prefix and two-update continuation, independently changing final `trainer.recipe.lr`, `extensions`, or `streams.manifest.seed`, then refreshing checkpoint and outer manifest hashes, was accepted by `verify_native_prefix`. The retained original prefix was unchanged. Bind final static context fields, trainer recipe and RNG manifest to the original prefix, and validate duplicated trainer/named RNG states. | Fixed and verified: static fields, named RNG manifest and duplicated RNG states are bound; altered or dropped trainer RNG entries are rejected. |

The structural collision used:

```python
payload = json.dumps(["int", 7]).encode()
a = [torch.tensor(list(payload), dtype=torch.uint8)]
b = [("tensor", "torch.uint8", (len(payload),)), 7]
assert state_digest(a) == state_digest(b)  # true before the framing fix
```

This is a serialization collision, not a SHA-256 collision. An identity helper
must distinguish these different structures before invoking the hash function.

## Positive checks and limits

`GANTrainer.extend_execution` changes only the execution allowance and requires
an increasing positive integer. It retains the original recipe horizon,
parameters, optimizer moments, EMA and RNG state.

The native adapter loads a public context checkpoint before extending that
allowance. Its retained prefix includes the original full artifact manifest,
checkpoint, diagnostic files and fixed target reference. Named data streams
are restored with the context. The independent prefix verifier reads actual
checkpoint tensors and checks the restored state, rather than accepting two
caller-provided hash strings alone. The existing few-step test compares resumed
and uninterrupted final checkpoint identities; that is useful evidence for the
implementation path, not a claim of 7k/14k scientific quality.

The clock probe compares complete model/EMA/optimizer state and named training
streams at every short prefix point. It genuinely changes external step labels,
horizon, evaluation cadence and restart path. Evaluation streams are separated
from training streams. Source auditing conservatively keeps delayed critic
guards and unreviewed extensions blocked even when a short observed prefix
matches; the formulation and counter checks above are required to make that
audit bind to the recorded execution.

The queue carries a concrete prerequisite row, attempt ID, candidate revision,
compatibility key and result hash to the child. The runtime dispatch consumes
that payload. These identities must remain bound when the view reducer selects
the parent, and compatibility must include the parent's compute cohort.

## Regression verification

The following targeted suite passed **133 tests in 14.29 seconds** after the
first six fixes, and **133 tests in 15.22 seconds** after the native static-field
and RNG-consistency fix:

```sh
python -m pytest -q tests/test_forge_adapters.py tests/test_forge_clockfree.py tests/test_forge_planning.py tests/test_forge_queue.py tests/test_forge_views.py tests/test_forge_adaptation.py tests/test_forge_api.py
```

After the final dropped-RNG completeness guard, the native continuation parity
and adversarial mutation test passed again (**1 test in 5.25 seconds**):

```sh
python -m pytest -q tests/test_forge_adapters.py::test_native_continuation_restores_own_prefix_and_matches_uninterrupted_state
```

All seven reproduced findings are fixed with their targeted checks passing.
No production adoption, long-run adaptation, clock-free candidate qualification
or 14k success follows from this implementation review.
