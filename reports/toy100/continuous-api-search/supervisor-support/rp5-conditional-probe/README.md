# RP5 conditional precision probe — unapplied proposal

This independent scratch snapshot changes only `ReversiblePrecision.observe` and adds one CPU contract-test file. It starts from immutable RP5 source ZIP SHA256 `43eaad25848e65605afc08b39ff2b714277e0d792315dcb5f5b6f49a25263df7`. It does not include the earlier LR-helper patch. The active package remains byte-identical to that baseline.

Proposed signature:

```python
observe(critic, real, activity, *, probe=None)
```

For a conditional critic, bind the current context once per observation:

```python
precision.observe(
    critic, real_fast, accepted_activity,
    probe=lambda module, x: module(slow_context, x),
)
```

The callback is invoked twice, first with the live critic and then with the controller's reference critic. Both receive separate differentiable clones of the same real tensor. The same callback/context applies to both. It must evaluate the **supplied module**, preserve differentiation with respect to the provided data tensor, and avoid mutating modules, input or context. Return the ordinary critic output; tuple/list outputs retain the existing first-element selection. Context should remain fixed for that observation, and condition coordinates should not be concatenated into the differentiated data unless that is the explicitly intended gradient domain.

`probe` and bound context are caller-owned per-observation inputs. They are not stored or serialized. The caller must reconstruct the appropriate context/callback after checkpoint reload and own any external context state. A closure bound permanently to the live critic would violate this contract. The extension does not select or aggregate multiple roles/critics, change precision's reference update, modify thresholds, change secant/optimizer behavior, or automatically bind GANTrainer custom tasks.

With `probe=None`, the original critic calls remain. Static AST comparison confirms the default observation body is identical after removing the optional dispatch/doc/validation. The complete reference-update block and every later controller method remain source-identical. A separate proof executes the old method loaded directly from the immutable ZIP and the proposed default method on identical tiny CPU inputs: **44 ATen operations per observation, every operation output byte, and full controller/reference state match across three observations**.

Validation: **5 CPU tests passed**. They cover default/direct-probe equality, nonlinear analytic conditional gradient-gap equality, current context applied to both live/reference across observations, preservation of caller tensors and existing gradients, noncallable rejection, and exact checkpoint continuation with a rebound context. No optimizer step, training loop, GPU execution or quality experiment ran. `git apply --check` passes against a separate pristine snapshot. This is integration preparation, not qualification evidence.

Files: `rp5-conditional-probe.patch`, reusable `snapshot/`, `baseline.json`, `results.json`, `cpu-contract-results.xml`, and `default_path_proof.py`/`default-path-proof.json`. Reproduction commands are recorded in results.json. No active-worker or research files were edited.
