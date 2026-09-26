# Separate source issue: projection RNG and global default device

The immutable DV7 `particlegan/continuous.py:42–44` creates a CPU generator and passes it to `torch.randn` without an explicit device, before converting the result with `.to(x)`:

```python
stream = torch.Generator(device="cpu").manual_seed(1729)
self.projection = (torch.randn(x.shape[1], 32, generator=stream)
                   / math.sqrt(x.shape[1])).to(x)
```

A host that sets the global default device to CUDA would request a CUDA allocation with a CPU generator. The later `.to(x)` cannot repair that allocation mismatch. An isolated fix would bind the random allocation explicitly to CPU, preserving the intended CPU draw followed by conversion. This is a source finding; no CUDA reproduction was run.

This issue is deliberately unpatched here. It is independent of component LR binding and does not establish a failure in a host that keeps the default device on CPU while moving model/data tensors explicitly. A future detector patch should preserve the fixed local stream, projection values under the ordinary CPU default, and training RNG isolation. Review the newly active candidate's own immutable source before transferring a fix.
