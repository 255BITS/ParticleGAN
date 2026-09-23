# Replay verifier mutation audit

All nine cases pass after the two verifier corrections. No GAN training ran.
The valid replay and timing-only changes are accepted. Empty or partial comparison
maps, nonboolean truthy flags, altered numerical observations, empty source
manifests, mismatched reference hashes, and absence of replay evidence are rejected.

Run against a checkout containing the retained control replay:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python audit.py /path/to/ParticleGAN
```

The script copies the real control archive into temporary directory fixtures and
executes a byte-for-byte copy of its verifier; the checkout is read-only. It
recomputes raw payload hashes after intentional mutations so comparison failures
cannot be attributed merely to stale hashes.

- `results.json`: final nine-case results and exact verifier SHA256.
- `verifier-under-test.py`, `replay-helper-under-test.py`: tested validation source.
- `results-before-source-fix.json`: retained failing audit showing the empty-source
  manifest gap after the comparison-key issue had already been fixed.
- `audit-before-source-fix.py` and corresponding log: original executed harness.

The final harness additionally accepts a repository path for portable invocation.
Only the parent-owned validator changed; all archived training evidence remains
unchanged.
