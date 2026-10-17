# Inert native scorer source fixture

`native100_score.py.txt` is an exact, 1,660-byte copy of the original
`/ml2/hypergan/lrfree-20260926/harness/native100_score.py` SOURCE. Its SHA256 is
`10cc14edfcd98ab34fd3768aaba2ee835dc2241dc1face2e18998c8f2b687feb`.
The source was read once to create this fixture; tests verify its size and hash
and never require that external path. The `.txt` fixture is never imported or
scored directly. Synthetic wrapper controls combine its unmodified source with
the prospective bootstrap and fake hosts, then stop before scoring. There are
no retained numerical inputs, checkpoint, model, or qualification results here.

This test-only successor preserves the frozen `c7b9bc46` helper and its source
hash `86549af57aee025b32ec1126f4703a74957a296f6a5fc56f9887d00dda24739d`.
It does not change any runtime scorer fingerprint or scientific protocol.

The portable structural subset is:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q reports/forge/pr223-original-full-retest-20261004/test_native_scorer_imports.py reports/forge/pr223-original-full-retest-20261004/test_run_retest.py -k 'native or import_source or import_guard or derived_wrapper or original_scorer or copied_preflight_uses or matching_copied or canonical_normalization or fresh_pythonpath'
```

It retains all 38 bootstrap checks and adds three fixture integrity controls.
The selected 13 older controls now read only repository source/metadata and
private synthetic files. Historical plan/parity controls that require archived
external inputs are outside this subset and are neither weakened nor rerun.
The v2 handoff records an additional run of this exact subset with original
archive opens denied, including the original external scorer in generated
fake-host child processes. This establishes test portability and no numerical
credit.
