# Running the tests

The default run is CPU only and never touches a GPU:

```sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m pytest -q
```

- **CPU threads.** A session fixture in `tests/conftest.py` pins
  `torch.set_num_threads` to `OMP_NUM_THREADS`/`MKL_NUM_THREADS` (the smaller
  one when both are set). It also restores that count after any test that
  changes it. Without either variable, torch keeps its own default.
- **CUDA tests are opt-in.** Every CUDA-only test carries `@pytest.mark.cuda`,
  and `pyproject.toml` deselects that marker by default (`-m 'not cuda'`).
  This keeps a workstation run from silently competing with queued GPU
  experiments. To run them, name a GPU and select the marker:

  ```sh
  CUDA_VISIBLE_DEVICES=0 python -m pytest -q -m cuda
  CUDA_VISIBLE_DEVICES=0 python -m pytest -q -m cuda tests/test_dualnorm_smoothing.py
  ```

  A later `-m` on the command line replaces the default selection.
  `python -m pytest -m "cuda or not cuda"` runs everything. A new CUDA-only
  test must carry the marker: `tests/test_suite_policy.py` fails any test
  file that skips on `not torch.cuda.is_available()` without it.
- **CI.** `.github/workflows/tests.yml` runs the whole suite in four
  deterministic file shards on Python 3.12, and shard 0 of the same four on
  Python 3.11. Only the complete 3.12 shard set feeds the release coverage
  receipt check (`scripts/ci_pytest.py check`).
- **End-to-end tests** (`tests/e2e/`) exercise the production Forge view
  validator, the public-API toy contract and the Git index on the real
  configuration. They contain no test-local copy of any contract.
