# PR245 CI orchestration correction

The first [CI attempt](https://github.com/255BITS/ParticleGAN/actions/runs/37021898314) at `737eb5e2fe4b717a9a0db630134cea840c31f248` failed in the new renderer test. Its [Python 3.12 job](https://github.com/255BITS/ParticleGAN/actions/runs/37021898314/job/110886784094) installed and validated dependencies successfully, then reported **3,438 passed, one failed, 72 skipped, one xfailed and 18 subtests passed in 776.05 seconds**. The failed case was `test_complete_trace_hash_phase_counts_use_actual_labels`. Python 3.10 and 3.11 passed their assigned wheel and smoke checks; those jobs did not run pytest. Release distribution building was skipped after the Python 3.12 failure.

The renderer sets `STARTED = time.monotonic()` at [source line 7](../examples/render_e22_routed_caption_late_phase.py#L7). Pytest imports that module during collection at [test line 8](../tests/test_e22_routed_caption_late_phase_renderer.py#L8). Earlier tests consume more than 60 seconds before the case calls `read_traces()` at line 70. The helper's `budget()` call at renderer line 128 therefore raises the deadline error at lines 28–29. This is a deadline lifecycle error when the CLI helper is imported into a long-lived test process, rather than a dependency-installation failure or a failed scientific metric. The sole completed standalone render remains the recorded 2.392-second execution.

The CI-only correction partitions the existing required Python 3.12 tests into two processes:

```sh
python -m pytest -q --ignore=tests/test_e22_routed_caption_late_phase_renderer.py
timeout --signal=INT --kill-after=5s 60s python -m pytest -q tests/test_e22_routed_caption_late_phase_renderer.py
```

The second invocation starts the unchanged renderer's deadline in a fresh process. All three renderer tests remain required, alongside every test in the first invocation. Neither step has `continue-on-error`; the 25-minute job limit and the renderer's 60-second limit remain unchanged. This correction edits only CI orchestration and adds this note. No benchmark, renderer, test, card, preparation record, result, media or scientific gate is changed. No test, science, model, API or rendering execution was performed while preparing the correction; the subsequent CI result is pending.

All 15 card-pinned source and test files retain their declared hashes. Key unchanged identities are:

| Artifact | SHA-256 |
| --- | --- |
| Main benchmark | `f4bad83b5ff9944a14c75bcf5d51d82e26fd3ed432a0730122255fffdc313b65` |
| Renderer | `61fd56ad58bdbe6b76fe292f57251533e6beef0d493d05dea62124fbe0e0823f` |
| Renderer tests | `d99a095adaf81043704a01d401e503cd68fe133151f353751c1e77c7a27b236a` |
| Frozen protocol | `740f96f96764026cda84f728986e9920007c96a9b948ecc0edbbe0a9a5b05b6b` |
| Independent scientific review | `e4dca76594f437c0e9f9039990b743675e545fd7a4125652cd9f77906f624be1` |
| Results JSON | `64aaff08df789972997fba5372c732df30ffc3664309a9414c4bc60f4798f16d` |

The first job's unmodified log body is retained locally at `/tmp/particle-pr245-python312-first-attempt.log`, SHA-256 `546f9110a87aea6aea40ecba8f18f9bb6db5582e94224b5f604eb9dc11fd1b52`, 56,946 bytes. It is not committed, and contains no request headers or redirect URL. The failed first attempt remains part of the evidence; this correction does not replace or requalify the original scientific results.
