# PR140 / PR143 CPU reproduction audit

**The saved acquisition and continuation results reproduce. The earlier local
failure is explained by MKL CPU-vendor dispatch, not the CPU-only PyTorch wheel.**
PR143 remains a provisional keep: its five late misses are real, and ordinary
AMD execution still fails acquisition. No release qualification or global
leaderboard win is established.

Pinned sources: PR140 `ef4084a46d999af33c8c484ef36dfdafbe3fc517`; PR143
`3384976cc6fb1931ac63f7b213e585654c9d8996`. Tests use seed 0 throughout, one CPU
thread, PyTorch 2.14.0+cpu, and ATen AVX2 on an AMD Ryzen 9 5900X.

| Execution | Cold ring at 1200 | Stay 1210–2400 | Worst modes / HQ | Final |
| --- | --- | --- | --- | --- |
| PR140, ordinary AMD dispatch | 7 / HQ 1 | 0/120 | 0 / 0 | 5 / .3826 |
| PR143, ordinary AMD dispatch | 7 / HQ 1 | 0/120 | 0 / 0 | 5 / .3826 |
| PR140, diagnostic Intel MKL dispatch | 8 / HQ .998779 | **114/120** | 5 / .362549 | 8 / 1 |
| PR143, diagnostic Intel MKL dispatch | 8 / HQ .998779 | **115/120** | 7 / .622314 | 8 / 1 |

For **both PRs, all 2,400 recorded update rows and all 240 diagnostic rows match
their submitted JSON exactly**. This is more than matching endpoint scores.
PR143's warm summary also matches exactly, including initial, final, and identity
state hashes; identity and method each pass 200/200. Its separate cold trajectory
and ring JSON match in every field other than elapsed timing. All 18 focused
reach, delayed-arm, and width-hold tests pass.

PR143 misses at 1560, 1810, 2110, 2210, and 2380; each recovers at the next
10-update observation. Step 2190 reproduces 8 modes / HQ .933349609375. These are
sparse observations, not proof that intervening updates all pass. Its passing
suffix is two observations. Preserve the result as the best sparse stay in
this compared family, with its hardware sensitivity and remaining failures.

## First divergence and controlled check

The divergence begins at update 1, before either delayed rule can activate:

| Execution | Critic sharpness | G curvature factor |
| --- | ---: | ---: |
| Submitted PR140 / PR143 | .38521575927734375 | .10629096826749508 |
| Ordinary AMD local run | .3852115273475647 | .10636833119977061 |
| Same run with diagnostic MKL dispatch override | .38521575927734375 | .10629096826749508 |

Initial generator, critic, prior, and RNG tensor hashes are identical between
local executions. Setting deterministic algorithms or inter-op threads to one
does not change the failing prefix. MKL conditional reproducibility controls
change the arithmetic but do not match the author's trace. Overriding only
`mkl_serv_intel_cpu_true()` to return 1, while retaining the AVX2 instruction cap,
reproduces the submitted three-update prefix, then the entire 2,400-update runs.
The model, seed, harness, optimizer, rates, and training source are unchanged.

This is a diagnostic intervention in MKL's vendor dispatch. The exact Cursor
cloud CPU model and command have not been supplied; no claim about that CPU's
identity is needed for the measured result. Matching the reported PyTorch
version and ATen capability alone did not establish equivalent math execution.
Intel documents separate vendor restrictions for its [conditional
reproducibility controls](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-1/get-started-with-conditional-num-reproducibility.html).

**The preload is not a production fix or a recommended benchmark default.**
It was confined to explicit audit commands on this AVX2-capable machine. The
ordinary search environment remains unchanged. Robustness on ordinary AMD
and GPU execution is still unqualified; do not silently overwrite the failing
receipts or select a favorable math backend as evidence of stability.

## Reproduce / hand off

`prefix_probe.py` observes the unchanged runner and stops after three completed
updates. It records CPU model, environment controls, loaded module paths/hashes,
loaded libraries, torch flags, initial tensor hashes, and per-phase sharpness.
It does not alter training values or consume random draws.

From this audit directory, point `PR143` to its pinned checkout and `PYTHON` to
the 2.14.0+cpu interpreter. Use fresh output paths:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 CUDA_VISIBLE_DEVICES= \
  "$PYTHON" prefix_probe.py --repo "$PR143" --output /tmp/pr143-prefix-audit
cc -shared -fPIC -O2 intel_dispatch_probe.c -o /tmp/pr143-mkl-dispatch.so
cd "$PR143"
env -u PYTHONPATH -u ONEDNN_MAX_CPU_ISA -u DNNL_MAX_CPU_ISA \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 CUDA_VISIBLE_DEVICES= \
  LD_PRELOAD=/tmp/pr143-mkl-dispatch.so \
  "$PYTHON" -u reports/toy100/gan_followup_probe.py \
  --phase stay --method holdw15 --output /tmp/pr143-stay-audit
```

The same audited command uses `--phase warm` or `cold` for those gates. PR140
uses its own pinned checkout, `--method delayg05`, and a distinct output path.
For an ordinary-dispatch control omit `LD_PRELOAD`. Do not preload this shim
into unrelated processes. No GPU is needed to reproduce this CPU discrepancy.

Run `compare_receipts.py --pr140 PINNED_PR140 --pr143 PINNED_PR143` from any
location to check the archived evidence against those checkouts. It asserts
all update/quality rows, warm state hashes, cold results excluding timing,
initial-state parity, and declared source hashes. [comparison.json](comparison.json)
contains the checked numbers. [prefixes/](prefixes/) contains the environment
manifests, including the ordinary and dispatch-controlled runs.

Next useful work: preserve PR143 as a contender, obtain a supported common
benchmark environment with explicit backend provenance, and test continued
quality after acquisition across ordinary execution backends. Additional
seed sweeps or changing the arm/width to make this receipt pass would not
resolve the demonstrated reproducibility issue.
