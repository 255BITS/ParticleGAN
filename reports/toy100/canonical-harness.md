# Canonical toy100 continuous-learning harness

Draft evidence only. This does not qualify a release, and it does not claim a
full toy-suite pass. No training rule or rate was changed.

The scoring entry point is `scripts/toy100_env.sh` plus
`benchmarks/toy100/canonical_env.py`. `reports/toy100/gan_followup_probe.py`
calls `require_canonical_env()` before it creates an output directory. If the
shell entry point is not active, or MKL did not keep the selected CBWR mode,
the process exits 2 and prints `REFUSING TO SCORE`.

## Which CBWR mode, and why

`MKL_CBWR=AVX2,STRICT` was probed first on this VM (Intel Xeon, KVM,
PyTorch 2.14.0+cpu, oneMKL 2024.2). `mkl_cbwr_get` is not exported by that
wheel. `mkl_serv_cbwr_get(-1)` returns the same `mkl_cbwr` word: branch in the
low 16 bits, STRICT in bit 0x10000.

| Dispatch | Requested | Effective raw | Name |
| --- | --- | ---: | --- |
| Native Intel | `AVX2,STRICT` | 65546 | AVX2,STRICT |
| `mkl_serv_intel_cpu_true` forced to 0 | `AVX2,STRICT` | 65538 | branch 2, STRICT |
| Either path | `COMPATIBLE` | 3 | COMPATIBLE |

AVX2,STRICT sticks on Intel dispatch and does not stick on the non-Intel
dispatch path. Scoring it would make the two vendors run different math, and
the guard refuses the non-Intel process because the effective mode is not the
requested one. The shell therefore falls back to **COMPATIBLE**. Both paths
keep raw 3.

The other pins are unchanged: `ATEN_CPU_CAPABILITY=avx2`,
`MKL_ENABLE_INSTRUCTIONS=AVX2`, `ONEDNN_MAX_CPU_ISA=AVX2`,
`DNNL_MAX_CPU_ISA=AVX2`, one OMP/MKL/OpenBLAS thread, `torch.set_num_threads(1)`,
`torch.set_num_interop_threads(1)`, `torch.use_deterministic_algorithms(True)`,
seed 0. Every declaration and JSONL event carries torch version and build, CPU
vendor and model, MKL version, effective CBWR, the ISA pins, and thread counts.

## Does it actually standardize?

300 unchanged `holdw15` updates, seed 0. State hash covers generator, critic,
prior, and RNG tensors.

| Setting | Intel vs AMD-like | max abs diff | Guard |
| --- | --- | ---: | --- |
| Canonical `COMPATIBLE` | bit-identical `19fdc2ab…` | 0 | accepts both |
| `AVX2,STRICT` | different hashes | 0.580 | accepts Intel, refuses AMD-like |

Under the canonical env the vendor shim changes nothing: the 300-update states
match. `AVX2,STRICT` does not. On that setting the AMD-like hash equals the
COMPATIBLE hash, and the Intel hash does not. Update-1 critic sharpness is
0.3852161765098572 on the canonical path. The saved Intel-dispatch prefix was
0.38521575927734375 and the ordinary AMD prefix was 0.3852115273475647. The
canonical run matches neither of those prefixes. Details:
[path-compare.json](canonical-harness/path-compare.json).

## Re-baseline next to the old receipts

Old columns are the submitted AVX2 receipts (Intel MKL dispatch, no CBWR, no
deterministic-algorithms flag): #107 `reachstall` at `d5ec127b`, #140
`delayg05` at `ef4084a4`, #143 `holdw15` at `3384976c`. New columns are the
same pins under the canonical env on this Xeon. Cold trajectory still passes.
The cold ring does not. That is an acquisition failure for all three.

| Gate | #107 old | #107 canonical | #140 old | #140 canonical | #143 old | #143 canonical |
| --- | --- | --- | --- | --- | --- | --- |
| Identity warm | 200/200, min HQ .990 | **200/200**, min HQ .998 | 200/200, min HQ .990 | **200/200**, min HQ .998 | 200/200, min HQ .990 | **200/200**, min HQ .998 |
| Method warm 1001–1200 | 200/200, min HQ .921 | **189/200 FAIL**, min HQ .840 | 200/200, min HQ .921 | **189/200 FAIL**, min HQ .840 | 200/200, min HQ .921 | **189/200 FAIL**, min HQ .840 |
| Cold trajectory | PASS, MSE .000942668 | **PASS**, MSE .000942669 | PASS, same MSE | **PASS**, same new MSE | PASS, same MSE | **PASS**, same new MSE |
| Cold ring | PASS, 8 / HQ .999, 10/24, suffix 8 | **FAIL**, 8 / HQ .925, **4/24, suffix 2** | PASS, same | **FAIL**, same | PASS, same | **FAIL**, same |
| Stay 1210–2400 | 97/120, min 0 / HQ 0, final 8 / .971 | **110/120**, min **4 / .353**, final 8 / .999 | 114/120, min 5 / .363, final 8 / 1 | **115/120**, min **7 / .749**, final **8 / .862** | 115/120, min 7 / .622, final 8 / 1 | **116/120**, min **8 / .831**, final **8 / .990** |

The three method warm failures are the same eleven steps (1054–1055 and
1104–1112). Those steps are before any delayed arm. The cold-ring miss is the
same 4/24 for every method: the rule wants a passing suffix of 5 and observed
2. Step 1200 is still 8 modes, but HQ is .925 instead of .999, so the old
acquisition margin is gone.

Stay counts are not the old receipts. #140’s endpoint check fails (8 / .862 at
2400). #107 and #143 still end on 8 modes with HQ above .9, on different fail
steps than the submitted JSON. Logs are
`reports/toy100/canonical-harness/runs/<pin>-<phase>.log` (one JSON line per
event; `tail -f` works).

## Leaderboard under this env

Cold ring fails for #107, #140, and #143. None of them acquire under the ring
rule here. Among the sparse stay checks only, the order is #143 (116/120, min
8 modes), #140 (115/120, min 7, endpoint fails), #107 (110/120, min 4). That
order is not a reason to promote any of them.

## Recommendation

Keep COMPATIBLE as the scoring mode. It is the setting both dispatch paths
actually share. Do not score `AVX2,STRICT` on Intel and call it the same
experiment as an AMD run. Do not treat the old Intel-dispatch stay tables as
portable. Do not retune the arm, the width, or the rates to recover the ring;
the miss is already present in the shared pre-arm updates.
