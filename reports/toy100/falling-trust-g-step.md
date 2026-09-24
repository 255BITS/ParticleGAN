# Falling own-curvature trust: halve the G step

Host: neural. Seed 0, one thread, PyTorch 2.14.0+cpu. **Kill.** Not 22/22. Not production-ready. Stall reach (PR #107) stays the GAN-native reference.

This is one mechanism, not a retune of a killed family. It is not the always-on G bound .125, not a 1-step or 2-step game bound, not translation trust, not G-trust/D-slope utilisation (#116), not an idle or mode-exit predicate, not rest-damping, and not a change to the #119 DD-exit floor.

## Mechanism

Stall reach is unchanged when `trust_fall` is off. With it on, G's ordinary Adam displacement is multiplied by **0.5** only when G's own-curvature ratio has a **negative delta** over the last **8** completed G updates (`ρ_now − ρ_8_ago < 0`). Moments still advance on the gradient. The own-curvature bound stays **.25** and is applied to the scaled proposal. D's step, the stall-reach width, and the losses are unchanged.

The window and the scale are fixed. Eight is the published #107 onset span (ρ ≈ 5.05 at update 1756 to ρ ≈ 1.87 eight updates later). The scale is the G×.5 fork that scored 28/29 from update 1755. No coefficient sweep and no clip ladder.

Purity: adversarial GAN dynamics only. No coverage, anchor, forward-KL training signal, mode quota, or Chamfer assignment.

Candidate: [pr84_reach_candidate.py](pr84_reach_candidate.py) (`trust_fall=True`, ramp `stall`). Runner: [gan_followup_probe.py](gan_followup_probe.py) method `reachstall_trustfall`. Receipts: [continuous-evidence/falling-trust-g-step](continuous-evidence/falling-trust-g-step/).

## Pin

Valid warm path:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2
```

Every receipt logs `cpu` and `cpu_env`. `ATEN_CPU_CAPABILITY=avx2` alone is not enough on this host: oneDNN still ran the AVX512 numerics, the identity fork was 0/200 with min modes 6 and min HQ .953125 (the archived AVX512 identity), and that run is **not ranked**. With the full pin, the identity fork is 200/200, min HQ .989990234375, matching the archived AVX2 control.

## Gates

| Gate | PR84 pin | Stall reach (#107) | **Falling-trust G ×0.5** |
| --- | --- | --- | --- |
| Warm 1001–1200 (AVX2, control 200/200) | 196/200, min HQ .866 | 200/200, min HQ .921 | **200/200**, min HQ .919, final 8 / 1.0 |
| Cold trajectory (AVX2) | PASS | PASS | **PASS** (3.7 s) |
| Cold ring (AVX2) | 8 | 8, terminal HQ 1.0 | **FAIL.** Max 7. Terminal 7 / .660. 0/24 checks pass |
| Cold ring (AVX512) | 7 | 8 | **FAIL.** Max 7. Terminal 4 / .286. Modes 0 at update 1100. 0/24 checks pass |
| Continued stay 1210–2400 | 53/120, final 6 | 97/120, final 8 | **Not run.** The ring never acquired 8 |

Warm does not regress against the PR84 pin or against stall reach's 200/200. The kill is cold-ring acquisition.

## Why it dies

The halve fired on 588/1200 AVX2 updates and 604/1200 AVX512 updates, about half in every phase (steps 1–399, 400–799, and 800–1200 are all ~.49–.51). Own-curvature ρ mean-reverts during acquisition, so the sign of an 8-update endpoint delta is a coin flip, not the dropout-onset fall from ~5 to ~.9. Slowing G on that coin flip is the same trade the #107 line already killed: a slower generator misses the 8th mode. AVX2 sticks at 7 from update 900. AVX512 reaches 7 and then collapses.

## Keep / kill

- **Kill** falling-trust G ×0.5. Do not sweep the window and do not add a magnitude threshold on the same delta.
- **Keep** stall reach as the GAN-native reference. This bet does not displace it.
