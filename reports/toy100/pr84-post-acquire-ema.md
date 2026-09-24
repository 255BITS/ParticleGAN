# DRAFT: post-acquire EMA discriminator for G — killed

Not solved. Do not merge. Production still uses scheduled decay. This is one
mechanism on stall reach (#107), not a 22/22 claim.

Host: torch 2.14.0+cpu, seed 0, one thread. Valid pin:

```
ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2
ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2
```

`torch.backends.cpu.get_cpu_capability()` is `AVX2` on every receipt.
ATen-only `avx2` (oneDNN left free) is not the pin: its identity control was
0/200 at 6 modes, and that receipt is refused
([log](continuous-evidence/pr84-post-acquire-ema/avx2/warm-aten-only-invalid.log)).

## Mechanism

Stall-reach widths, curvature bounds, and D's live Adam step are unchanged.
After the run first reads 8 modes at HQ ≥ 0.9 (the mode-hold gate's own
4,096-sample check, forked RNG), G's adversarial forward uses an EMA copy of
D with τ = 0.99. D keeps stepping on the live weights. Before that arm, a
2-step host matches stall reach, including the global RNG.

No coverage term, anchor, mode quota, clip, or τ / threshold sweep.

## Gate

| Gate | PR84 pin | #107 stall reach | **EMA-D for G** |
| --- | --- | --- | --- |
| Warm AVX2, 1001–1200 | 196/200, min HQ .866 | 200/200, min HQ .921 | **154/200, min modes 6, min HQ .371, final 6 / .423, suffix 0** |
| Cold trajectory | PASS | PASS | not run |
| Cold ring AVX2 / AVX512 | 8 / 7 | 8 / 8 | not run |
| Stay 1210–2400 | 53/120, final 6 | 97/120, final 8, hard dips to 0 | not run |

Control on the valid pin: identity **200/200**, min modes 8, min HQ .990,
final 8 / .999. Warm is rankable.

EMA armed at the fork (`acquired_step` 1000, 8 modes, HQ .997), so G read the
EMA critic for every continuation update (200 EMA mixes). It held until 1152,
passed 1156–1157, then stayed off the ring through 1200.

## Kill

**Kill.** Decisive counterexample: on a solved AVX2 warm ring whose control
is 200/200, this arm walks the cloud to 6 modes and HQ .423 and does not
come back (passing suffix 0). That is a collapse against both the #107 pin
(200/200, min HQ .921) and the PR84 pin (196/200, min HQ .866).

Do not retune τ or the arm threshold. Cold and stay were not run.
