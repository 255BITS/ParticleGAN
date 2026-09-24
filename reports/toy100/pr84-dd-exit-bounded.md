# DD-exit shrink on the curvature-bounded stall-reach step

Host: neural. Seed 0, one thread, PyTorch 2.14.0+cpu. Primary build:
`ATEN_CPU_CAPABILITY=avx2` with oneDNN capped at AVX2
(`ONEDNN_MAX_CPU_ISA=AVX2`). `ATEN_CPU_CAPABILITY=avx2` alone reproduces the
AVX512 warm identity (6 modes, min HQ .953125). Receipts:
[continuous-evidence/dd-exit-bounded](continuous-evidence/dd-exit-bounded/).

**Call: KEEP on the AVX2 pin. Stall reach stays the dual-build reference.**

## Mechanism

#107 stall reach is unchanged: stencil widths, stall predicate, G curvature
bound .25, D curvature bound 3.

1. Take the normal curvature-bounded G Adam step. That accepted step is Δx.
2. On particles with sharp D above the batch median, measure grad_D · Δx
   (the #113 diagnostic).
3. If that mean is negative, shorten the accepted step by
   `max(0.3, 1 + mean_dd / max(|mean_dd|, 0.01·‖Δx‖))`. The scale is at most 1,
   so the curvature bound is not loosened and is not replaced.

#113 applied this shrink only when the bound had accepted the full Adam step.
On this wiring that legacy gate fires **0** times during the AVX2 cold ring.
The post-bound shrink fires **192 / 1200** times there (mean scale 0.32). The
two signals are not mutually exclusive.

**Purity.** Adversarial dynamics only. The signal is the sharp critic and its
input gradient along the accepted particle step. No coverage objective,
anchor, forward-KL training term, mode quota, Chamfer assignment, or clip ladder.
The trajectory host's critic is not the 2D ring MLP, so the shrink stays idle
there.

## Gates

| Gate | PR84 pin | #107 stall reach | **DD-exit on bounded step** |
| --- | --- | --- | --- |
| Warm 1001–1200 (AVX2) | 196/200, min HQ .866 | 200/200, min HQ .921 | **200/200**, min HQ .921, final 8 / .980 |
| Cold trajectory (AVX2) | PASS | PASS | **PASS** (3.6 s, shrink idle) |
| Cold ring (AVX2) | 8 | 8, suffix 8 | **8**, live HQ .983, on 8 modes from step 750, 192 clipped shrinks |
| Cold ring (AVX512) | 7 | 8 | **6**, live HQ .443 (8 modes only at 1050, HQ .840) |
| Stay 1210–2400 (AVX2) | 53/120, final 6 / .904 | 97/120, final 8 / .971, suffix 25 | **97/120**, final 8 / 1.0, suffix 18 |

Identity on the AVX2 warm fork is 200/200, min HQ .98999, the published pin.
Bounds in every receipt: G .25, D 3, ramp stall.

Stay still has the shared dropout. Worst check is step 1810 at 1 mode, HQ .069.
Failures also land at 2200–2220 (7, then 3, then 8 at HQ .799). The run is back
on the ring by 2230 and ends at 8 modes, HQ 1.0. That is the same pass count
as stall reach, with a shorter clean suffix.

## Keep / kill

- **Keep** on the ordered AVX2 gates. Warm does not regress versus PR84.
  The cold ring is the full 8. The continued run ends on the ring. The
  projection is the post-bound shrink #113 left unused, and it actually runs.
- **Do not promote** over #107. AVX2 stay is tied at 97/120. The AVX512 cold
  ring, which stall reach holds at 8, ends at 6.
- The floor 0.3 is the #113 constant, reused once. It was not swept.
