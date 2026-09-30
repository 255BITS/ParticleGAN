# Common-mode translation trust on stall reach

Host: neural. Seed 0, one thread, PyTorch 2.14.0+cpu. Draft only. This is not a
22/22 claim and not a production candidate.

The AVX2 pin is `ATEN_CPU_CAPABILITY=avx2` together with
`ONEDNN_MAX_CPU_ISA=AVX2`, `DNNL_MAX_CPU_ISA=AVX2`, and
`MKL_ENABLE_INSTRUCTIONS=AVX2`. On this machine ATen-only AVX2 still ran the
AVX512 numeric path: the warm identity matched the published AVX512 receipt
(6 modes, min HQ .953125). With oneDNN and MKL clamped, the warm identity
matches the AVX2 receipt exactly (200/200, min HQ .989990234375, final HQ
.9990234375). AVX512 below is the native build check.

Receipts: [continuous-evidence/pr84-translation-trust](continuous-evidence/pr84-translation-trust/).
Runner: [gan_followup_probe.py](gan_followup_probe.py) (`--method trans`).
Candidate: [pr84_translation_trust.py](pr84_translation_trust.py).

## Mechanism

Stall reach is unchanged. The stencil widths, the stall predicate, and the
curvature bounds (.25 on G, 3 on D) are the same as PR #107.

After that curvature-bounded G step, the clean particle cloud's displacement
is split into the mean translation and the motion of each particle relative
to that mean. An output-bias correction scales only the mean. Relative
positions are unchanged. Trajectory outputs are not a 2D particle cloud, so
the bound is idle there (`translation_steps` 0).

The scale is D's value on a pure translation of the pre-step cloud. With
`v(a)` the mean critic score of that shift and `L` the proposed length:

- slope `s = (v(+L) - v(-L)) / (2L)`
- curvature `q = (v(+L) + v(-L) - 2v(0)) / L^2`
- `s <= 0`: reject the translation
- `q < 0`: keep it only up to the quadratic peak `-s/q`
- otherwise D is still rising through the probed step, and the translation is kept

The length itself never sets the factor. A step D endorses is kept at that
length; the same length is rejected when the slope refuses it.

**Purity:** GAN dynamics only. The probe reads D on generated particles.
There is no coverage, likelihood, anchor, assignment, mode quota, or clip
ladder. Mode centers are read only by offline diagnostics, after training.

## Why this shape

PR #107's per-update trace shows the continued dropout starting as common
translation (translation ≈ RMS), while G's own-curvature trust opens and D's
slope stays moderate. A magnitude cap on that component was already killed
on #107: the same translation is 56–83% of output motion in acquisition and
in healthy holds. This bet uses D's probe so an endorsed translation can
still move, which is the acquisition case, and an unendorsed one is removed.

## Gates

| Gate | PR84 pin | #107 stall reach | **Translation trust** |
| --- | --- | --- | --- |
| Warm 1001–1200 (AVX2) | 196/200, min HQ .866 | 200/200, min HQ .921 | **200/200**, min HQ .928, final 8 / 1.0 |
| Cold trajectory (AVX2) | PASS | PASS | **PASS** |
| Cold ring (AVX2) | 8, terminal HQ .988–.999 | 8, terminal HQ 1.0 ×5, suffix 8 | **FAIL**, 8 modes at 900–1150 with HQ 1.0, terminal HQ .830, suffix 0 |
| Cold ring (AVX512, build check) | 7 | 8, terminal HQ .92–1.0 | **FAIL**, terminal 7 / .698; 0 modes at 1050 |
| Continued 1210–2400 (AVX2) | 53/120, final 6 / .904 | 97/120, final 8 / .971, suffix 25 | **92/120**, min modes 1, min HQ .018, final 8 / .998, suffix 3 |

Warm does not regress against the PR84 pin. The stay does regress against
stall reach, and the cold ring does not clear.

The continued run still has the ~1720–2150 episode. From 1720 to 2150, 20 of
44 checks pass and the cloud reaches 1 mode (HQ .018 at 1800). Failures
continue through 2100, and 2370 fails again, so the terminal suffix is 3.
Stall reach's episode includes 0-mode checks and then stays clean after 2150.
Here the trust was open during the walk: at 1756–1772 the factor was 1 on
most updates while the translation ran .05–.16, because D's slope along that
translation was still positive. Over the whole stay the bound shrunk 1023 of
2400 steps and rejected 479. That was not enough to hold the ring.

## Rank

Stall reach (PR #107) stays first on the GAN-native board. This candidate
matches its warm score and passes trajectory, then loses the cold ring on
both builds and finishes the continued run with a shorter clean suffix.

## Keep / kill / next bet

**Kill.** Do not retune the peak rule and do not add a magnitude cap. The
cap is the bet #107 already killed offline, and this probe shows the missing
signal: while the cloud is leaving, D's value along the translation still
endorses the step.

**Next single bet:** bound G by cross-curvature. Measure how D's response to
G's previous step changes G's gradient, and trust the step by that product.
#107's 1755 fork already shows the dropout grows with G's step and gets worse
when D is faster, so it is a D×G mode rather than D lagging. Test that bound
on the 1755 fork before another full stay. Leave stall reach and this
translation rule unchanged.
