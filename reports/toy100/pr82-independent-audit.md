# Independent PR82 alternating-update audit

At PR82 `c1515197`, the new alternating own-curvature adapter reproduces
plain alternating Adam exactly when its two bounds are disabled. The declared
G bound `.25` / D bound `3` candidate passes fixed-target trajectory and the
scheduled-prefix warm check here, but **fails cold ring acquisition**. It is
still scratch-only and supplies no production replacement.

All runs use seed 0, Python 3.12.13, PyTorch 2.13.0+cu126 on CPU, one thread,
and AVX2. The fixed PR82 config removes network horizon and sets the common
LR scale to one. Applied G/D/prior nominal rates are `.00425/.00425/.0085`.
The [manifest](continuous-evidence/pr82-independent-audit/manifest.json)
contains source hashes and hashes for the compressed raw curves, optimizer
receipts, warm summary, parity result, and tailable logs. No seed sweep or
parameter grid was run.

| Fixed-seed arm | Trajectory | Ring | Warm continuation |
| --- | --- | --- | --- |
| Plain alternating Adam, no LR decay | PASS, MSE `.0034514049`, 15-check suffix | FAIL, 7 modes / HQ `.36938` | — |
| PR82 alternating adapter, both bounds `1e9` | Exact plain-host parity | Exact plain-host parity | — |
| ExtraAdam's one-evaluation **simultaneous** Adam | FAIL, MSE `.25443435`, 0/24 passing checks | Not run | — |
| PR82 alternating, G `.25`, D `3` | PASS, MSE `.0009426624`, 18-check suffix | **FAIL, 0/24 passing checks; 6 modes / HQ `.89209`** | 200/200, minimum HQ `.94043` |

The warm candidate is conditional on an unchanged scheduled prefix through
update 1000. Its identity child matches the uninterrupted control's final
state hash; the constant warm control passes 6/200 checks. It is not a cold
acquisition result. The cold ring's terminal live values at updates
1000/1050/1100/1150/1200 are respectively 4/.474, 4/.592, 6/.820,
6/.921, and 6/.892 (modes/HQ). Its final HQ is close to the threshold, but
the required eight modes and sustained suffix are missing. The candidate
performs three field evaluations and one Adam moment update per player per
outer update; same-noise replay was verified 800 times on trajectory and
2400 on ring.

The full-host parity driver compares untimed complete result hashes, every
optimizer parameter and moment tensor, global RNG state, isolated input and
output RNG stream states, applied role rates, and final live metrics. All six
comparisons are exact on **both** 400-step trajectory and 1200-step mode-hold.
The adapter's noise *counting* receipt differs by design because it executes
two additional same-sample field evaluations per update; final training RNG
states still match. Five focused PR82 tests also pass.

The update-order finding is real and scoped. In the frozen host, D's Adam step
occurs before the G gradient is computed. In `extra_adam_scratch.py`, D's
`step` call captures its gradient and returns without moving; G's gradient is
then captured at the old D. That simultaneous joint-field method is
intentional, not a faulty replay of alternating Adam. Its one-evaluation
control fails trajectory while direct alternating Adam passes here. Thus
the earlier ExtraAdam-derived EG/implicit/cross-only trajectory failures mix
their particular stabilization rule with an update-order change. This finding
does not apply to direct Adam-step controllers such as the functional metric.

PR82's two-bound phase sequence has the intended order: ordinary D proposal,
same-batch D own-field replay and interpolation to D*, G gradient/ordinary
proposal against D*, then G own-field replay and interpolation. The Adam
metric uses the post-proposal bias-corrected second moment, and the same
training RNG stream is restored before each replay. The tests and full-host
parity found no numerical or order defect for these two supported hosts.
The reported `rho` is a **directional endpoint secant** along the full
proposal in that frozen metric. Scaling by `min(1, c/rho)` does not certify
the secant at the scaled point, global smoothness, monotonic player loss, or
nonlinear convergence; statements that `rho<1` guarantees monotonicity or
`rho>2` proves divergence would overstate this measurement. The `.25/3` arm
always evaluates both bounds; it does not use the module's optional critic
advantage gate.

PR81's optional `error_relative_gain` cap is a separate diagnostic. It reads
the ring's exact mode centers or trajectory's exact fast target inside the
update to compute remaining error. That target oracle is unsuitable as a
general production optimizer signal, irrespective of its warm result.

The archived PR82 `.25/3` config and declaration equal ours, and the adapter
bytes have the same SHA-256
`cb6278251cf04d321afe7109f5eb91b3a5ebb41a866abb03c9a694b8c9969450`.
Our first curvature ratios differ from its archive by roughly `1e-5` to
`1e-3`; by the
first ring observation at update 50, the particle clouds differ markedly.
The archive reports ring 8 modes/HQ 1 at update 1200 but only 3/24 passing
checks, so **both environments reject it**. Our trajectory MSE differs from
the archive by under `6e-9`. These are same-seed numerical sensitivity facts,
not statistical replications or evidence of a winner.

Reproduce from this worktree using `/tmp/pr38-default-env/bin/python` with
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`,
`CUDA_VISIBLE_DEVICES=''`, `ATEN_CPU_CAPABILITY=avx2`,
`MKL_ENABLE_INSTRUCTIONS=AVX2`, and `ONEDNN_MAX_CPU_ISA=AVX2`:

```bash
python -m pytest -q tests/test_alternating_curvature_scratch.py
python -u reports/toy100/pr82_alternating_parity_audit.py --output NEW_PARITY.json
python -u reports/toy100/alternating_curvature_warm_probe.py --curvature-bound .25 --bound-d --d-curvature-bound 3 --output NEW_WARM
python -u reports/toy100/alternating_curvature_probe.py --curvature-bound .25 --bound-d --d-curvature-bound 3 --output NEW_COLD_DIR
```
