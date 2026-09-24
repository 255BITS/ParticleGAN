# PR84 rest-slope implementation repair

**Corrected measurement, no fixed-target solution.** One predeclared arm
repairs only the double convolution in PR84 head
`468ad2619a0536f8c69ef878b86bc514d296338a`. It passes warm stability and cold
trajectory, then fails the original cold ring with six modes. No threshold
scan or longer hold follows that failure. The prior PR84 results retain
their original labels; this is a distinct implementation-repair candidate.

The [summary](continuous-evidence/pr84-rest-repair/summary.json),
[manifest](continuous-evidence/pr84-rest-repair/manifest.json),
[untouched head source](continuous-evidence/pr84-rest-repair/source/pr84_head_original.py)
and [three-line patch](continuous-evidence/pr84-rest-repair/source/implementation-repair.patch)
preserve the precise source and result provenance. The canonical PR82 module
is unchanged. The isolated adapter is
[`pr84_rest_repair_scratch.py`](pr84_rest_repair_scratch.py), SHA256
`098c76a8c79cb062e4be7382e0105e06fd6a7c65e96bd90bb2a8702536bc5e4c`.

## Exact defect and repair

PR84's G loss uses the five-point spatial stencil

```text
K D(x) = [D(x) + D(x+w*e1) + D(x-w*e1)
               + D(x+w*e2) + D(x-w*e2)] / 5.
```

The later rest gate manually computes the same stencil to estimate its
input slope. Its `module(points)` calls go through a critic forward already
patched to return `K D`. The measured slope is therefore that of `K² D`.
This differs from the documented slope of the critic that G actually uses.

The repair retains the captured, unpatched `original_forward` and calls it
inside the manual rest stencil. Exactly two call-site lines and one captured
function assignment change. The .2 rest threshold, width
`min(.15, .5/sharpness)`, finite-difference epsilon .001, G curvature bound
.25, D bound 3, losses, alternating order and Adam moments remain unchanged.
There is no new field correction, penalty, oracle, schedule or gain.

The existing optional mode latch, timed boost, latent nudge and angular-hole
rules are disabled. Occupied-mode computation inherited from PR84 is
diagnostic only and does not feed this arm's updates. The training policy
uses current critic scores/slopes, and the same rule applies warm and cold.

For an independent analytic check, choose `D(x)=5*x1³`, width .15 and evaluate
at zero. The exact central-difference slopes are

```text
single convolution: 5 * [(6/5)*.15² + .001²]  = .135005
double convolution: 5 * [(12/5)*.15² + .001²] = .270005
```

The repaired code therefore rests and the original code moves, at the
unchanged threshold .2. This test exercises each actual adapter's patched
critic and rest decision rather than a separate approximation helper.

## Gate results

All runs use the existing seed 0, one-thread CPU/AVX2 environment and frozen
recipe. Nominal G/D rates remain .00425 and prior .0085. Noise clocks retain
the original mode-hold 1,200 and trajectory 400 horizons.

| Gate | Result | Rest decisions |
| --- | --- | --- |
| Warm 1001–1200 | PASS 200/200; min 8 modes/HQ .916016; final 8/.995850 | 162 of 200 G/prior proposals rejected |
| Cold trajectory, 400 | PASS MSE .000942662; 18-check passing suffix, first 117 | Slope repair does not apply to this host's non-2D critic input |
| Cold ring, 1,200 | FAIL 0/24 passing; final 6 modes/HQ .960938 | 204 of 1,200 proposals rejected; diagnostic never reaches eight clean occupied modes |
| Fixed-target hold >=2,400 | Not run after failed ring | — |

The five original ring tail checkpoints retain six modes throughout, with
HQ .811035, .951416, .963135, .906250 and .960938. Missing modes at the
endpoint are 0 and 7. Endpoint HQ alone is insufficient; the coverage gate
is unchanged.

The warm scheduled identity reproduces its cold control exactly, with
shared warm-state hash
`6cc79b6e0d11eafae176b68e7d9d8c26c02c886866370134cd70c864fe882e21`.
The corrected warm slope ranges from .083332 to .423403, mean .173033.
Repairing the slope changes the trajectory of the gate, as expected; it
does not demonstrate that the .2 threshold solves acquisition and retention.

## Validation and accounting

Six tests pass: the exact single/double cubic derivative and gate distinction;
the independent common-stencil cross-block example below; disabled-smoothing
parity with untouched PR84; exact parity of smoothed training when both rest
decisions are disabled; delayed-activation ordinary-host parity; and active
host RNG/moment accounting. Full parameter and Adam-state tensors, noise
receipts and global/input/output RNG streams are compared in the parity tests.

The controller records every active optimizer callback's actual role rate.
Warm uses 600 gradient evaluations per player and 400 same-RNG replay checks;
trajectory uses 1,200 and 800; ring uses 3,600 and 2,400. Warm full-host Adam
counters are exactly 1,200. The actual-host unit test checks one moment advance
per update despite three gradient evaluations. The inherited cold recorder
reports that control-path invariant, while its cold rate records enumerate
all three phases. It does not store an additional final cold Adam tensor
snapshot, so those should not be mistaken for a separately hashed cold state.
All artifacts remain `shared_gate_eligible=False` and scratch-only.

## Independent note on using one score for both players

This is a separate mathematical motivation for the common-stencil arm being
tested elsewhere, not part of the rest-slope repair. For scalar score
`D_theta(x)=theta*x³` and a zero-mean stencil of marginal variance `v`,
`K D_theta(x)=theta*(x³+3*v*x)`. With a real point at zero, sharp-D and
smoothed-G descent fields have cross derivatives `-3*x²` and
`3*(x²+v)`. At `theta=x=0`, their Jacobian is

```text
J_unilateral = [[0, 0], [3*v, 0]]
J_common     = [[0, -3*v], [3*v, 0]]
```

The first has one-way nilpotent coupling; sharing the convolved score
restores opposite cross blocks. A test verifies both matrices by autograd,
with `v=2*w²/5` for the five-point 2D stencil. This illustrates a structural
inconsistency in unilateral smoothing, not a convergence guarantee for
the capped, relativistic neural game or proof that common smoothing wins.

Reproduce the repaired policy with fresh output paths:

```bash
python -m pytest -q tests/test_pr84_rest_repair.py
python reports/toy100/pr84_rest_repair_probe.py --phase warm --output /tmp/rest-warm
python reports/toy100/pr84_rest_repair_probe.py --phase cold \
  --previous /tmp/rest-warm/run/summary.json --output /tmp/rest-cold
```

The current ring verdict prevents the controller's `--phase hold` path.
