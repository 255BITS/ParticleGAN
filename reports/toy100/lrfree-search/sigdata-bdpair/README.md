# sigdata-v1 + paired birth/death graft: native100 2/3 (honest generator fix)

First core (non-sampler) candidate to pass grid100 **and** staggered100 with the
real generator law. The preceding paired-BD graft on `sigfloor-v3` failed grid
(centre .26σ). This graft puts the same `bdguard-dRpair` transport on the
`sigdata-v1` base (settle-gated learnable sigma, init .029) and passes 2/3.

## Candidate (one declared mechanism)

- Base: `/ml2/hypergan/lrfree-20260926/candidates/sigdata-v1/package`
  (all files byte-identical except below).
- Graft: `particlegan/birth_death.py` replaced by the `bdguard-dRpair` file
  (d_R guard: use the real-law dimension directly clamped to [1, D_out];
  paired-only transport, zero neutral clones). `training.py`, `continuous.py`,
  controller, sigma rule, init, seeds, `total_steps=null` all unchanged.
- Overrides: sigdata-v1 overrides + `particle_birth_death: true` only
  (`overrides.json`). Runner package SHA-256 `df0515e1...` (see job headers).
- Grid run reproduced the sigdata-v1 baseline first (prec .9686 / centre .265 /
  min-eig .627 / max-eig 2.39 / TV .061), then the graft beat it on every column.

## Frozen results (7,000 updates, QR `batch_feature_zero`, noisy scoring)

| Task | Verdict | Checks | Arrival | Streak | Prec | Centre/σ | Eig range | TV | Trace | KS | σ | Holdout prec |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| grid100 | PASS | 20/34 | 1500 | 12 | .9850 | .196 | .670–1.484 | .032 | .021 | .018 | .029 | .9848 |
| rotated100 | FAIL | 0/34 | — | 0 | .9516 | .266 | .717–1.627 | .034 | .124 | .063 | .029 | .9509 |
| staggered100 | PASS | 18/34 | 2250 | 6 | .9761 | .176 | .693–1.512 | .032 | .014 | .014 | .029 | .9745 |

Raw `result.json`, fixture, header, 34 scored rows, 7,000 rate rows and native
diagnostics per task are committed here (`{grid100,rotated100,staggered100}-*`).
Local run dirs (with sample clouds + final states, ~19M each):
`/ml2/hypergan/gan-attempts/formulations-20260928T040504Z/custom_follow/20260928T040504Z-4138988/runs/cf1-bdpair-{grid100,rotated100,staggered100}/`.
Executed package:
`.../custom_follow/20260928T040504Z-4138988/candidates/cf1-bdpair/`.

## Read vs the row-EM sampler patch

`row-em-renew` reports 3/3 noisy but 0/3 clean (per-mode clouds too tight;
eval-time sigma inflation covers it). This candidate is 2/3 with the **trained**
law itself — no separate sampling sigma, no reweighting. Rotated100 is the
honest remaining failure: mass transport works (TV .034) but per-mode shape is
wrong (trace +.124, KS .063, centres .266σ). That is the next structural target,
not another sampler fit.
