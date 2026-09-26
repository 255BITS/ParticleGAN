# Deterministic init search C (CPU screen)

CPU screening only. These receipts need A6000 confirmation. CPU training does
not follow the GPU path, including MKL dispatch, so a pass or fail here is a
screen. The default PyTorch init is unchanged. LR, loss coefficients, clips,
and the K3P recipe knobs are the develop values (`configs/toy100/k3p_screen.json`).

Host: CPU, torch 2.14.0+cu130, `PYTHONHASHSEED=0`, one thread
(`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`). Repo seed is offset 0.
Offsets 101…707 are applied by `benchmarks/toy100/seed_offset_run.py` and move
samples and noise only.

## Determinism

`pytest tests/test_deterministic_init.py`: 6 passed. For each of `ortho_lsuv`,
`sobol`, `halton`, and `fixedgen`, two processes at seeds 0 and 101 build the
same parameter sha256 (MLP generator, critic, and learned particle prior).
The following `torch.rand` matches a no-install run at the same seed and
differs across seeds. `sobol`, `halton`, and `fixedgen` weights are not
orthogonal. `ortho_lsuv` hidden weights are one positive scale times an
orthogonal matrix.

## Repo-seed screen (offset 0)

Ring gate is modes ≥ 8, HQ ≥ 0.9, and a passing suffix of 5. Unequal mass is
the frozen vector verdict.

| init | orthogonal | ring modes | ring HQ | ring suffix | ring | unequal suffix | unequal min mass | unequal |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| K3P default | no | 6 | 0.995 | 0 | FAIL | 12 | 0.813 | PASS |
| ortho_lsuv | weights only | 5 | 0.900 | 0 | FAIL | 15 | 0.659 | PASS |
| sobol | no | 5 | 0.913 | 0 | FAIL | 7 | 0.926 | PASS |
| halton | no | 6 | 0.992 | 0 | FAIL | 7 | 0.696 | PASS |
| fixedgen | no | 7 | 0.870 | 0 | FAIL | 19 | 0.842 | PASS |

`fixedgen` uses one CPU generator seeded with 123456789. It is not orthogonal.

## Other baseline gates (offset 0)

| gate | K3P default on CPU |
| --- | --- |
| ring (`mode_hold`, 1200) | FAIL, 6/8, HQ 0.995, suffix 0 |
| ring hold (2400 steps, check every 10) | FAIL, continued 0/120, stationary 0/5, final 5 modes |
| stay (shift at 2400, 3600 steps) | FAIL, continued hold 0/120, recovery 0/120 |
| unequal mass | PASS, suffix 12, min mass ratio 0.813 |
| grid100 | coverage FAIL (final 100/100, HQ 0.989, 0/5 terminal checks), accuracy FAIL |
| rotated100 | coverage PASS and accuracy PASS (stable from step 5750, final HQ 0.984) |
| img_stripes2 | PASS, suffix 23, 2 modes, HQ 1 |
| img_blobs4 | PASS, suffix 19, 4 modes, HQ 1 |

Halton and fixedgen also pass both image gates at offset 0 (stripes suffix 19
and 6, blobs suffix 11 and 17). On rotated100, fixedgen matches the baseline
(coverage PASS and accuracy PASS, stable from step 5750, final HQ 0.985).
Halton reaches 100/100 modes but fails the terminal window (4/5 checks) and
the accuracy gate.

## Seed offsets, ring and unequal

Ring modes / unequal verdict. A ring PASS needs 8 modes, HQ ≥ 0.9, and suffix ≥ 5.

| offset | fixedgen ring | fixedgen unequal | halton ring | halton unequal |
| --- | --- | --- | --- | --- |
| 0 | 7, HQ 0.87, FAIL | PASS (19) | 6, HQ 0.99, FAIL | PASS (7) |
| 101 | 6, HQ 1.00, FAIL | PASS (7) | 5, HQ 1.00, FAIL | PASS (9) |
| 202 | 7, HQ 1.00, FAIL | PASS (18) | 7, HQ 1.00, FAIL | PASS (15) |
| 303 | 3, HQ 0.67, FAIL | FAIL | 6, HQ 0.99, FAIL | PASS (9) |
| 404 | 5, HQ 0.81, FAIL | PASS (8) | 6, HQ 0.75, FAIL | PASS (6) |
| 505 | 6, HQ 0.92, FAIL | FAIL | 4, HQ 0.83, FAIL | PASS (7) |
| 606 | 7, HQ 1.00, FAIL | PASS (5) | **8, HQ 1.00, suffix 6, PASS** | PASS (8) |
| 707 | 5, HQ 1.00, FAIL | FAIL | 6, HQ 1.00, FAIL | FAIL |

Halton ring passes on 1/8 offsets. Fixedgen ring passes on 0/8. Unequal passes
on 7/8 (halton) and 5/8 (fixedgen). Neither holds the ring across seeds.

## Recommendation

Confirm **fixedgen** first on the A6000, then **halton**. Both are flagged
with `--init` and are not orthogonal.

Fixedgen is the closer CPU match to baseline K3P: unequal suffix 19 (baseline
12), both image gates, and rotated100 coverage plus accuracy. Its repo-seed
ring has 7 modes against the baseline's 6, with HQ 0.87 against 0.995, so it
misses the HQ bar. It never reached 8 modes on the 8 offsets.

Halton matches the baseline ring counts at offset 0 (6 modes, HQ 0.99) and is
the only init that passed the ring at all (offset 606: 8 modes, HQ 1, suffix
6, unequal PASS). It loses rotated100 on CPU (4/5 terminal checks), which the
baseline and fixedgen both pass.

Do not spend the first A6000 pass on `ortho_lsuv` or `sobol`: both drop the
repo-seed ring to 5 modes. Orthogonal LSUV did not reproduce the CPU baseline
ring. The CPU baseline itself misses the ring, the hold, the stay, and the
grid100 terminal window, and passes unequal mass, both images, and rotated100.
