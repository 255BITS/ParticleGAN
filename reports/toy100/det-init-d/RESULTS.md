# Family D CPU screen

Torch `2.14.0+cpu`, build `08187d9e0fba026dc8217405802ab5381dc88d90`. One thread (`OMP_NUM_THREADS=1`). This is a CPU screen. Finalists still need an A6000 run; earlier CPU builds already disagreed with each other and with the A6000.

Offsets are added to the repo seed `1234`: `0, 101, 202, 303, 404, 505, 606, 707`. Offset 0 is the repo seed. Ring is `mode_hold`. Unequal is `vector_unequal_mass`. PASS/FAIL is the frozen gate verdict.

## Leaderboard

Rank is ring passes plus unequal passes, then ring, then unequal. Priority gates are the repo seed only, and only for the top two.

| rank | `--init` | ring | unequal | priority (repo seed) | determinism |
| --- | --- | ---: | ---: | --- | --- |
| 1 | `mix_zb_pq` | 5/8 | 6/8 | 4/8 | pass |
| 2 | `mix_pb_weyl` | 6/8 | 4/8 | 4/8 | pass |
| 3 | `mix_wq_pq` | 5/8 | 5/8 | | pass |
| 4 | `hid_q` | 5/8 | 4/8 | | pass |
| 5 | `hq_pb_pq` | 7/8 | 1/8 | | pass |
| 6 | `qr_wq_pq` | 4/8 | 4/8 | | pass |
| 7 | `hq_zb_pq` | 6/8 | 1/8 | | pass |
| | **K3P baseline** | **5/8** | **2/8** | | random init |
| 8 | `hh_pb_pq` | 4/8 | 3/8 | | pass |
| 9 | `qr_pb_pq` | 4/8 | 3/8 | | pass |
| 10 | `mix_pb_pq` | 2/8 | 5/8 | | pass |
| 11 | `hq_pq` | 4/8 | 1/8 | | pass |
| 12 | `hq_pb` | 3/8 | 2/8 | | pass |
| 13 | `hh_wq_pq` | 3/8 | 0/8 | | pass |
| 14 | `hh_zb_pq` | 2/8 | 1/8 | | pass |

`hid_q` on this build is ring 5/8 and unequal 4/8, the same counts as the earlier CPU screen of that parent.

## What the hybrids did

The weight split that moves the joint score is Householder on square hidden layers and QR on rectangular input/output layers (`mix_*`).

- `mix_zb_pq` (that split, zero bias, R2 prior) is the only init at 11/16. Unequal goes from the baseline 2/8 to 6/8, and the ring rate stays at the baseline 5/8. Both gates pass at the repo seed.
- `mix_pb_weyl` (same weights, pattern bias, Weyl prior) is the best ring among the joint leaders, 6/8, with unequal 4/8.
- `mix_wq_pq` (same weights, quarter-Weyl bias, R2 prior) is 5/8 and 5/8.
- Putting the QR rectangular maps back to hid_q's tiled identity (`hq_*`) keeps a strong ring (`hq_pb_pq` is 7/8) and drops unequal to 1/8.
- Putting Householder rectangles on every layer (`hh_*`) does not inherit the mix unequal scores. `hh_wq_pq` is 0/8 unequal. `hh_pb_pq` matches `qr_pb_pq` at 4/8 and 3/8.
- `qr_pb_pq` here is ring 4/8 and unequal 3/8, with the repo-seed ring passing and unequal failing. On the A6000 that same flag was ring 1/8 across seeds and 21/22 at the repo seed. The two machines are not interchangeable.

Per-seed letters are P/F at offsets 0, 101, 202, 303, 404, 505, 606, 707.

| `--init` | ring | unequal |
| --- | --- | --- |
| baseline | P F F P F P P P | P F F F F F F P |
| hid_q | P F F P P F P P | F P P P F P F F |
| qr_pb_pq | P P F P F P F F | F P P F F F F P |
| mix_zb_pq | P P P F F P F P | P P F P P P P F |
| mix_pb_weyl | P P F F P P P P | P P F F F P F P |
| mix_wq_pq | P F P P F P F P | P P F F F P P P |
| hq_pb_pq | P P P P P P P F | F F F F F F F P |

## Priority gates at the repo seed

| gate | `mix_zb_pq` | `mix_pb_weyl` |
| --- | --- | --- |
| ring | PASS (8/8 modes, hq 1.0) | PASS |
| unequal | PASS | PASS |
| stripes | PASS | PASS |
| blobs | PASS (4 modes, hq 0.906) | PASS (4 modes, hq 0.906) |
| hold | POST_CONVERGENCE_FAIL, converged step 1400, failed at 1421 (21 checks) | POST_CONVERGENCE_FAIL, converged step 1400, failed at 1530 (130 checks) |
| shift | FAIL, stay 53/120 | FAIL, stay 86/120 |
| grid100 | FAIL, 14/100 modes, hq 0.183, never full coverage | FAIL, 32/100 modes, hq 0.379, never full coverage |
| rotated100 | FAIL, final 100 modes at precision 0.958 (gate needs 0.97), 0/5 terminal checks | FAIL, final 100 modes at precision 0.961, 0/5 terminal checks |

Both finalists clear the four short transfer gates at the repo seed and miss hold, shift, grid100, and rotated100. grid100 is still a collapse relative to 100 modes. It is milder than the 3/100 reported for `hid_q` on the other CPU screen (14 and 32 modes here).

## Determinism

Every variant: two init-only builds of ring and of unequal, seed offsets 0 and 101, identical `initial-values.pt` bytes and identical tensor hashes (28/28). Across the full 8-seed screen, each variant kept one initial-checkpoint hash per gate. Baseline hashes change with the seed, as a random init should.

## Recommendation

Send `mix_zb_pq` to the A6000 first (`--init mix_zb_pq`: Householder hidden layers, QR input/output layers, zero bias, R2 particle prior). It is the joint winner on this CPU, and it passes both screen gates at the repo seed.

Send `mix_pb_weyl` second if the A6000 cares more about ring rate than unequal mass. Same weight rule, pattern bias, Weyl prior.

Leave `hh_*` and `hq_pb_pq` off that queue. Full Householder rectangles did not help, and the tiled-identity ring specialist gives up unequal mass.

Re-check grid100, rotated100, hold, and shift on the GPU. This CPU screen does not show a finalist that holds the ring after convergence or that covers grid100.
