# Family D deterministic inits

One patch, `particlegan/det_init.py`, selected with `--init`. Default training does not call `install`, so the PyTorch init is unchanged. Recipe knobs (LR, schedules, clips, loss coefficients) are untouched.

Parents, included so this CPU build can be compared with the earlier screens:

| `--init` | weights | bias | particle prior |
| --- | --- | --- | --- |
| `hid_q` | Householder on square hidden layers, tiled scaled identity on rectangles, center-tap conv | Kronecker-Weyl at 1/4 of the declared bound | Kronecker-Weyl |
| `qr_pb_pq` | QR orthogonal, declared RMS | hash pattern at the declared std | Roberts R2 |

Hybrids:

| `--init` | weights | bias | particle prior |
| --- | --- | --- | --- |
| `hq_pb` | hid_q | qr pattern | Weyl |
| `hq_pq` | hid_q | quarter Weyl | R2 |
| `hq_pb_pq` | hid_q | qr pattern | R2 |
| `hq_zb_pq` | hid_q | zero | R2 |
| `mix_pb_pq` | Householder hidden, QR input/output | qr pattern | R2 |
| `mix_wq_pq` | Householder hidden, QR input/output | quarter Weyl | R2 |
| `mix_pb_weyl` | Householder hidden, QR input/output | qr pattern | Weyl |
| `mix_zb_pq` | Householder hidden, QR input/output | zero | R2 |
| `qr_wq_pq` | QR, declared RMS | quarter Weyl | R2 |
| `hh_pb_pq` | Householder semi-orthogonal on every layer, declared RMS | qr pattern | R2 |
| `hh_wq_pq` | Householder semi-orthogonal on every layer, declared RMS | quarter Weyl | R2 |
| `hh_zb_pq` | Householder semi-orthogonal on every layer, declared RMS | zero | R2 |

Every value is a function of shape and construction order. The torch seed is not read. Host `copy_` / `zero_` after the random draw still wins (identity generator, explicit zero bias).
