K3P still leads. Removing the horizon from the noise and forcing the critic anchor on did not produce a hold plus a timely recovery.

The parent was not rerun: 22/22 toys, hold 1200/1200, extension 300/300, recovery 28/81. All three candidates keep that lineage and replace only the noise rule, the anchor clock, and the rate map. Output noise is constant at 0.029. Input noise is the Adam surprise ratio times the declared peak 0.5, and it fell to zero once that ratio settled. The anchor is in every penalty. Toy screens and the stress tests were not opened.

| Candidate | Hold | Recovery deadline | Delay |
|---|---|---|---|
| K3P parent (published) | 1200/1200, extension 300/300 | 28/81 | 1130 |
| sn3 asymmetric rate | 156 updates, then fail at step 2679 | 80/81 | 550 |
| sn1 constant full rate | 32 updates, then fail at step 3818 | 44/81 | 890 |
| sn2 symmetric reversal | 23 updates, then fail at step 1424 | not run | |

sn3 is the closest. After the shift at step 2400 the optimizers kept updating, and the only deadline miss is step 2940 (7 modes). The critic learning rate never left the band 0.00269–0.00425. The parent holds near 4.25e-5. Mixed gradient signs released the reversal memory before it could reach that floor, so the ring still broke.

Details, replay commands, and the remaining horizon reads are in `result.md`. Gate rows are in `tests.jsonl`.
