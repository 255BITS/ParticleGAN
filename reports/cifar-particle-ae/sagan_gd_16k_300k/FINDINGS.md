# Stopped continuation review

The unchanged 200k-to-300k attempt was stopped by user after the 260k checkpoint. Every saved evaluation from 205k through 260k remained worse than the 200k parent (12.5345). Best continuation: 250k, 12.6204; latest saved: 260k, 13.0597. This supports trying another intervention after 60k without a new best, but does not establish a permanent ceiling.

All historical checkpoints retained. Source hashes verified; this interrupted run has no final completion certificate, so partial evaluations are recorded separately. Recommend expanding both G and D attention (per subsequent user steering) from the certified 200k best, comparing at 205k–240k against this existing unchanged trajectory. Identity initialization preserves initial generator outputs. This tests whether adding attention helps late training, not the scratch potential of a deeper architecture. No seed comparison or new baseline needed.
