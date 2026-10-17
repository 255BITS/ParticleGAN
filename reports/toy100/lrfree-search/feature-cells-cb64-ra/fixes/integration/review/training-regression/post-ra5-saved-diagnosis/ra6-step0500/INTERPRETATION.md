# RA6 saved update500

Read-only CPU analysis from sealed checkpoints0/100/250/500 and captured matched metrics. Frozen sources/checkpoints, checkpoint tensors and global RNG remain unchanged; no CUDA, new seeds, training, proposals or emissions. Computed FAST clean counts match saved serving counts exactly. No quality verdict is assigned.

| At500 | Clean fast P / modes | Clean EMA P / modes | Saved emitted P / modes |
| --- | --- | --- | --- |
| RA6 | .621094 /23 | .715820 /23 | .525879 /24 |
| RA4 | .659180 /25 | .875000 /25 | .583252 /25 |
| E22 | .357422 /17 | .511719 /22 | .305908 /18 |

RA6 emitted TV is .474121. FAST serves; its missing clean modes are8/18, while EMA's are11/18. Coverage improved sharply since250, but the strict final precision/coverage/TV gates remain unqualified at this milestone. Population scheduling is inactive with retained drift decision, s=1, b=16, zero stationary participants and no coverage rejection/expiry events.

## Births and accessibility

Between250 and500,31 reactions accepted120 paired births,3.871 per reaction. Cumulative acceptance is242/244 target attempts over62 reactions. The latest496 reaction made47 copies+4 births=51 ordinary actions. At the checkpoint four updates later, two latest newborn rows pass raw support and current CPU p>Q/inside for FAST, and two pass for EMA. One latest row has different nearest fast/EMA modes20/21. Acceptance at the reaction used both models' unchanged learned gates; birth-time raw oracle status was not saved.

The current CPU refit has400 flagged rows,574 p>Q rows and419 inside eligible rows; unique-copy physical capacity upper bound190. Missing FAST modes8/18 have6/3 inside eligible parents, physical copy capacity6/3, real targets53/48 and cell vacancies44/40. Their raw-supported counts9/8 remain below11 required clean rows for coverage. Sparse supply remains relevant, but these modes are currently accessible to copying; this checkpoint does not reproduce an absent-parent or blocked-target implementation error.

Among636 raw-supported fast rows,86 fail p>Q and139 additional rows fail inside, leaving411 raw-supported eligible inside rows. Eight further learned eligible inside rows are outside the raw oracle radius. These thresholds and annotations are unchanged; they describe the current learned/raw mismatch, not a proposed gate adjustment. Reference cell labels are annotations only, and initial capacity excludes later action reservations.

## Fixed-coordinate generator/table comparison

| Generator / latent table | Fast P / modes | EMA P / modes |
| --- | --- | --- |
| G250 / z250 | .185547 /6 | .224609 /8 |
| G500 / z250 | .152344 /2 | .152344 /5 |
| G250 / z500 | .303711 /14 | .483398 /21 |
| G500 / z500 | .621094 /23 | .715820 /23 |

With z250 held fixed, G500 retains raw support for67/190 previously supported FAST coordinates;64 retain the same mode. EMA retains102/230;101 retain the mode. Thus the actual improving table coexists with substantial generator-induced churn. Saved latent-table changes include optimizer updates, births and copies, so the four evaluations do not provide an additive causal decomposition.

For the four saved250 coordinate values associated with the latest248 births, raw support under G500 is2/4 FAST and2/4 EMA, compared with3/4 and4/4 at250. Holding the saved250 critic/head partition fixed, two remain p>Q/inside for FAST and none for EMA. These values are an offline cohort, not surviving row incarnations:31 intervening reactions may overwrite the same row indices. Current496 newborn fitness and fixed248-coordinate fitness have distinct scopes. No concrete production implementation error is reproduced.

Evidence: `receipt.json`, `motion-comparison.json`, captured metrics and pinned source/checkpoint maps. CPU support refits do not replay historical GPU geometry. Later final quality is determined by the root's unchanged CUDA evaluator and canonical gates.
