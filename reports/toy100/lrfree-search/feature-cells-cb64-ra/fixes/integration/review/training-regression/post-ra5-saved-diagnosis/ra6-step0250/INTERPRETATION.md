# RA6 saved update250

CPU read-only analysis of sealed checkpoints and captured metric prefixes. Frozen package, checkpoint tensors and global RNG are unchanged. No CUDA, emissions, proposals, updates or new seeds. The framework's historical RA5 title also serves RA6; the input/source maps identify the actual package.

| At250 | Clean fast P / modes | Clean EMA P / modes | Saved emitted P / modes |
| --- | --- | --- | --- |
| RA6 | .185547 /6 | .224609 /8 | .173462 /4 |
| RA4 | .281250 /12 | .458984 /20 | .261475 /14 |
| E22 | .191406 /8 | .259766 /15 | .177246 /6 |

RA6 serves FAST, and computed clean served counts match the saved evaluator exactly. Population scheduling remains drift/inactive, with no population coverage rejections or expiries. This is an intermediate diagnostic, with no quality verdict.

## Birth actuation and parent supply

The100-to250 interval contains19 reactions and76 accepted paired births, four per reaction. Cumulative acceptance is122/124 attempts. The latest reaction at248 made47 copies and4 births, using the51-action ordinary cap. At250, three of those four rows pass both raw support and the current CPU p>Q/inside gates for FAST; all four pass for EMA. Birth-time GPU p values were accepted, but birth-time raw oracle support and exact GPU geometry were not saved.

The current saved critic/FIFO CPU refit has25 learned groups and64 oracle-pure reference cells. Group-level oracle purity is not recorded. It has871 flagged rows,150 p>Q rows,111 inside eligible rows and initial unique-copy capacity upper bound103. Annotated modes10/11/19 have no inside eligible parent despite real targets30/48/37 and positive vacancies. All25 raw modes have at least one supported fast row, but only six exceed the unchanged coverage threshold. Mode labels are annotations; they do not enter the policy.

## Generator motion at fixed saved coordinates

These four clean-table evaluations hold either a saved generator or a saved latent table fixed. They use no sampling or mutation.

| Generator / latent table | Fast P / modes | EMA P / modes |
| --- | --- | --- |
| G100 / z100 | .101563 /1 | .111328 /1 |
| G250 / z100 | .086914 /2 | .096680 /2 |
| G100 / z250 | .134766 /1 | .136719 /4 |
| G250 / z250 | .185547 /6 | .224609 /8 |

With z100 fixed, only16/104 previously raw-supported fast coordinates retain support under G250;15 retain their mode. EMA retains22/114, all22 in the same mode. G motion is therefore sufficient to displace many supported coordinates even though the actual new table improves. The comparison is nonlinear and does not assign additive causal contributions to G versus latent learning/copies/births.

For the four saved100 coordinate values associated with the latest96 birth, fast raw support goes2/4 to0/4 under G250; EMA goes4/4 to0/4. Their features also become outside and p<=Q when the saved100 CPU critic/head partition is held fixed. These are offline fixed-coordinate values. Nineteen intervening reactions prevent any claim that those birth incarnations remained in the250 table, or that their actual lifetime is150 updates. The observation reproduces generator-induced fitness movement, not an acceptance/accounting implementation error.

Evidence: `receipt.json`, `motion-comparison.json`, captured metric prefixes and source/input maps. CPU support geometry is descriptive, not historical GPU replay.
