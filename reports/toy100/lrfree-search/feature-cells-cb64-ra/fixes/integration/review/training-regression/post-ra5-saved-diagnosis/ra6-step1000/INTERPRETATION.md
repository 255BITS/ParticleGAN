# RA6 saved update1000

Read-only CPU diagnosis with unchanged frozen sources, checkpoint tensors and global RNG; no CUDA, emissions, proposals, updates or new seeds. Captured metrics are authoritative, and FAST clean counts reproduce the saved evaluator exactly. Intermediate checkpoint scores do not qualify final quality.

| At1000 | Clean fast P / modes / TV | Clean EMA P / modes / TV | Saved emitted P / modes / TV |
| --- | --- | --- | --- |
| RA6 | .781250 /25 /.223750 | .989258 /25 /.071758 | .697266 /25 /.303140 |
| RA4 | .621094 /17 /.419180 | .941406 /22 /.139219 | .858276 /23 /.186836 |
| E22 | .706055 /25 /.293945 | .708008 /25 /.291992 | .609985 /25 /.390015 |

## Controller and current support

RA6 serves FAST. Its prior population is inactive; s=1 and b=64. The latest904 scheduling decision is inconclusive, with525 participating rows versus973 required. Two population coverage rejections have occurred, with no stationary population or expiry event. EMA clean quality cannot be substituted for the serving decision or emitted evaluation.

All25 annotated reference modes have inside eligible parents, with minimum supply7. Current CPU head/FIFO refit counts134 flags,744 p>Q rows,556 inside eligible rows and initial physical unique-copy capacity upper bound122. Current remaining precision loss is therefore not explained by absent-mode support or globally inaccessible real targets. Of800 raw-supported FAST rows,90 fail p>Q and160 additional rows fail inside, leaving550; six further learned eligible inside rows are raw-unsupported. The learned/raw distinction remains material.

## Births and an observed-reference tail

Cumulative births are459 from476 attempts over125 reactions. The500-to1000 interval accepted217 births over63 reactions. Latest reaction1000 makes3 local copies+2 new-latent births+46 global copies=51 ordinary actions, with no isolation. At checkpoint age0, child0 is raw-supported for both models, while child7 is raw-unsupported despite logged live/EMA p=1 and current CPU p=1/inside.

The additional raw annotation places child7 at oracle distances .097576 FAST and .097505 EMA, beyond the unchanged .09 radius. Its selected even real FIFO reference728 is itself outside at .097404. Direct distances from the new points to that observed reference are .000305/.000185. This is compatible with learning an observed-reference tail, rather than later G motion for this age0 example. The selected reference is not asserted to be the historical GPU target. Float32 cdist rounded these tiny reference distances to zero; original evidence is retained and the direct-norm correction is explicit. No production gate or accounting violation is reproduced.

## Fixed500 coordinates versus generator and table motion

| Generator / latent table | Fast P / modes | EMA P / modes |
| --- | --- | --- |
| G500 / z500 | .621094 /23 | .715820 /23 |
| G1000 / z500 | .508789 /20 | .573242 /19 |
| G500 / z1000 | .697266 /25 | .915039 /25 |
| G1000 / z1000 | .781250 /25 | .989258 /25 |

G1000 at fixed z500 preserves465/636 FAST and562/733 EMA previously raw-supported coordinates, all in their original mode. The actual new latent table substantially improves both models. These nonlinear counterfactuals separate measured quantities but do not allocate additive causal effects among G optimization, prior optimization, copying and births. The four offline coordinate values associated with the saved496 birth also have their fixed-old-head support recorded in `motion-comparison.json`; intervening actions preclude an incarnation-survival claim.

Evidence: `receipt.json`, `motion-comparison.json`, `birth-raw-annotation.json`, `birth-raw-distance-correction.json` and captured metric prefixes. CPU head partitions are refits, not historical GPU replay. Mode labels enter annotations only.
