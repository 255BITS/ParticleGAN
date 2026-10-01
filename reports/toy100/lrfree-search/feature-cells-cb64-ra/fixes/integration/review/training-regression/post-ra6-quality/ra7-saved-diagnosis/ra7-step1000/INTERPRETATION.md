# RA7 scheduled500/1000 diagnosis

The unchanged frozen framework analyzed the available sealed checkpoint prefix0/100/250/500/750/1000/1250 in one pass. This report covers the requested500 and1000 milestones; no separate750 or1250 job was run. CPU source/checkpoint tensors, hashes and global RNG are unchanged; no CUDA, emissions, training, virtual optimizer steps, proposals or new seeds. FAST clean counts match the saved evaluator exactly.

RA7 quarters G and learned log-sigma optimizer base rates while preserving prior/D bases. The noise formula, floor, mode, serving rule and official evaluator are unchanged. Population-clock validity is owned by the independent state reviewer; all observations below refer to saved endpoints.

## Update500

| Variant | Clean fast P / modes | Clean EMA P / modes | Saved emitted P / modes / TV |
| --- | --- | --- | --- |
| RA7 | .402344 /21 | .656250 /24 | .355469 /20 /.644531 |
| RA6 | .621094 /23 | .715820 /23 | .525879 /24 /.474121 |
| RA4 | .659180 /25 | .875000 /25 | .583252 /25 /.416748 |
| E22 | .357422 /17 | .511719 /22 | .305908 /18 /.694092 |

RA7 FAST misses modes8/13/18/23; EMA misses8. The current CPU refit has243 inside eligible rows and physical unique-copy capacity upper bound125. Annotated mode8 has no inside eligible parent despite target53 and vacancy52; its four raw-supported rows are rejected by p>Q or inside. Other missing FAST modes retain2/4/1 inside parents and usable capacity, respectively. Labels annotate diagnostics only.

At500,246/248 attempted target cells have produced paired births over62 reactions. Latest496 makes2 mass copies+26 local copies+4 new births+19 global copies=51 ordinary actions, no isolation. Four updates later, those four rows have FAST raw support2/4 and current learned p>Q/inside1/4; EMA passes both4/4. Birth-time raw status and historical GPU partition were not saved. Population is inactive, s1/current b16, latest312 inconclusive at tested b8, no coverage rejection or expiry.

## Update1000

| Variant | Clean fast P / modes | Clean EMA P / modes | Saved emitted P / modes / TV |
| --- | --- | --- | --- |
| RA7 | .611328 /25 | .794922 /25 | .536499 /25 /.463501 |
| RA6 | .781250 /25 | .989258 /25 | .697266 /25 /.303140 |
| RA4 | .621094 /17 | .941406 /22 | .858276 /23 /.186836 |
| E22 | .706055 /25 | .708008 /25 | .609985 /25 /.390015 |

All25 annotated reference modes have inside parents at1000, minimum supply4; current physical copy capacity upper bound198. The latest reaction1000 makes8 local copies+4 new births+39 global copies=51. All four newborn rows pass raw support and current learned p>Q/inside for both models at checkpoint age0. Cumulative acceptance is473/478 attempts over125 reactions;500→1000 contains227 births over63 reactions. Absent-mode support is no longer the sole limitation at this endpoint.

Population remains inactive and FAST serves. Latest696 tested b16 reports both negative pair verdicts but coverage591/973, so its scheduling decision is inconclusive and current b32 follows. There is one coverage rejection and no expiry. This is a saved decision description, not a new stationarity inference.

## Realized adaptive rates

At500 RA7 effective G LR is .000265625 with Gtester s=.25. At1000 it is .0001328125 with s=.125, equal to RA6's effective G LR at1000, where RA6 s=.03125. Thus the quarter base rate does not guarantee a quarter effective rate across different adaptive trajectories. Prior LR remains .0085, D is about .0031875 and saved output sigma is .029. Sigma group's base rate is .0010625, with unchanged noise semantics. The earlier G-only virtual comparison did not simulate these different learned controller histories.

## Fixed-coordinate generator/table motion

| Crossed saved state | Fast P / modes | EMA P / modes |
| --- | --- | --- |
| G250 / z250 | .204102 /8 | .346680 /16 |
| G500 / z250 | .220703 /9 | .248047 /8 |
| G250 / z500 | .316406 /16 | .539063 /23 |
| G500 / z500 | .402344 /21 | .656250 /24 |
| G1000 / z500 | .261719 /11 | .393555 /17 |
| G500 / z1000 | .444336 /19 | .619141 /23 |
| G1000 / z1000 | .611328 /25 | .794922 /25 |

At fixed z250, G500 retains106/209 old FAST-supported coordinates and205/355 EMA, all in their old mode. At fixed z500, G1000 retains199/412 FAST and360/672 EMA, again all in their old mode. Generator/table coadaptation remains substantial despite the lower configured G base. These nonlinear crossings separate measured quantities but do not provide additive causal effects or identify a historical next gradient.

Offline old newborn-coordinate fitness under fixed old CPU heads is retained in each motion receipt. Row indices do not establish that those birth incarnations survived later reactions. Initial parent capacities omit a subsequent reaction's reservations; CPU geometry is not historical GPU replay.

Evidence: `receipt.json`, `comparison-summary.json`, `motion-0250-0500.json`, `motion-0500-1000.json`, captured metric prefixes and logs. RA6 is reused from its frozen final receipt; RA4/E22 are the unchanged framework controls. These intermediate outcomes establish no strict final quality pass and select no serving or noise override.
