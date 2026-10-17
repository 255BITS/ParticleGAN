# RA6 saved update100

Read-only CPU diagnosis completed from sealed checkpoints0/100, frozen RA6 source, captured metrics prefixes and matched RA4/E22 states. No emissions, proposals, updates, new seeds, CUDA contexts or frozen writes. CPU clean served counts match the saved evaluator exactly.

| At100 | Clean fast P / modes | Clean EMA P / modes | Saved emitted P / modes |
| --- | --- | --- | --- |
| RA6 | .101563 /1 | .111328 /1 | .117554 /1 |
| RA4 | .108398 /0 | .135742 /4 | .116211 /0 |
| E22 | .105469 /1 | .114258 /3 | .113037 /0 |

These are early stages. The saved prior tester reports drift, s=1, population inactive and zero population expiry/rejection events. Serving is FAST. No quality gate conclusion follows.

## Births and current fitness

There were46 accepted paired births from48 attempted target cells over12 reactions:3.833 births/reaction on the checkpoint interval. The latest reaction at96 accepted four births with live p=.908–.967 and EMA p=.897–.977 under the frozen birth-time support/cell/inside gates. It made36 copies+4 births=40 ordinary actions within the51 cap.

At the saved checkpoint four updates later, those four newborn rows have raw oracle support2/4 for the fast generator and4/4 for EMA. In the newly fitted CPU head geometry, fast p>Q holds for1/4 and inside for1/4; EMA passes both for4/4. Current fast nearest modes are17,12,16,2; EMA modes17,12,21,2. The raw labels are annotations only. Exact birth-time raw oracle status and GPU geometry were not saved. The observed difference is compatible with changing live G/prior/head during those four updates; it does not establish a birth-time acceptance bug or a same-GPU-law expiry result.

## Parent supply

The saved current critic/FIFO CPU refit has25 learned groups and64 oracle-pure reference cells. Group-level oracle purity is not recorded. There are942 flagged current rows,81 rows with p>Q,51 inside eligible rows and an initial physical unique-copy capacity upper bound49. Real cell vacancies sum944 and group vacancies942. Annotated modes0,7,15,20 have no inside eligible parent despite positive real targets38,38,49,41, respectively. Another annotated mode6 has one inside parent but no usable cell vacancy at that parent cell. These are initial-state capacities, without a reaction's row reservations.

Among104 raw-supported fast rows,32 fail p>Q and22 more fail the inside boundary, leaving50 raw-supported inside eligible parents; one additional learned eligible inside row lies outside the raw oracle radius. The support/count policy remains label-free. The first100 updates show successful bounded learned-anchor actuation plus continued sparse/noisy support throughout the table, similar to the matched early controls. They do not demonstrate durable all25-mode coverage or P>=.90.

Receipt: `receipt.json`. Historical GPU target cell IDs are preserved as metadata and are not reused as IDs in the CPU refit. Later checkpoint intervals can report birth averages, population expiry deltas and lost/restored modes. Unlogged intermediate row overwrites prevent exact birth-incarnation survival claims.
