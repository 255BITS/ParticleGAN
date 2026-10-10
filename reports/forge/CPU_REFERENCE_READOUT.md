# Bounded CPU reference readout

The frozen K3P baseline passed `trajectory`, `residual_student`, and `unipolar`
for **20.864916388 CPU wall seconds**. These are three independent task passes,
not a positive verdict for the full 16-task reference. Its other 13 reference
tasks and all 32 ablation reference cells remain unmeasured.

The [registered selection](calibration-lanes/current-k3p-mog-cpu-reference-a-v2/registration.json)
fixed three 400-update tasks, seed 0, one CPU thread, and a 5,400-second reservation
ceiling before execution. Its [contract](../../configs/forge/campaigns/current-k3p-mog-cpu-reference-a-v2.json)
uses diagnostic evidence only. No GPU capacity, additional seed, changed
threshold, or ordinary qualification was involved. All three attempts completed;
the campaign has zero reserved seconds. Central logs remain at
`runs/forge/calibration-current-k3p-mog-cpu-reference-a-v2/progress.jsonl` in the
main checkout's shared queue.

| Task | Sustained live verdict | Terminal metrics | Passing suffix / 24 checks | Paid wall seconds |
| --- | --- | --- | ---: | ---: |
| [trajectory](attempts/beb49d1044cc48c1882fd958055a1522/result.json) | PASS | identity MSE .000918549 <= .02 | 22 | 6.821018327 |
| [residual_student](attempts/270c483fe3b94932b7f7edcecd89c872/result.json) | PASS | identity MSE .000902660 <= .02; success 1; wrong-pad rate 0 | 19 | 6.874851209 |
| [unipolar](attempts/6e2b5fdf4c8e4ddaba57316096f777a2/result.json) | PASS | coverage .998118 >= .85; neutral hold .989503 >= .85; off-caption .000008545 <= .05 | 19 | 7.169046852 |

Each independently graded terminal suffix exceeds the five-check minimum.
The table compares different tasks, not competing formulations or training
speed. Cost includes supervised runner and evaluator time; FLOPs are unavailable.
These hosts explicitly use learned particle clouds or nonsampled parameter
controls. Their passes do not establish sampled learned-MoG quality.

Source remains `c673226cad226889b05269d71786100dcfe2122c8fbe880067d2d2ebd79ab32d`,
and K3P revision remains `229770eceb2d51236085562985c341033df825c4c85fc20adbebdef313f8c65b`.
The [updated readout](records/readout-acc70fcaf6cedb9a3772e41e.json) covers all six
v2 baseline attempts. The [calibration reduction](calibration/current-k3p-mog-v2.json)
retains `UNKNOWN` for every complete-reference decision and `BLOCKED` adoption.

## Stop spending toward an impossible adoption decision

The [v2 smoke evidence](CURRENT_SMOKE_READOUT.md) already records `FAIL` for
all three declared lineages. Under the unchanged
[criteria](../../configs/forge/calibration/criteria-v1.json), this exact profile
cannot be accepted, regardless of the remaining reference results:

- If at least one full reference passes, every reference-positive lineage is a
  false rejection. That violates the required zero false-reject fraction.
- If no full reference passes, the minimum of one positive reference is unmet.
- Incomplete references remain unknown and cannot satisfy complete adoption.

This is a feasibility argument, not a measured false-reject rate. The three CPU
passes do not make K3P reference-positive. Do not fill the remaining v2 matrix
solely to seek acceptance, drop failed smoke tasks from this profile, or relax
its criteria.

The next justified screen study is the already-declared historical alternative:
`mode_hold`, `img_bars4`, and `img_intensity2`, with its unchanged task thresholds
and the same 16 independent references. Register that question as a separate
current profile before spending, retain v2 as a failed adoption design, and
require exact evidence compatibility for any reuse. Its historical replay also
failed adoption, so it is a hypothesis to test rather than a presumed solution.

That [separate study is now declared](QUICK_SCREEN_STUDY.md), with the three
existing CPU reference cells imported explicitly and zero additional training.

The [physical GPU pilot](MULTI_GPU_PILOT.md) has a distinct operational question
and remains registered but unlaunched pending compute ownership. This calibration
finding does not establish or replace its multi-GPU acceptance evidence.
