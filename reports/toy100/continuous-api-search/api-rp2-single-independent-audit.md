# API-RP2 public-constructor single-shift audit

The completed single-shift result is supported. Eager optimizer initialization now lives inside public recipe factories, controlled by `adam_eager_state=True`. The snapshotted worker calls `get_recipe()` and `GANTrainer.step()` and does not mutate optimizer state.

Source ZIP SHA-256: `d95316f81a56599316bbb4469bee6802f18d8486810588bc567601daf8d54832`.
All 19 declared source hashes and retained artifact hashes verify. The package controller/factory/trainer changes are in the snapshot; evaluator, MLP, KA2, shared Adam helpers and prior are unchanged from `fa511ce`. Initial model/RNG hashes match the ordinary public baseline. Initial optimizer state intentionally differs.

The learner receives no evaluator budget or target notification. Rates follow critic disagreement and realized update activity. Noise uses explicitly declared absolute bootstrap windows of 360/720 updates; training has no enforced end. This supports horizon-independent design, not a claim that all noise schedules have disappeared.

Independent raw-data totals:

- Initial arrival 640; 177/177 observations pass through 2400.
- Pre-shift hold 120/120; minimum HQ .966552734375.
- Shifted arrival 2900, 500 updates after the change; 171/171 subsequent observations pass through 4600, minimum HQ .967529296875, eight modes throughout.
- Frozen comparator 0/220. Controller closes at 1546, reopens at 2437, closes at 3170 autonomously.

All 460 observation rows and 4600 rate/controller rows exactly match the earlier worker-mutated CUDA-eager diagnostic in `runs/rp1`. All 46 state receipts agree except full trainer hashes, which include the new recipe option. This does not make eager state equivalent to standard lazy CPU-counter Adam; RP2 is an explicit distinct configuration.

Caveats: standalone LR helpers still expect a numeric horizon; result LR maxima include unused initialization rates; controller labels remain `rp1`, so recipe flags identify RP2. Stationary, delayed/repeated, long continuation, checkpoint, horizon-prefix, matched K3P and own 22-task evidence remain separate qualification requirements. No winner claim.

No training, tests, model execution, extra seeds or worker edits were performed.
