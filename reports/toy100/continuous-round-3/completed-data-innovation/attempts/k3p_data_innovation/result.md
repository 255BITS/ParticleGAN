# K3P data-innovation attempt 3774450

No formulation passed target-shift recovery. **DI2** is the strongest partial lead in this lane: own ring hold 1200/1200, extension 300/300, stationary 5/5, pre-shift hold 120/120, deadline recovery **77/81**. The four misses are the last checks, 3570–3600, so there is no stable suffix and the gate is FAIL. K3P stays the selected base. This attempt does not promote anything and does not stop other searches.

Parent, not rerun: hold 1200/1200 (min HQ 0.9072265625), extension 300/300 (min HQ 0.98779296875), recovery 28/81 with sustained delay 1130. Pinned source `reports/toy100/gap-fill-20260925/sources/k3p/mechanism.py` still hashes to `d2eb08ee932b288cbba25cd1e7be3a9572b129bd1baf0b79718be1eb37ba9391`. Each candidate copies that bundle and changes only `mechanism.py`.

## Leaderboard

| Rank | Candidate | Hold | Extension | Stationary | Pre-shift | Deadline recovery | Toys | Stress / seeds / 30000 |
|---|---|---:|---:|---|---|---|---|---|
| — | K3P parent (prior evidence) | 1200/1200 | 300/300 | prior pass | prior pass | FAIL 28/81, delay 1130 | 22/22 prior | not this attempt |
| 1 | di2 persistent latch | **PASS** 1200/1200, min HQ 0.9072265625 | **PASS** 300/300, min HQ 0.98779296875 | 5/5 | 120/120, min HQ 0.90869140625 | **FAIL 77/81**, suffix 0, misses 3570–3600 | NOT_RUN | NOT_RUN |
| 2 | di3 latch then anchor snap | **PASS** same minima | **PASS** same minima | 5/5 | 120/120 | **FAIL 45/81**, suffix 0 | NOT_RUN | NOT_RUN |
| 3 | di1 proportional reopen | **PASS** same minima | **PASS** same minima | 5/5 | 120/120 | **FAIL 33/81**, suffix 0 | NOT_RUN | NOT_RUN |

All three stationary rings matched the parent's reported hold and extension minima. Mobility stayed 0 on those runs, so the K3P schedule was left in place. That is one run each, not a tensor-level identity proof.

## Rules

Real minibatches update a slow mean and variance (EMA 0.01). The score is the batch-mean residual divided by the reference standard deviation over sqrt(batch size). Batch size on these ring runs was 128. The first batch initializes the reference and scores 0, so cold start is not treated as a change. Nothing reads the training budget, scores, convergence, targets, shift time, or centers.

Rates stay on K3P's horizon schedule while mobility is 0. While mobility is positive, each multiplier is pulled toward max(schedule, 0.20) and never below the schedule. Critic mixing `s` uses the scheduled network multiplier, so a reopen does not restore the early penalty or turn off the 0.999 EMA anchor. Guard, direct particle response, sparse latent rule, auxiliary host losses, architecture, seed 1234, and step budgets are the copied K3P files. Input and output noise remain K3P's horizon schedule. That noise is a labeled intermediate, not a horizon-independent result.

- **DI1** maps mobility with a 0.05 leak toward clamp((z_rms − 3) / 3, 0, 1). The shift reached z_rms 3.67 and mobility about 0.064. The network multiplier peaked near 0.022 and was gone within about 100 updates. `s` stayed 0. Adam continued (3600 updates, nonzero displacement). Deadline passes were 33/81 with no suffix.
- **DI2** opens mobility to 1 only after five consecutive batches with max |z| ≥ 3.5, holds that open for 200 further steps after evidence stops, then multiplies mobility by 0.97. Stationary z_max peaked at 3.488, so the hold never latched. On the shift, z_max reached 6.599 and the latch started at penalty call 2429. While open, critic and generator LR were 0.00085 and prior LR was 0.0017 (0.20 of the initial 0.00425 and 0.0085). `s` stayed 0. By the deadline the multiplier was back near the 0.01 floor. Modes were 8 with HQ ≥ 0.99 from step 2800 through 3560, then 7 modes at HQ 0.9109, 0.9087, 0.9060, 0.8972. Real-batch z was already near 1. The tail is generator-side drift after the data signal had closed.
- **DI3** is DI2 plus one copy of the current critic into the anchor when the 200-step hold ends. The stationary hold still never snapped (anchor_snaps 0). The shift snapped once. Deadline passes fell to 45/81, with a miss block at 2870–3160 and another at 3550–3600. Final live state was 7 modes, HQ 0.753. The hard retarget of the anchor cost precision that DI2 had held.

Probe `rate_ranges` min/max still span the full K3P schedule (4.25e-5 to 0.00425 for G/D, 4.25e-4 to 0.0085 for the prior), because the 0.20 reopen sits inside that span. Applied reopen rates are in the run logs and in `regularizer_receipt` (`innovation_trace`, `rate_trace`, `applied_network_max`). At the shift checkpoint the recorded rates are still the floor: the boost starts on the following updates. Optimizer moment counters were 2400 at the shift and 3600 at the end on both optimizers. No counter reset.

No extra critic forwards were added beyond K3P's existing anchor evaluation. The innovation uses the real batch already passed into the penalty.

## Budget dependencies that remain

The detector, the 3.5 streak, the 200-step hold, the 0.97 close, and the 0.20 ceiling do not read `total_steps`, the cap, or the shift time. The rate baseline still does. `phase_multipliers` calls K3P's schedule, which uses the control horizon (these drivers pass noise horizon 1200), anneal start 0.6, network floor 0.01, prior floor 0.05, and `network_lr_horizon_cap` 1600. Input and output noise still anneal on that 1200-step horizon. A different declared horizon would change the learning-rate and noise prefix. No second-horizon replay was run. This is not a horizon-independent formulation.

Cold acquisition is the K3P schedule. A confident real-data reference forms in a few dozen batches, while the ring still needs the long high-rate portion of that schedule (8 modes and HQ 0.992 at step 600, convergence at step 1400). Using the data score to close the rate would end learning during acquisition. The reference is initialized from the first batch so cold start does not look like a shift.

## Limit

Real-mean innovation cannot see generator-side forgetting, or a change that preserves the batch mean. DI2's 3570–3600 loss happened with the data score back at the null. DI3 shows that snapping the anchor when the latch ends does not repair that loss. Long-term stability was not measured. The 30000-update protocol, the second shift, delayed shift, frozen control, sensitive toys, the 22-gate matrix, and the extra native seeds were not run. Live recovery never returned UNCONFIRMED, so the matched frozen control was not started. A frozen run would still be a true no-adaptation control: the driver stops Adam after the shift, and these mechanisms only change rates and, in DI3, the anchor copy.

## Replay

Environment on every benchmark process: `CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `PYTHONHASHSEED=0`. Python: `/tmp/pr38-default-env/bin/python`. Frozen runtime: `/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda`. Fixture: `/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt`.

```sh
# DI2 hold + 300-update extension (PASS, 106.334 s)
/tmp/pr38-default-env/bin/python -u \
  reports/toy100/data-innovation-3774450/di2/hold.py \
  --repo <frozen runtime> \
  --config reports/toy100/data-innovation-3774450/di2/config.json \
  --task mode_hold --backend cuda --initial-state <mode_hold fixture> \
  --output <fresh dir> --network-floor 0.01 --prior-floor 0.05

# DI2 target shift (FAIL 77/81, 131.649 s)
/tmp/pr38-default-env/bin/python -u \
  reports/toy100/data-innovation-3774450/di2/shift.py \
  --repo <frozen runtime> \
  --config reports/toy100/data-innovation-3774450/di2/config.json \
  --task mode_hold --backend cuda --initial-state <mode_hold fixture> \
  --output <fresh dir> --network-floor 0.01 --prior-floor 0.05
```

DI1 and DI3 use the same drivers under `di1/` and `di3/`.

## Artifacts

- Sources: `repo/reports/toy100/data-innovation-3774450/di{1,2,3}/` (only `mechanism.py` differs from the pinned K3P hashes).
- DI1 hold / shift: `runs/di1-hold/result.json` (109.156 s), `runs/di1-shift/result.json` (129.093 s).
- DI2 hold / shift: `runs/di2-hold/result.json` (106.334 s), `runs/di2-shift/result.json` (131.649 s).
- DI3 hold / shift: `runs/di3-hold/result.json` (109.546 s), `runs/di3-shift/result.json` (122.030 s).
- Gate log: `tests.jsonl` beside this file. Six rows, all PASS or FAIL. No ERROR.

## Next mechanism

Keep DI2's five-batch real-mean latch. It is the first rule in this lane that preserves the stationary ring and still applies a real 0.20 reopen with the anchor and `s` left on the K3P clock. The remaining miss is one mode lost after that signal has already returned to the null. A further real-data threshold will not see it. The anchor snap in DI3 is a failed repair. A later proposal needs a generator-side maintenance signal that stays quiet on a stationary target, which is the failure RP1 showed when a gradient reopen fired around step 5672 and center error worsened.
