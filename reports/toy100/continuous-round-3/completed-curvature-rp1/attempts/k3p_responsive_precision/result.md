# k3p_responsive_precision — rp1_signal_close

K3P stays the selected base. Nothing here is promoted. `current-research-base.json` was not edited. The pinned parent was not rerun and was not modified.

Parent evidence, unchanged: 22/22 GPU toys, ring hold 1200/1200 (min HQ 0.90723), extension 300/300 (min HQ 0.98779), target-shift deadline FAIL 28/81, delay 1130. Parent mechanism `d2eb08ee932b288cbba25cd1e7be3a9572b129bd1baf0b79718be1eb37ba9391`.

One proposal. Learner files are frozen at the hashes below. Later observation drivers live outside `candidates/rp1/` and do not change those bytes.

## Leaderboard

Ranked by hold, extension, and timely recovery.

| Candidate | Own hold | Extension | Shift pre-hold | Deadline recovery | Frozen control | Toys |
|---|---|---|---|---|---|---|
| K3P parent (not rerun) | 1200/1200 | 300/300 | parent evidence | FAIL 28/81, delay 1130 | parent evidence | 22/22 |
| **rp1_signal_close** | **PASS 1200/1200**, min HQ 0.96851, converged step 1400, 0 settling failures | **PASS 300/300**, steps 2601–2900, min HQ 0.96802, min modes 8 | **120/120**, stationary 5/5, min HQ 0.97559 | **81/81**, delay 310, stable from step 2710, deadline min HQ 0.91382, min modes 8 | **0/81** and full-probe `match_frozen_control` PASS | **grid100 seed 1234 FAIL**: coverage PASS, accuracy FAIL. rotated100 and staggered100 not scored |

Early post-shift checks before the deadline failed at steps 2410, 2420, 2430, 2440, 2450, 2460, 2470, 2480, 2500, 2510, 2680, and 2700 (12 of 120 recovery checks). The deadline window from step 2800 is 81/81. Final live and EMA are 8 modes, HQ 0.99976. Each role took 3600 Adam updates on the live run. Post-shift rates left the floor (generator/critic 0.000884, prior 0.00204) and then decayed back to the floor. This is not a frozen generator.

## Rule

Cold path matches the AP3 diagnostic: critic-gradient RMS versus a peak set after 200 steps and decayed at 0.9997. Full rate and the early penalty until 250 quiet steps, then rate gain and mixing gain decay at 0.99. When mixing gain first hits 0 it stays 0, so the gradient cap and the 0.999 EMA anchor stay on.

AP3 then forced rate gain 0.2 for 800 steps and reset the peak, and the deadline misses were during that forced mobility. rp1 does not arm that dwell and does not replace the peak. A later reopen (level at least 0.5, or fast/slow at least 3 for 3 steps) sets rate gain to 0.2 only. After 250 quiet steps the same 0.99 decay closes it. Another reopen can raise it again. The early penalty is not restored and the base rate is not restored. On the measured shift, level stayed under 0.25, quiet reached 250 near step 2700, and gain decayed from 0.2 to the floor without a second snap-open. Applied network multiplier is `0.01 + 0.99 * rate_gain`. Applied prior multiplier is `0.05 + 0.95 * rate_gain`. Guard remains 5× Adam RMS after 200 steps. Anchor extra critic forwards: hold 2095, shift 2795, anchor starts at penalty call 805.

## Noise

Input noise falls from 0.5 to 0 over a declared 120 updates. Output noise rises from 0 to 0.029 over a declared 240 updates. Those lengths equal K3P's 0.1 and 0.2 of a 1200-update horizon, stored as constants. The schedule does not read `noise_horizon` or the training budget. A formula audit in the mechanism receipt checks steps 0, 60, 119, 120, 240, and 1200 at declared horizons 1200 and 4800 and matches the 1200-horizon K3P curve in both cases.

Remaining budget dependencies, not used by the rate or noise values:

- The frozen driver still requires the `noise_horizon` argument to equal 1200.
- `network_lr_horizon_cap` is still in the config and is still passed by the host. `phase_multipliers` ignores step, total, and that cap.

The paired full-state horizon prefix is owned by the Codex audit lane and was not repeated here. Late and repeated shifts are NOT_RUN. Grid/rotated/staggered seeds 1235–1237 are NOT_RUN; seed 1234 is the native run below.

## Frozen control and state witness

Live driver status was UNCONFIRMED, which is what that driver returns when the quality windows pass. The matched frozen run keeps Adam updates at 2400 after the shift and scores deadline 0/81 (min modes 0, min HQ 0).

The first `match_frozen_control` call restored `mode`, `noise_horizon`, `diagnostic_every`, `dense_after`, `dense_until`, and `freeze_after_shift` from the hashed `shift.py` call, and recomputed `source_sha256` / `runtime` from the runtime tree, because that wrapper omits them. That row is a partial metadata reconstruction. It is not the full-state confirmation.

A second active run and a second frozen run, observation-only, saved the probe fields the wrapper had dropped. `match_frozen_control` on those saved dicts returned PASS. The active diagnostic, stationary window, continued hold, shift pair, and deadline window are identical to the original 81/81 result. Pre-shift hashes match between the active and frozen captures for generator, critic, prior, both optimizers (including latent row history stored in Adam state), generator/prior EMA lists, critic-parameter EMA, controller scalars, response previous-gradient tensors, latent call counts, host stream, CPU RNG, and CUDA RNG.

Witness limits, left pending rather than claimed:

- `means_pre_shift` subtracts the declared shift `(1, 0)` because the host hook runs after the mean update. The check that the difference is `(1, 0)` is tautological. `means_post_shift` is the direct hash, and it matches. `shift_pair` is the host's before/after quality record.
- `latent.stats` was not normalized to parameter index, so per-row scope memory is only inside the optimizer-state hash.
- `response.prior_ids` membership was not recorded.
- Mechanism `pending`, `depth`, and learning-rate bookkeeping beyond the saved controller scalars were not a separate witness.

Two earlier capture attempts ERROR at the hook (device mismatch, then `ema_g` being a list). Those outputs are kept. They are harness failures, not candidate quality failures. Canonical scores come from the original hold and shift.

## Hashes

- mechanism `ed49869c6e06e1ac638cf04dba36b123aa7cb3b883650d7e16aa8843aaa22d14`
- config `a1475108a82f67a93e0cdcd793b920b0cc2b1e1ccf31285974adc3b341b2fca2` (pinned K3P)
- latent `197df6350f5295f7d396f7d3c821808be1d15168d6e5586a89ebfbd403586139` (pinned K3P)
- response `7e71d60a343f9f47e1c16600279364f0482863ce116c00f4657355638615987d` (pinned K3P)
- shift.py `f049e86eac4e1b65212ea2d40c2eee8c9f6a63082a666f6d3f31010f336375a8`
- shift_frozen.py `9970dd0195d65aeccd2ffeaf874f2a5ed0df2e06f0f934192201258e41f63fc6`
- hold.py `2ae5a9551739e6bad6190b76bcf252689793181786239238d0b08d2d1cc0c00e`

Observation adapter, copied unchanged: `3b60a1d9e5ab679376ccfd9efe6559c80e5eef23e4813f776349c709577cdf81`. Wrapper `f49720316080578dad000045c621276859f40b25261256c735cad71d42ba7d21`.

## Native seed 1234

grid100, 7000 updates, through the observation adapter: **FAIL**. Coverage PASS (100 modes, final HQ 0.98905, precision 0.98905, stable from step 2250). Accuracy FAIL on all 5 terminal checks, steps 6000–7000. The only missed limit is live `center_rms_sigma` 0.261, 0.262, 0.278, 0.289, 0.287 against 0.20. Mass TV, absolute covariance-trace bias, and radial KS stay inside their limits. Precision stays about 0.986–0.989. The EMA center is inside 0.20 only at step 6000 (0.198) and is 0.227–0.280 after that. This stops further qualification. Ring recovery does not carry grid-center accuracy.

rotated100 was killed after grid100 failed, during or just after its 7000-step training log and before `result.json`. The log's last eval is live 97/100 modes, HQ 0.8514. That is not a canonical coverage or accuracy verdict. staggered100 was not started.

## NOT_RUN or not scored

- 19 transfer toys, including the sensitive four. verify_p3_gates owns those. Do not read another lane's scores as this candidate's pass.
- rotated100 and staggered100 canonical gates. rotated100 has an interrupted training log only.
- Native seeds 1235–1237. Not started, because seed 1234 grid100 accuracy failed.
- Delayed change, second change, and the 30000-update protocol.
- Paired horizon prefix (Codex lane).

## Next mechanism, not run

The ring failure mode this proposal targeted is closed on this seed: historical critic-RMS scale, no 800-step dwell, no peak reset, partial rate 0.2, then the existing 250-quiet 0.99 decay, anchor left on. The new failure is grid center RMS above 0.20 at the 7000-update accuracy checks, while coverage still passes. A higher rate or a slower close would need its own hold, shift, and grid accuracy; this attempt does not have that candidate.

## Replay

Environment on every benchmark process: `CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `/tmp/pr38-default-env/bin/python`.

```bash
BASE=/ml2/hypergan/gan-attempts/formulations-20260925T172651Z/k3p_responsive_precision/20260925T172651Z-3715977/repo/reports/toy100/k3p-responsive-precision
RUNTIME=/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda
FIXTURE=/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt
PY=/tmp/pr38-default-env/bin/python

# canonical hold + 300 extension
$PY -u "$BASE/candidates/rp1/hold.py" --repo "$RUNTIME" --config "$BASE/candidates/rp1/config.json" \
  --task mode_hold --backend cuda --initial-state "$FIXTURE" \
  --output "$BASE/runs/rp1-hold-replay" --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6 --post-window 300

# canonical shift
$PY -u "$BASE/candidates/rp1/shift.py" --repo "$RUNTIME" --config "$BASE/candidates/rp1/config.json" \
  --task mode_hold --backend cuda --initial-state "$FIXTURE" \
  --output "$BASE/runs/rp1-shift-replay" --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6

# frozen control
$PY -u "$BASE/candidates/rp1/shift_frozen.py" --repo "$RUNTIME" --config "$BASE/candidates/rp1/config.json" \
  --task mode_hold --backend cuda --initial-state "$FIXTURE" \
  --output "$BASE/runs/rp1-shift-frozen-replay" --network-floor 0.01 --prior-floor 0.05 --anneal-start 0.6

# seed-1234 native, observation adapter, one task at a time
$PY -u "$BASE/qualify/adapter/run_with_adapter.py" "$BASE/candidates/rp1/native100.py" \
  --repo "$RUNTIME" --candidate "$BASE/candidates/rp1" --task grid100 \
  --output "$BASE/runs/rp1-native-grid100-replay"
```

Same native command with `--task rotated100` and `--task staggered100`. Ledger: the attempt `tests.jsonl`. Logs: `$BASE/logs/`.
