# Continuous K3P search: round 1

Eight attempts (3 Codex, 5 Grok) tested 24 mechanisms. **No qualified winner.**
Keep K3P selected: its measured 22/22 gates, hold and extension are unchanged.
Full toy qualification was not opened for any modified candidate because none
passed its own hold, extension and canonical shift protocol.

| Candidate | Hold | Extension | Recovery deadline | Horizon status |
|---|---|---|---|---|
| K3P, selected parent | 1200/1200 | 300/300 | FAIL 28/81 | Scheduled |
| **A3 reversal strength** | **1200/1200** | **300/300** | **FAIL 71/81** | 1000-update full-state prefix PASS |
| RCM3 applied-step mixing | 1200/1200 | 300/300 | FAIL 0/81 | Inherited schedules remain |
| INNOV3 anchor memory | 1200/1200 | 300/300 | FAIL 0/81 | Inherited schedules remain |
| RUC3 quiet-then-leak | 1200/1200 | 300/300 | FAIL 0/81 | Noise schedule remains |
| SN3 asymmetric rates | FAIL after 156 good checks | NOT_RUN | FAIL 80/81 | Formula audit only |

A3 is the best stable lead. It uses stationary rates (G/D .000425, prior .00085),
zero input noise, constant output noise .029, capped real/fake critic gradients,
and an EMA-gradient anchor whose strength is `1 + relu(-cos(g_t,g_previous))`.
It retains the .999 critic EMA and existing K3P particle mechanisms. No training
horizon enters its four schedule/noise primitives. Its hold minimum HQ is .93042
and extension minimum .98560. Recovery retains all eight modes after the deadline,
but ten precision checks miss HQ .90; worst HQ .81299, sustained delay 810 updates.
Its shift pre-hold is also only 89/120 because acquisition is slower. All of these
failures must be fixed; 71/81 is not a pass. All 22 toys remain NOT_RUN for A3.

SN3's 80/81 is a recovery-only lead with failed hold and pre-shift stability.
It cannot outrank a stable formulation merely on the recovery fraction. Its one
miss loses a mode; continued hold is 57/120. Constant full-rate reopening and
coherence-only controllers often damage acquisition or keep rates too high.

A3 is an unverified candidate, not the next shared search base. The selected
K3P remains the base until a candidate passes the failing problem and all frozen
gates. A dedicated qualification lane can test A3 before any further reuse;
independent searches continue from K3P. No unchanged baseline reruns or coefficient grids. Winners must pass the full
unchanged shift verdict, hold/extension, matched frozen recovery, all 22 toys,
and separately declared delayed/repeated changes before a continuous-learning claim.

## Evidence

[Machine-readable ledger and raw snapshot hashes](evidence.json),
[exact A3 source bundle](sources/a3_reversal_strength/),
[source hashes](source-hashes.json). Each `attempts/LANE/` retains the report,
final response and complete ledger. Compressed raw JSON snapshots are under
`evidence/`; hashes refer to decompressed bytes. Interrupted settling runs are
reported as partial failures, never terminal benchmark passes.

Canonical attempt ledger: **4 PASS / 35 FAIL / 2 ERROR**. The four passes are
hold+extension only. This includes invalid launches and partial terminated runs;
it is not a count of qualified formulations. Full ledger including diagnostics:
25 PASS / 38 FAIL / 6 ERROR / 42 SKIPPED. Three short prefix passes cannot establish
whole-formulation independence; the stronger A3 prefix compared full state after
1000 updates with different declared horizons. Raw evidence, not monitor totals,
settles qualification. No seed sweeps were used.

| Lane | Candidate | Executed canonical outcomes |
|---|---|---|
| continuous_critic | c1_static_anchor | hold: FAIL; shift: FAIL |
| continuous_critic | c2_innovation_damping | hold: FAIL; shift: FAIL |
| continuous_critic | c3_stationary_innovation | hold: FAIL; shift: FAIL |
| reversible_plasticity | c01_coherence | hold_setup_error: ERROR; hold: FAIL; shift: FAIL |
| reversible_plasticity | c02_constant_stationary | hold: FAIL; shift: FAIL |
| reversible_plasticity | c03_energy_ratio | hold: FAIL; shift: FAIL |
| adaptive_anchor | a1_stationary_anchor | hold: FAIL; shift: FAIL |
| adaptive_anchor | a2_coherent_memory | hold: FAIL; shift: FAIL |
| adaptive_anchor | a3_reversal_strength | hold: PASS; shift: FAIL |
| relative_update_control | ruc1_energy_moment | mode_hold_shift_recovery: FAIL |
| relative_update_control | ruc2_moment_leak | mode_hold_shift_recovery: FAIL |
| relative_update_control | ruc3_quiet_then_leak | mode_hold_shift_recovery: FAIL; mode_hold_extension: PASS |
| reversible_critic_mixing | rcm1 | hold_extension: FAIL |
| reversible_critic_mixing | rcm2 | hold_extension_invalid_launch: ERROR; hold_extension: FAIL |
| reversible_critic_mixing | rcm3 | hold_extension: PASS; target_shift_recovery: FAIL |
| anchor_innovation | innov1 | ring_hold_extension: FAIL |
| anchor_innovation | innov2 | ring_hold_extension: FAIL |
| anchor_innovation | innov3 | ring_hold_extension: PASS; target_shift_recovery: FAIL |
| particle_network_balance | pnb1 | ring_shift_recovery: FAIL |
| particle_network_balance | pnb2 | ring_shift_recovery: FAIL; ring_hold_extension: FAIL |
| particle_network_balance | pnb3 | ring_shift_recovery: FAIL |
| stationary_noise | sn1 | ring_hold_extension: FAIL; target_shift_recovery: FAIL |
| stationary_noise | sn2 | ring_hold_extension: FAIL |
| stationary_noise | sn3 | ring_hold_extension: FAIL; target_shift_recovery: FAIL |
