# A2-off native mechanism diagnostic

Prepared, registered and source-frozen; **not enqueued or trained**. No candidate
readout, lifecycle conclusion, promotion or adoption is recorded by this study.

The diagnostic asks whether removing A2 sparse-row damping changes the late
covariance undershoot on the current affine grid100 learned-MoG host. Saved
prior art does not predict a positive outcome or establish A2 as its cause.

| Item | Frozen declaration |
|---|---|
| Candidate | [`k3p-a2-off-native-diagnostic`](../../../configs/forge/ideas/k3p-a2-off-native-diagnostic.json) |
| Only recipe change | `latent_damping_max_rate: 0.5 → 0`; remove the required `a2` capability because it is intentionally disabled |
| Candidate revision | `8493b15ff2d0f677de94a07cf344762a1d9efd58c609987fff7dab311cad8581` |
| Scientific source | `5c9c929877c141ccf7352c16987d3a5aadf1aff3a0fedbfa78e7d9b8fe06fdb7` |
| Selected task | `grid100_affine_square_named_v1` only |
| Protocol | Seed 0; 7,000 updates; named identity-affine G, Fourier3/Xavier D and uniform `[-5,5]` prior locations |
| Prior and scoring | Learned MoG locations, fixed latent sigma `.025`, uniform masses, no standardization; clean live sampling |
| Gates | Unchanged five terminal checks at 6,000/6,250/6,500/6,750/7,000 and independent 100,000-sample holdout |
| New cost ceiling | Task/candidate/campaign each 3,600 seconds |
| Cost context | Original paired grid took 92.037 seconds; roughly 90 seconds is an estimate, not a guarantee |

The original K3P control is
[`dca8c7aa0eca484c9270d07125f7cb0b`](../attempts/dca8c7aa0eca484c9270d07125f7cb0b/result.json).
It failed the terminal shape gates. The existing `diagnostic_imports` interface
binds its original registration, exact candidate revision, complete scientific
cohort, diagnostic/qualification keys and request/result/evidence hashes. Its
92.037 seconds remain reported cost; no replacement receipt or control rerun is
created. The import remains diagnostic evidence with `qualification_reuse=false`.

The separate [profile](../../../configs/forge/calibration/host-profile-a2-off-v1.json)
contains K3P and the A2-off candidate. It preserves the predecessor's **three
smoke tasks, 16 independent reference tasks, reference exclusions, criteria and
resource limits**. This two-lineage study cannot satisfy the unchanged minimum
of three paired lineages, including one full reference positive and two
negatives. Unmeasured cells remain unknown. Only the new A2-off native cell is
selected by the [bounded contract](../../../configs/forge/campaigns/host-profile-a2-off-native-v1.json);
no other task is authorized by this registration.

Prior-art rationale:

- The [historical A2 comparison](../../toy100/continuous-practical-leaderboard.md#earlier-round-leader-not-promoted-a2_bounded_damp)
  changed only `latent.py` and improved unequal-width covariance `.9813 → .0590`.
  Its native no-op comparisons had mixed outcomes; the earlier attribution of
  collapse to A2 was explicitly withdrawn. These cloud-prior results motivate
  isolating the existing mechanism, without forecasting improvement on MoG.
- The [MoG width study](../../transfer_suite/solvability/mog/README.md) found no
  full pass in 16 fixed-seed trials. It used calibrated relative width, unlike
  this host's fixed absolute latent sigma. No width is selected or tuned here.
- Historical K3P [22/22 evidence](../../toy100/k3p-base/README.md) belongs to its
  original particle/cloud protocol and does not supply a current positive.

Validation used the existing Forge interfaces: resolve both formulations;
compare every task, protocol, RNG declaration and cohort; verify that only the
one resolved recipe field differs; register and freeze first in a disposable
mirror, then in this worktree; verify the frozen source and registered request;
and check the certified control import without conflicts. All candidate and
task preflights passed. Original K3P declarations, criteria and receipt bytes
were checked unchanged. [Machine-readable validation and plan](a2-off-native-v1-plan.json)
and the [immutable registration](../calibration-lanes/host-profile-a2-off-native-v1/registration.json)
record the exact identities.

Read-only preview from the feature worktree:

```bash
python -m experiments.forge calibration-lane plan host-profile-a2-off-native-v1
```

After any separately authorized execution, compare the full terminal live and
holdout verdicts, covariance bias, radial KS, centre accuracy, coverage and paid
cost against the control. Preserve both outcomes. A passing diagnostic would
justify considering further qualification; it would not establish the full
reference positive required for calibration adoption.
