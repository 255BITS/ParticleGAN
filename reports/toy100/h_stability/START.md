# Current research base: eps_net_1m

Selection: [current-base.json](current-base.json). Replay declaration:
[eps-net-base/declaration.json](eps-net-base/declaration.json).
This is a measured search starting point, not a release-qualified winner.

This publication is based on current `develop`. Reproduce the measured recipe
in an isolated directory: the archived training package differs from current
production source, and the source verifier deliberately rejects that mismatch.

```bash
BENCH_PYTHON=/path/to/python bash reports/toy100/h_stability/replay-selected.sh /tmp/eps-replay-new
# Optional modes after NEW_DIRECTORY: cold, long, or prepare. Default: short.
```

Use Python3.12 with the experiment dependencies (recorded PyTorch2.13.0).
The script checks the archives, restores all125 frozen training sources,
pins CPU/AVX2 and one thread, then runs the200-check own-state replay.
The lower-level commands below are for the prepared directory or the original
research checkout. Production package defaults are unchanged.

It is still a GAN. The generator and trainable latent particles learn only
through the discriminator's logistic relativistic adversarial loss. The
critic retains R1+R2 .6, mixup consistency .01 and fixed input noise .05.
The change is fixed Adam epsilon .001 for G and D; particle epsilon stays1e-8.
Actual G/D/particle rates stay .001125/.0015/.00225 with Adam(0,.999), starting
at cold initialization. No additional generator-fitting objective is present.

| Case | Selected result |
| --- | --- |
| Cold ring1200 | PASS:8 modes, HQ.996826172, terminal suffix14 |
| Own-state200 | PASS:200/200 dense checks, minimum HQ.918945313 |
| two_pole80 | FAIL:spread.029888831 <.30 |
| Other17 older hosts | Audit complete:9 PASS /8 FAIL; complete19 total10 PASS /9 FAIL |
| Own-state1200 / native100 | FAIL at1692/HQ.864746094 after491 passing checks / SKIPPED |

The authorized one-time audit is complete. See [RESULTS.md](RESULTS.md) and
[audit evidence](baseline-qualification/audit/audit-results.json). The long hold
stopped at its first failed dense check after492 of1200 requested updates;
the first200 reproduce minimum HQ.918945313. Cold ring's retained terminal HQ
is .996826172; .999755859 is the own200 endpoint. This remains an unqualified
experimental base. Do not repeat the completed baseline audit.

Own-state200 uses the candidate's OWN cold-acquired models, Adam moments, RNG
and the same epsilon/rates. It is not borrowed-H evidence. Old g_threequarter_rate
failed its own short hold after83 passing checks; H failed after54. Those older
candidates remain in selected-base/ and ../critic_signal_attempt/ respectively.
Do not borrow their passing toy results or use their ordinary-epsilon runners.

```bash
# Run with pinned CPU/AVX2 and one thread, as supplied by the launcher.
python reports/toy100/h_stability/selected_base_probe.py --output NEW_OUTPUT
python reports/toy100/h_stability/adam_response_cold.py --declaration reports/toy100/h_stability/eps-net-base/declaration.json --output NEW_OUTPUT --ledger NEW_LEDGER --workers 1 --tasks two_pole mode_hold unipolar mid_scale_identity cover_leftover trajectory
```

Executable policy is adam_response.py::response_policy. adam_response_cold.py
installs it for all frozen cold hosts; selected_base_probe.py selects it for
own-state diagnostics. The generic selected_h_remaining.py lacks this epsilon
policy and must not be used to claim a selected-base replay.

Best separate mobility comparator: g_radial_split_adam passes two_pole and cold
ring, but fails unipolar neutral retention .8084<.85 and own continuation at1211.
Its source and declaration are in radial-mobility-reference/, with executable
particle_geometry.py, geometry_runner.py and geometry_probe.py. The current combination lane has tested it with eps_net_1m; the first
combined proposal passes two_pole and ring but fails unipolar retention. Combining independent optimizer wrappers
requires checking the actual applied epsilon, roles and state, not nesting
observers blindly. Every candidate needs its own gates and checkpoint.

Finished failures: seven particle geometry proposals, nine Adam-response
proposals, and relativistic-mean/product losses. Do not repeat the same rows.
The geometry and response reports retain exact failures; broader H-era sweeps
are historical. Pure AE has no encoder adversarial path; unused-token hold has
a documented shared-gradient conflict. Neither is a parameter-sweep target.

The assigned baseline audit ran remaining older hosts and a1200-step own
continuation as diagnostics despite two_pole failure. It stopped the long hold
at its first failed check. This completed exception never promotes a failed
recipe or enables expensive native100 runs.
