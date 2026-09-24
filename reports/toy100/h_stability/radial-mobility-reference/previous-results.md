No promotion. Stopped this geometry family after **7/12 proposals**; all seven are rejected. The selected base remains `g_threequarter_rate`. First batch launched at 18:10:06 UTC, within five minutes. No seed experiments, rate interpolation, nested agents or publication.

Gate-progress leaderboard. `s` is the terminal passing suffix; the unchanged gates require at least 5 passing observations. A passing endpoint alone earns no PASS.

| Candidate | two_pole80: travel / suffix / status | ring1200: modes / HQ / suffix / status | Later result |
|---|---|---|---|
| [g_radial_split_adam](geometry/batch02/g_radial_split_adam/status.json) | 0.956653 / s20 / PASS | 8 / 0.999756 / s9 / PASS | unipolar FAIL; own200 diagnostic FAIL at 1211 |
| [g_radial_network_adam](geometry/batch04/g_radial_network_adam/status.json) | 0.956653 / s20 / PASS | 8 / 1.000000 / s3 / FAIL | SKIPPED after failure |
| [g_radial_row_rms](geometry/batch03/g_radial_row_rms/status.json) | 0.956653 / s20 / PASS | 8 / 0.999756 / s3 / FAIL | SKIPPED after failure |
| [g_geometry_contrast_adam](geometry/batch01/g_geometry_contrast_adam/status.json) | 0.832373 / s5 / PASS | 8 / 0.997803 / s1 / FAIL | SKIPPED after failure |
| [g_geometry_split_adam](geometry/batch01/g_geometry_split_adam/status.json) | 0.954565 / s21 / PASS | 8 / 0.987549 / s1 / FAIL | SKIPPED after failure |
| [g_radial_tensor_rms](geometry/batch03/g_radial_tensor_rms/status.json) | 0.956653 / s20 / PASS | 7 / 0.991455 / s0 / FAIL | SKIPPED after failure |
| [g_radial_contrast_adam](geometry/batch02/g_radial_contrast_adam/status.json) | 0.677850 / s4 / FAIL | SKIPPED | SKIPPED after failure |

Best measured mobility result: `g_radial_split_adam` passes two_pole (travel .956653118, critic median gradient .006417856, suffix20) and cold ring (8 modes, HQ .999755859, suffix9). It fails unipolar neutral retention .808412477 < .85; cover .869257144 and off-caption .000013524 pass. Its **own** cold checkpoint passes 10 dense continuation checks and fails at 1211 with 8 modes/HQ .805175781. That 11-update run is only a blocker diagnostic, not an own1200 hold.

Inherited reference, not rerun or credited: g_threequarter_rate fails two_pole at .028546154, passes cold ring with suffix9, and passes 83 own-state dense checks before failing at 1284/HQ .761230469. Its borrowed-H 200/200 result is excluded. The new mobility candidate is worse on measured own-state stability. H and the fast cold-repair comparator remain historical evidence only.

Policy and effective rates: every proposal starts from the selected declaration and keeps nominal G/D/prior rates **.001125/.0015/.00225**, Adam(0,.999), epsilon1e-8 and the same frozen hosts. Particle mean gain is .1 (coefficient .000225); amplified shape/radial gain is8 (coefficient .018). Isotropic variants apply gain8 to every zero-mean particle displacement. Radial variants retain tangential gain1 (.00225), applying `I + 7 uuᵀ/(uᵀu + 1e-12)` to the centered particle displacement, where u is the current centered cloud. Translation remains positive. This is a fixed parameter-geometry rule, with no iteration input, target radius or support bound.

`contrast_adam` transforms the ordinary Adam displacement. `split_adam` separately normalizes the mean and zero-mean adversarial gradients, projects the normalized shape direction back to zero mean, and serializes its two additional second moments. Batch03 adds either one RMS denominator per G tensor or per matrix row; G vectors use one tensor denominator. Batch04 retains ordinary G Adam and amplifies only each current parameter tensor’s radial displacement, with fixed G radial coefficient .009 and tangential coefficient .001125. All rules apply by parameter role/shape across hosts, never by host name.

New adversarial training signal: **none**. G and particles use the inherited logistic relativistic discriminator loss exclusively. D retains R1+R2 .6, mixup .01 and fixed input noise .05. Geometry uses only parameters, adversarial gradients and optimizer moments. There are no additional training forwards/backwards or target-fitting terms. All preconditioner gains and denominator floors are strictly positive. The raw receipts retain actual rates and movement; these remain experimental scratch policies, not production-qualified defaults.

Executed: **14 cold gates (7 PASS, 7 FAIL)** plus **one own-state diagnostic FAIL**; no ERROR. First-run work is 8,171 outer updates. Every unexecuted gate is explicitly SKIPPED in the runtime ledger. The best candidate has older19 = 2 PASS / 1 FAIL / 16 SKIPPED; original10 = 1 PASS / 9 SKIPPED. No proposal reaches mid_scale_identity, cover_leftover or the other nine original gates. Full own1200 and native grid100/rotated100/staggered100 are SKIPPED for all seven. The documented AE and unused-token structural blockers were not retrained.

Unit tests: **77 passing case executions across three pytest runs: 41 + 15 + 21; 0 failures, 0 errors, 0 skips**. Existing optimizer/noise regressions ran after code changes. New tests cover state continuation, network/critic isolation, positive subspace movement and rotational covariance of shared RMS metrics. Logs/XML: [regression01](geometry/regression01.log), [regression02](geometry/regression02.log), [regression03](geometry/regression03.log). The full suite is explicitly SKIPPED because there is no viable final candidate.

One archived-source cold replay adds 1,280 regression updates. All scored observations match exactly; the ring checkpoint is byte-identical (SHA256 `64a97c6b154ae7a44bd6d43bb95939750c411d841a5bfd7a9558421f14f280c2`). Two_pole has 36 identical training tensor leaves; its checkpoint bytes differ only in unused process-start Python/NumPy RNG snapshots. [Replay comparison](geometry/replay-comparison.json). Replays, comparison and unit tests are labelled **candidate=regression**, never new toy passes. No immutable H audit was repeated.

Code: [particle_geometry.py](particle_geometry.py), [geometry_runner.py](geometry_runner.py), [geometry_probe.py](geometry_probe.py), [tests](../../../tests/test_particle_geometry.py). Exact declarations: [batch01](geometry/batch01.json), [batch02](geometry/batch02.json), [batch03](geometry/batch03.json), [batch04](geometry/batch04.json). Each batch contains its own `manifest.json` and `source.tar.gz`; every executed gate retains raw episode metrics, actual optimizer/geometry receipt and checkpoint. [Machine-readable summary](geometry/attempt-summary.json).

Recommendation: retain g_threequarter_rate as the experimental base. Keep radial split Adam only as a rejected mobility comparator. Further gain interpolation is not justified: broader motion loses ring, while the surviving radial policy loses neutral retention and dense stability. A future intervention must address those two blockers together.

Exact replay from this repository root; the launcher reads the supervisor, pins CPU/AVX2 and loads each archived implementation. Use fresh output/source directories and a separate ledger. These commands replay all seven proposals and the one diagnostic:

```bash
cd /ml2/hypergan/gan-attempts/g075-20260924T180736Z/cold_mobility/20260924T180736Z-1586956/repo
for batch in batch01 batch02 batch03 batch04 own200; do
  bash reports/toy100/h_stability/geometry/replay.sh "$batch" \
    "reports/toy100/h_stability/geometry/new-$batch" \
    reports/toy100/h_stability/geometry/new-replay.jsonl \
    > "reports/toy100/h_stability/geometry/new-$batch.log" 2>&1
done
```

Exact unit-test commands (same pinned environment as replay):

```bash
/tmp/pr38-default-env/bin/python -m pytest -q tests/test_particle_geometry.py tests/test_continuous_candidates.py tests/test_legacy_learnable_noise.py tests/test_legacy_noise_remaining.py
# Original later invocations used only the first two files (15 then 21 cases as tests evolved).
bash -n reports/toy100/h_stability/geometry/replay.sh
```

Concise log tail:

```bash
tail -F reports/toy100/h_stability/geometry/batch0{1,2,3,4}.log
```

Runtime files: [result.md](/ml2/hypergan/gan-attempts/g075-20260924T180736Z/cold_mobility/20260924T180736Z-1586956/result.md), [tests.jsonl](/ml2/hypergan/gan-attempts/g075-20260924T180736Z/cold_mobility/20260924T180736Z-1586956/tests.jsonl).
