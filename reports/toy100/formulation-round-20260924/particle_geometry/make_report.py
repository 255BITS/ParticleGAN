"""Preserve the measured leaderboard and explicitly withheld qualifications."""
from collections import Counter
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parents[2]
attempt = repo.parent
order = ['shared_coordinate', 'exposure_trust', 'exposure_mean', 'shared_geometry',
         'visit_isotropic', 'density_mobility']
later = ['trajectory', 'img_intensity2', 'img_bars4', 'img_blobs4', 'own_full_22', 'own_state_continuation']
ledger_path = attempt / 'tests.jsonl'
ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
existing = {(row['candidate'], row['gate']) for row in ledger}
rows = []
for slot, candidate in enumerate(order, 1):
    gates = {gate: json.loads((root / 'runs' / candidate / gate / 'result.json').read_text())
             for gate in ['mode_hold', 'vector_unequal_mass']}
    assert any(record['status'] != 'PASS' for record in gates.values())
    for gate in later:
        if (candidate, gate) not in existing:
            skipped = dict(candidate=candidate, gate=gate, status='SKIPPED', seconds=0,
                metrics=dict(qualification_status='NOT_RUN', reason='Both primary gates must pass; ring failed'),
                artifact=str(attempt / 'result.md'))
            ledger.append(skipped)
            with ledger_path.open('a') as stream:
                stream.write(json.dumps(skipped) + '\n')
    row = dict(candidate=candidate, proposal=slot,
               primary_passes=sum(r['status'] == 'PASS' for r in gates.values()),
               mean_shortfall=sum(r['verdict']['shortfall'] for r in gates.values()) / 2,
               mean_confirmation_fraction=sum(r['verdict']['confirmation_fraction'] for r in gates.values()) / 2,
               seconds=sum(r['seconds'] for r in gates.values()),
               gates={gate: dict(status=r['status'], metrics=r['result']['live'],
                     passing_suffix=r['verdict']['convergence']['passing_suffix'],
                     seconds=r['seconds'], artifact=str(root / 'runs' / candidate / gate / 'result.json'))
                     for gate, r in gates.items()},
               later={gate:'NOT_RUN' for gate in later},
               code=str(root / 'candidates' / candidate),
               declaration=str(root / 'candidates' / candidate / 'declaration.json'))
    rows.append(row)
rows.sort(key=lambda row: (-row['primary_passes'], row['mean_shortfall'], row['mean_confirmation_fraction']))
for rank, row in enumerate(rows, 1):
    row['rank'] = rank
totals = Counter(row['status'] for row in ledger)
primary = [row for row in ledger if row['candidate'] != 'regression' and row['status'] != 'SKIPPED']
regression = [row for row in ledger if row['candidate'] == 'regression']
summary = dict(qualified=False, stop_reason='Six distinct proposals exhausted', proposals=6,
               primary_totals=dict(Counter(row['status'] for row in primary)),
               regression_totals=dict(Counter(row['status'] for row in regression)),
               ledger_totals=dict(totals), full_22='NOT_RUN', stable_convergence='NOT_RUN',
               candidates=rows)
(root / 'leaderboard.json').write_text(json.dumps(summary, indent=2) + '\n')
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
source_files = [path for path in root.glob('*.py')] + [root/'replay.sh']
source_files += [path for code in (root/'candidates').iterdir() for path in code.iterdir()
                 if path.suffix in ('.py', '.json')]
source_hashes = {str(path.relative_to(root)):sha(path) for path in sorted(source_files)}
(root / 'source-hashes.json').write_text(json.dumps(source_hashes, indent=2) + '\n')
commands = [json.loads(line) for line in (root/'commands.jsonl').read_text().splitlines()]
audit = json.loads((root/'audit.json').read_text())
mechanism = json.loads((root/'mechanism-checks.json').read_text())
lines = [
    '**No formulation qualified. Six proposals exhausted; no candidate passed both primary gates.**',
    '',
    f'Work and artifacts: `{root}`. All writes stayed in this checkout except the explicitly requested '
    '`result.md` and `tests.jsonl` in its attempt directory. Prior checkouts were read-only. '
    'The first real candidate launch was ' + commands[0]['started_utc'] + ', within five minutes of the 23:34:26 UTC attempt start. '
    'No baseline/version control or seed experiment was repeated. One serial GPU worker, one CPU thread, '
    'FP32, deterministic algorithms, TF32 off; CPU initialization fixtures and native CUDA training/random draws.',
    '',
    'Started from `configs/toy100/constraints_simple_regularization.json` '
    '(SHA-256 `4af9863a319378b362bfb925b161d9ae8b8b07c9ecf1a452bb645570e04b99b7`). '
    'Architectures, data, fixed seeds, evaluation, thresholds, 1,200-step primary budgets, critic loss/regularizer, '
    'network Adam, original LR/noise schedules, and auxiliary AE/token host terms stayed unchanged. '
    'Mechanisms only use particle gradients, sampled indices, or latent positions. No target-derived training correction.',
    '',
    'The diagnostic ranking uses primary sustained pass count, then mean frozen shortfall, '
    'then mean confirmation/budget (failure = 2). It confers no full-suite eligibility. '
    'Ring requires 8 modes and HQ >= .90; unequal mass additionally requires eigen ratio >= .15 '
    'and minimum mass ratio >= .25, among its unchanged gates. Each sustained pass needs at least five passing suffix observations.',
    '',
    '| Rank | Proposal | Ring verdict; modes / HQ | Unequal verdict; eigen / min mass ratio | Passing suffix ring / unequal | Seconds, both | Own 22 / stability |',
    '|---:|---|---|---|---:|---:|---|',
]
for row in rows:
    ring, mass = row['gates']['mode_hold'], row['gates']['vector_unequal_mass']
    r, m = ring['metrics'], mass['metrics']
    lines.append(f"| {row['rank']} | `{row['candidate']}` | {ring['status']}; {r['modes']} / {r['hq']:.4f} | "
                 f"{mass['status']}; {m['component_min_eigen_ratio']:.4f} / {m['min_mass_ratio']:.4f} | "
                 f"{ring['passing_suffix']} / {mass['passing_suffix']} | {row['seconds']:.1f} | NOT_RUN / NOT_RUN |")
lines += [
    '',
    f'Executed primary gates: **12 = 5 PASS + 7 FAIL + 0 ERROR**, {sum(r["seconds"] for r in rows):.1f} seconds total measured gate runtime. '
    f'Regressions: **2/2 PASS**, including {mechanism["cases"]} mechanism cases. '
    '**36 SKIPPED** follow-up entries explicitly carry `NOT_RUN`; they are not executed tests. '
    'No warm probes or continuation gates ran. Every primary used 1,200 G/particle and 1,200 D updates: '
    '28,800 optimizer calls across the 12 gates, including 14,400 particle updates.',
    '',
    '`density_mobility` is an explicit sustained-failure example: its final unequal-mass metrics all meet their bounds '
    '(eigen .16980), but the passing suffix is only 3/5. It remains FAIL. '
    'Every ring curve had zero passing observations. Thus trajectory, intensity, bars, blobs, each candidate’s own '
    'full 22 gates, broader host regressions, and post-convergence continuation were NOT_RUN. '
    'The prior native 16/22 and CPU-init diagnostic 4/6 scores are context only; no passes were combined with them.',
    '',
    'Measured adaptations, in execution order:',
    '',
    '1. `shared_coordinate`: share the particle second moment across rows while preserving row gradients. '
    'Unequal mass recovered (11-check suffix), but ring lost another mode. O(Nd) pooling; no extra model passes.',
    '2. `exposure_trust`: ordinary Adam followed by a displacement bound divided by EMA sampled-row exposure. '
    'Rare mobility recovered compared with the prior fixed cap; ring still only seven. '
    'Adds an index count, table copy and row reductions, with no extra model passes.',
    '3. `exposure_mean`: add multiplicity averaging to exposure trust. '
    'Best unequal-mass covariance error (.09803) and eigen ratio (.82172), but ring fell to six. '
    'Adds a sampled-gradient rescaling, with no extra backward evaluation.',
    '4. `shared_geometry`: full shared gradient covariance, trace-preserving isotropic shrinkage, inverse square root. '
    'Rare mass passed with exactly five final checks; ring remained six. '
    'Adds O(Nd²) Gram/product work and O(d³) eigendecomposition each update.',
    '5. `visit_isotropic`: per-row scalar moment, visit-based bias correction, multiplicity averaging, '
    'and inverse-sqrt exposure scaling. Unequal mass passed; ring regressed to five and HQ .8333. '
    'Adds O(Nd) reductions and per-row CUDA moment clocks.',
    '6. `density_mobility`: inverse latent-neighborhood-density scaling of multiplicity-averaged Adam displacement. '
    'Ring reached five; rare shape fluctuated and failed the sustained suffix. '
    'Adds O(A²d) pairwise work on active rows; no extra sampled batch or model pass.',
    '',
    'All results are **equal-step** comparisons. Added reductions, matrix operations and displacement transformations '
    'are declared in each candidate’s immutable `declaration.json`; no additional optimizer step, forward evaluation '
    'or backward evaluation was used. Exact equal-compute comparisons were NOT_RUN. '
    'The reported seconds include probe, random-stream auditing and evaluation overhead, not isolated kernel timings.',
    '',
    f'`audit.json` regraded all 12 complete 24-observation curves, verified {audit["prepared_files"]} prepared source files, '
    'candidate/worker hashes, 2×1,200 optimizer calls, CUDA particle/moment receipts, and exact equality of '
    'initial parameter hashes, the complete ordered random-draw receipt, configuration and action/schedule traces '
    'against each retained CPU-init/native-CUDA control. `mechanism-checks.json` verifies network Adam identity, '
    'unchanged RNG, unused-row immobility, finite CUDA updates and counts for all six mechanisms, plus '
    'rotation equivariance for the full-covariance and isotropic-visit rules. These are mechanism checks, not 22-toy passes.',
    '',
    'Recommendation: do not promote any candidate. Keep `exposure_trust` as the strongest partial particle-only '
    'reference, and `shared_coordinate` / `exposure_mean` as useful rare-mass evidence. '
    'The next untested hypothesis is preconditioning with generator-output sensitivity rather than latent proximity: '
    'the latent-density rule provided no ring recovery and destabilized rare shape. Any such next round should '
    'declare its Jacobian/VJP work, retain these fixed seeds/budgets, and again require both blockers. '
    'No seventh proposal was launched in this attempt.',
    '',
    'Exact artifacts and replay:',
    '',
    f'- Attempt report: `{attempt / "result.md"}`; append-only gate ledger: `{ledger_path}`.',
    f'- Executed argument lists, shell commands, prelaunch declaration/fixture hashes and launch times: `{root / "commands.jsonl"}`.',
    f'- Each exact code snapshot: `{root}/candidates/<candidate>/{{probe.py,particle_update.py,declaration.json}}`; aggregate hashes: `{root / "source-hashes.json"}`.',
    f'- Each raw verdict, all observations, schedule and CUDA/RNG receipts: `{root}/runs/<candidate>/<gate>/result.json`; logs: corresponding `<gate>.log`.',
    f'- Prepared once with the existing `prepare.py`: `{root / "prepared"}`; reused initialization fixtures: `{repo / "reports/toy100/cpu-recipe-gpu-port/initialization-fixtures"}`.',
    f'- Only proposal six additionally retained its own final model/optimizer/RNG states: `{root}/runs/density_mobility/{{mode_hold,vector_unequal_mass}}/own-final-state.pt`. These are failed acquisition states, not converged qualification checkpoints.',
    '',
    'The executed setup command was:',
    '',
    '```bash',
    '/tmp/pr38-default-env/bin/python reports/toy100/cpu-recipe-gpu-port/prepare.py --root reports/toy100/particle-geometry/prepared',
    '```',
    '',
    'Each candidate was launched with the following command (in the six-proposal order above); '
    '`run_batch.py` reads the supervisor, verifies hashes, and runs both gates serially:',
    '',
    '```bash',
    '/tmp/pr38-default-env/bin/python -u reports/toy100/particle-geometry/run_batch.py --candidate exposure_trust > reports/toy100/particle-geometry/exposure_trust.batch.log 2>&1',
    '```',
    '',
    'Preserve the existing evidence when replaying. These commands use a new output path and the retained fixtures; '
    'replace the candidate/gate and choose another new output path for other entries:',
    '',
    '```bash',
    f'cd {repo}',
    'bash reports/toy100/particle-geometry/replay.sh exposure_trust mode_hold reports/toy100/particle-geometry/replays/exposure_trust-mode_hold-new',
    'bash reports/toy100/particle-geometry/replay.sh exposure_trust vector_unequal_mass reports/toy100/particle-geometry/replays/exposure_trust-vector_unequal_mass-new',
    'tail -F reports/toy100/particle-geometry/exposure_trust.batch.log',
    '```',
    '',
    'The regression commands were:',
    '',
    '```bash',
    'env CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/pr38-default-env/bin/python reports/toy100/particle-geometry/check_mechanisms.py',
    'env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/pr38-default-env/bin/python reports/toy100/particle-geometry/audit_results.py',
    '```',
]
report = '\n'.join(lines) + '\n'
(attempt / 'result.md').write_text(report)
(root / 'README.md').write_text(report)
print(json.dumps(dict(report=str(attempt/'result.md'), proposals=6, totals=dict(totals), qualified=False)))
