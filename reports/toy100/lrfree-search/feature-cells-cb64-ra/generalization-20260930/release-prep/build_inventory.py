"""Prepare a small-artifact archive plan; reads JSON/source only, writes here only."""
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-generalization-20260930')
HERE = ROOT / 'release-prep'
DESTINATION = 'reports/toy100/lrfree-search/feature-cells-cb64-ra/generalization-20260930/'
TEXT_SUFFIXES = {'.json', '.jsonl', '.py', '.md', '.patch', '.log'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def write(name, value):
    target = HERE / name
    with target.open('x') as output:
        output.write(value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    HERE.mkdir(exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    board_raw = (ROOT / 'validation-ra14/scoreboard-all.json').read_bytes()
    board = json.loads(board_raw)
    write('QUEUE-SNAPSHOT.json', board_raw.decode())
    tree = ast.parse((ROOT / 'validation-ra14/run_all.py').read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('PORTS', 'NATIVE'):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
    plan = [('screen', task) for task in constants['PORTS']]
    plan += [('moving', task) for task in constants['NATIVE']]
    plan += [('screen', task) for task in constants['NATIVE']]
    assert len(plan) == 19
    reported = {(item['kind'], item['task']): item for item in board['results']}
    assert set(reported) <= set(plan)
    gates = []
    for kind, task in plan:
        relative = ('validation-ra14/runs/' + task + '/acceptance-receipt.json'
                    if kind == 'screen' else 'validation-ra14/moving/' + task + '/COMPLETION.json')
        gate = dict(kind=kind, task=task, completion='PENDING', quality_status='PENDING',
                    evidence_validity='PENDING', receipt=relative)
        if (kind, task) in reported:
            receipt = read(relative)
            quality = receipt.get('acceptance_status', receipt.get('quality_status'))
            assert quality == reported[kind, task]['status']
            validity = receipt.get('evidence_validity',
                                   (receipt.get('source_integrity_after') or {}).get('status'))
            gate.update(completion='CLOSED_AS_REPORTED', quality_status=quality,
                        evidence_validity=validity)
        gates.append(gate)

    ra13 = read('mnist/ra13-settled/CLOSED.json')
    replay = read('mnist/ra14-replay-r2/CLOSED.json')
    replay_summary = read('mnist/ra14-replay-r2/summary.json')
    completion = read('mnist/ra14-replay-r2/COMPLETION-replay.json')
    assert digest((ROOT / 'mnist/ra14-replay-r2/CLOSED.json').read_bytes()) == '07ac938ad49588b320d2bc83943bf691d85aa97db55d937c081267a8c9df3d97'
    assert replay['replay_status'] == {'toy': 'PASS', 'mnist': 'PASS'}
    assert completion['status'] == 'COMPLETE' and completion['returncode'] == 0
    assert digest(Path(completion['result']).read_bytes()) == completion['result_sha256']
    assert replay['original_quality_receipt_sha256'] == digest((ROOT / 'mnist/ra13-settled/CLOSED.json').read_bytes())
    assert ra13['mnist_all_checkpoint_metrics_equal_corrected_E22'] is True
    assert ra13['mnist_all_checkpoint_lrs_equal_corrected_E22'] is True
    assert ra13['toy_all_postupdate_metrics_and_lrs_equal_original_RA11'] is True
    assert ra13['replay_status'] == {'toy': 'FAIL', 'mnist': 'FAIL'}

    learned = []
    for variant, folder in [('RA12-auto', 'mnist/ra12-auto'), ('RA13-settled', 'mnist/ra13-settled')]:
        for task in ['toy', 'mnist']:
            relative = f'{folder}/training/{task}/{variant}/result.json'
            result = read(relative)
            final = result['final']
            diag = final['diagnostics']
            learned.append(dict(variant=variant, task=task, status=result['status'],
                                steps=result['steps'], original_quality_gate=result['original_quality_gate'],
                                metrics=final['metrics'], output_sigma=diag['output_sigma'],
                                surprise=diag['surprise'], result=relative,
                                result_sha256=digest((ROOT / relative).read_bytes())))

    source_ra13 = read('RA13-SOURCE-CLOSURE.json')
    source_ra14 = read('RA14-SOURCE-CLOSURE.json')
    full_old = read('validation-ra13-r2/FULL-TESTS.json')
    assert full_old['status'] == 'PASS' and full_old['returncode'] == 0
    assert '1404 passed, 12 skipped, 18 subtests passed' in (ROOT / 'validation-ra13-r2/full-pytest.log').read_text()
    full_new_path = ROOT / 'validation-ra14/FULL-TESTS.json'
    full_new_snapshot = None if not full_new_path.exists() else json.loads(full_new_path.read_text())
    # This preparation is intentionally not a final suite attestation.
    suite = dict(status='PENDING', required_gate_count=19, completed_in_snapshot=len(reported),
                 queue_source_status=board['status'], queue_source_current=board.get('current'),
                 queue_snapshot_sha256=digest(board_raw), gates=gates,
                 full_suite=dict(status='PENDING', receipt='validation-ra14/FULL-TESTS.json',
                                 passed=None, skipped=None, subtests=None,
                                 observed_record=full_new_snapshot))

    selected = {}
    def add(path, category, scope='CLOSED_SMALL_ARTIFACT'):
        path = Path(path)
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            assert '__pycache__' not in path.parts
            relative = str(path.relative_to(ROOT))
            raw = path.read_bytes()
            selected[relative] = dict(local_path=str(path), archive_relative_path=relative,
                                      category=category, scope=scope, bytes=len(raw), sha256=digest(raw))

    for name in ['RA12-SOURCE-CLOSURE.json', 'RA13-SOURCE-CLOSURE.json', 'RA13-SOURCE-REVIEW.json',
                 'RA14-SOURCE-CLOSURE.json', 'RA14-SOURCE-REVIEW.json', 'ROOT-REVIEW.json',
                 'PR155-BASELINE-TESTS.json', 'pr155-baseline-pytest.log',
                 'R1-PREPARATION.json', 'R1-CPU-PREFLIGHT.json', 'build_r1.py', 'test_r1_composition.py']:
        add(ROOT / name, 'source_and_baseline_provenance')
    for name in ['RA12-auto.json', 'RA13-settled.json', 'RA14-replay.json']:
        add(ROOT / 'configs' / name, 'candidate_configuration')
    for folder in ['pkg-RA12-auto', 'pkg-RA12-cpu-fix', 'pkg-RA13-settled', 'pkg-RA14-replay']:
        for path in sorted((ROOT / folder).rglob('*.py')):
            if '__pycache__' not in path.parts:
                add(path, 'candidate_source')
    folders = {
        'portability/cpu-fix': 'cpu_optimizer_repair',
        'portability/settled-guard': 'settled_guard_contracts',
        'portability/replay-alias': 'restoration_alias_repair',
        'mnist/ra12-auto': 'retained_ra12_false_fire_and_quality',
        'mnist/ra13-settled': 'original_ra13_training_and_failed_replay',
        'mnist/ra13-replay-diagnosis': 'retained_ra13_replay_failure',
        'mnist/ra14-replay': 'retained_zero_update_preparation_failure',
        'mnist/ra14-replay-r2': 'qualified_ra14_restoration_replay',
        'mnist/r1-observe': 'closed_static_observation',
        'mnist/r1-moving-observe': 'closed_moving_observation',
        'mnist/r1-observation-supplement': 'observation_erratum_and_supplement',
        'diagnostics/optimizer-surprise-dynamics': 'independent_source_scalar_and_lane_reviews',
    }
    for folder, category in folders.items():
        for path in sorted((ROOT / folder).rglob('*')):
            if '__pycache__' not in path.parts:
                add(path, category)
    for folder in ['validation-ra13', 'validation-ra13-r2', 'validation-ra14']:
        for path in sorted((ROOT / folder).iterdir()):
            if path.name not in ('scoreboard-all.json', 'FULL-TESTS.json', 'full-pytest.log'):
                add(path, 'frozen_validation_adapter')
    for name in ['FULL-TESTS.json', 'full-pytest.log']:
        add(ROOT / 'validation-ra13-r2' / name, 'ra13_historical_full_suite')
    for gate in gates:
        if gate['completion'] == 'CLOSED_AS_REPORTED':
            folder = (ROOT / gate['receipt']).parent
            for path in sorted(folder.iterdir()):
                add(path, 'ra14_completed_gate_snapshot', 'CLOSED_IN_QUEUE_SNAPSHOT_NOT_SUITE_COMPLETE')
            for prefix in ['screen', 'collect'] if gate['kind'] == 'screen' else ['moving']:
                add(ROOT / 'validation-ra14/logs' / f'{prefix}-{gate["task"]}.log',
                    'ra14_completed_gate_log', 'CLOSED_IN_QUEUE_SNAPSHOT_NOT_SUITE_COMPLETE')

    artifacts = [selected[key] for key in sorted(selected)]
    excluded = dict(patterns=['*.pt', '*.pth', '*.npz', '*.npy', '*.png', '*.gif', '*.jpg',
                              'datasets/**', '__pycache__/**'],
                    policy='Raw tensor checkpoints, datasets and emitted arrays stay at their original local paths; saved manifest hashes remain archived.',
                    checkpoint_references=[
                        {key: read(f'mnist/ra14-replay-r2/replay/{task}/RA14-replay/result.json')[key]
                         for key in ['checkpoint', 'checkpoint_sha256']} for task in ['toy', 'mnist']])
    inventory = dict(schema=1, status='PREPARATION_COMPLETE_FINAL_GATES_PENDING', prepared_utc=now,
                     repository_destination=DESTINATION, pull_request=223, local_root=str(ROOT),
                     scope='Plan only: no archive copy, repository mutation, GPU operation or tensor load.',
                     source_identities=dict(
                         manifest_rule=source_ra14['manifest_rule'],
                         ra13_manifest_sha256=source_ra13['package_sha256'],
                         ra14_manifest_sha256=source_ra14['package_sha256'],
                         run_digest_rule='Sorted particlegan-relative path, NUL, raw Python source bytes, NUL.',
                         ra13_run_sha256=read('RA13-SOURCE-REVIEW.json')['package_sha256'],
                         ra14_run_sha256=read('RA14-SOURCE-REVIEW.json')['package_sha256'],
                         config_sha256=source_ra14['config_sha256']),
                     closed_evidence=dict(learned=learned, ra14_replay_status=replay['replay_status'],
                                          ra14_replay_updates_total=replay['fresh_replay_updates_total'],
                                          fresh_ra14_training_updates=0,
                                          inherited_training_scope='Original RA13 labels and receipts retained; restoration-only source bridge.',
                                          replay_semantic_exclusion='birth_death.last.eval_seconds only',
                                          ra13_historical_full_suite=dict(status='PASS', passed=1404, skipped=12, subtests=18)),
                     test_counts=dict(cpu_optimizer_focused=13, cpu_optimizer_existing=35,
                                      settled_guard_focused=19, settled_guard_existing=35,
                                      settled_guard_default_r1=4, settled_guard_total=58,
                                      alias_focused=4, alias_existing=54, alias_total=58,
                                      counting_note='Overlapping controls are listed per receipt, not added as independent tests.'),
                     pending_validation=suite, artifacts=artifacts,
                     archive_totals=dict(files=len(artifacts), bytes=sum(item['bytes'] for item in artifacts)),
                     local_only=excluded,
                     pending_archive_groups=[
                         dict(status='PENDING', path='validation-ra14/scoreboard-all.json', required='Closed19/19 records and final source integrity'),
                         dict(status='PENDING', paths=['validation-ra14/runs/**/{result,acceptance-receipt,execution-receipt,job-header}.json',
                                                      'validation-ra14/runs/**/{metrics,rates}.jsonl',
                                                      'validation-ra14/moving/**/{COMPLETION,LAUNCH}.json',
                                                      'validation-ra14/moving/**/*.verdict.json',
                                                      'validation-ra14/logs/*.log'],
                              required='Bind only finalized small files; retain each original PASS/FAIL/INVALID verdict'),
                         dict(status='PENDING', paths=['validation-ra14/FULL-TESTS.json', 'validation-ra14/full-pytest.log'],
                              required='Actual final RA14 integrated suite completion, explicit skip count/reasons'),
                         dict(status='PENDING', path='final independent acceptance/source/archival manifest',
                              required='Coordinator closes final recommendation, copied-byte manifest and staged scope verification')])
    write('INVENTORY.json', inventory)

    completed = len(reported)
    report = f'''# Generalization evidence and archive plan — preparation snapshot

Prepared {now}. Destination: `{DESTINATION}` in PR223. This is an archive plan; the **19-gate qualification and final RA14 full suite remain PENDING**. No repository files, frozen evidence or numerical jobs were changed. The queue snapshot has {completed}/19 completed records, status `{board["status"]}`; individual observed verdicts are in `QUEUE-SNAPSHOT.json` and `INVENTORY.json`. Remaining records are explicitly pending.

## Closed learned evidence

| Original fresh run | Toy noisy precision / modes / TV | MNIST active39 FD / precision / recall | Reopen events |
|---|---|---|---|
| RA12-auto | 0.668579 / 22 / 0.336016 — Toy FAIL | 1.880969 / 0.767578 / 0.719238 | Toy820, MNIST202 |
| RA13-settled | 0.965332 / 25 / 0.052114 — Toy PASS | 0.544488 / 0.869141 / 0.847168 | Zero in both fixtures |

Both fresh runs completed the unchanged original2000 updates. RA13's nine postupdate Toy score/LR records exactly match frozen RA11; all ten MNIST score/LR records exactly match corrected E22, with ten confident classes. MNIST is a comparative regression fixture with **no newly invented numerical PASS threshold**. Backend selection is capability based: complete bounded raw-output frame plus finite-population feasibility selects feature cells/quarter generator+noise bases; otherwise reference KNN/original bases. Prior and critic bases are preserved.

RA14 performed **no fresh2000-update training**. Its configuration is byte identical to RA13. The only code change is checkpoint helper `_state_to_device`; all fresh training/sampling arithmetic is unchanged. The formal source bridge keeps the original RA13 source, inputs, checkpoints, labels and receipts. It supports carrying that evidence forward without relabeling it as a fresh RA14 training run.

## Repairs and retained failures

1. **Static endogenous R1 fires.** RA12 Toy820 follows the known KA2 objective epoch; MNIST202 occurs in an uncontracted game. Closed scalar reconstruction reproduces270 original rows and suppresses both first static fires with the settled-network witness plus objective-epoch rebase. It preserves original moving fires514/1021 exactly. Existing detector thresholds and all gradient groups are unchanged. The guard uses actual contracted generator/encoder/router/critic ownership; it is not a classifier for external target changes. Actual original moving qualification remains pending.
2. **CPU optimizer context.** Current Torch Adam's accelerator health check could open CUDA for CPU-only parameters. The device-scoped repair preserves accelerator delegation and exact CPU Adam/AdamW update/moment behavior. Thirteen focused checks plus35 existing controls passed without CUDA initialization; marker tests establish forwarding, not GPU numerical performance.
3. **Strict RA13 replay failure.** Both original native versus CPU-map continuations remain FAIL at1008–1010: cross-device conversion broke `last_block is blocks[-1]`, then moved-row rebase left the diagnostic tensor stale. Within the ten-update window, model/optimizer/active-history/RNG/loss/sample bytes match; this does not waive the strict semantic-state failure.
4. **RA14 restoration correction.** A transfer-local Tensor-identity memo preserves that alias. Both corrected original Toy/MNIST replays PASS: two branches×ten updates each,40 updates total; restored state, per-update semantic state, losses and noisy-primary samples agree. Native RA14 continuation also matches the sealed original RA13 native branch. Only original observational `birth_death.last.eval_seconds` is excluded. Distinct storage views/shared container identity are outside the repair.

Retain the RA14 r1 zero-update bridge preflight failure: raw shared configuration was compared before original N1024/z128/batch128 and tuple normalization. It performed zero updates/forwards/sampling and no CUDA initialization; corrected r2 changed adapter preparation only. Retain RA13's untrained first lane with missing external-validator guards and the corrected frozen r2 lane, plus private review preparation errors. Historical closure files keep their historical pending status; the final report should append current outcomes rather than edit them.

## Tests and pending qualification

| Scope | Closed evidence | Final status |
|---|---:|---|
| Settled guard/default controls | 19+35+4=58 PASS | Closed CPU contracts |
| Restoration alias/default controls | 4+54=58 PASS | Closed CPU contracts |
| RA13 integrated suite | 1404 PASS,12 skipped,18 subtests | Historical RA13 PASS |
| RA14 original13 ports +3 moving +3 static native | {completed}/19 observed in snapshot | **PENDING** |
| RA14 integrated full suite | No final attestation in this plan | **PENDING** |

These test counts overlap; do not sum them into an independent total. Preserve skip reasons. The current frozen lane retains original host/scorer/seeds/budgets, live noisy primary sampling, indexed row IDs, both scheduled moving rotations, and full native five-terminal/100k-holdout conjunction. Source freeze pins103 files including the unchanged external validator and lane. Final quality and validity are distinct: original FAIL stays FAIL even with valid evidence.

## Archive and recommendation

`INVENTORY.json` lists {len(artifacts)} already available small files ({sum(item['bytes'] for item in artifacts)/1048576:.2f}MiB), their source paths, target relative paths and SHA256. It includes source/configs, patches/helpers, source/run/failure closures, metric/LR curves, observation traces and independent reviews. Candidate closure digests use compact source-hash JSON; runner digests use path/NUL/source bytes. Both conventions are explicit.

Raw tensor checkpoints, datasets and saved clouds stay local; archived input manifests retain their original path/hash references. No rescoring or checkpoint conversion is needed for archival. After root closes all19 gates and the actual RA14 full suite, append the final receipts/logs and source hashes, state every failure, then byte-verify the archive and intended staged Git scope. Keep original RA12/RA13 failures and bridge preparation errors.

The learned recovery and replay repair support RA14 as the current prospective general candidate. A completed general recommendation requires the pending gates/full suite. Preserve E22 as the broad baseline in comparisons; these two learned fixtures and short replay windows do not establish universal optimizer-shock discrimination, high-dimensional moment coverage, serving equivalence, or scalability.
'''
    write('REPORT-DRAFT.md', report)
    write('CONSISTENCY-RECEIPT.json', dict(status='PASS_ARCHIVE_PREPARATION_WITH_FINAL_GATES_PENDING',
        prepared_utc=now, source_reads='stdlib source/JSON/log only', tensor_loads=0,
        model_calls=0, numerical_reruns=0, repository_writes=0,
        assertions=['Closed RA14 replay actual PASS/40 updates/zero training',
                    'Original RA13 training parity and replay FAIL retained',
                    'Actual RA12/13 final metrics copied from completed result JSON',
                    'Observed queue verdicts agree with corresponding completed receipt',
                    'RA13 suite count quoted from closed log, never transferred to RA14'],
        files={name:digest((HERE / name).read_bytes()) for name in
               ['build_inventory.py', 'QUEUE-SNAPSHOT.json', 'INVENTORY.json', 'REPORT-DRAFT.md']}))
    print(json.dumps(dict(status='PREPARED_FINAL_GATES_PENDING', files=len(artifacts),
                          bytes=sum(item['bytes'] for item in artifacts), completed_snapshot=completed,
                          report_sha256=digest((HERE/'REPORT-DRAFT.md').read_bytes()),
                          inventory_sha256=digest((HERE/'INVENTORY.json').read_bytes())), sort_keys=True))


if __name__ == '__main__':
    main()
