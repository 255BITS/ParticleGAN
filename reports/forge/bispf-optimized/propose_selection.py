"""Validate completed BISPF evidence and propose one whole-row pin; never apply it.

Original wide-task request/evidence/result receipts must be hydrated first.
Performance inputs are immutable compact manifests, not raw timing traces.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.forge.contracts import atomic_json, file_hash, identifier, read_json, stable_hash
from experiments.forge import scaling_publication as performance
from experiments.forge.trainer_families import CURRENT_SELECTION, family_row_pin
from experiments.forge.views import task_fingerprint
from reports.forge.regenerate_technique_inventory import _published_report, _validate_published_row, project_receipt

NATIVE = 'bispf-v2'
R1R2 = 'recipe-prior-refactor-r1r2-v1'
WIDE = ('img_stripes2_w64_optimizer_diagnostic', 'img_bars4_w64_optimizer_diagnostic')


def require(condition, message):
    if not condition:
        raise ValueError('BISPF selection: ' + message)


def statuses(row):
    rows = row.get('tasks', []) + row.get('nonrequired_tasks', [])
    values = {item['task_id']: item['status'] for item in rows}
    require(len(values) == len(rows), 'duplicate numerical task verdict')
    return values


def recipe_comparison_hash(recipe):
    """Only the new implicit polar default may differ in an archived packet."""
    recipe = deepcopy(recipe)
    if recipe.get('optimizer_convolution_rule', 'polar') == 'polar':
        recipe.pop('optimizer_convolution_rule', None)
    return stable_hash(recipe)


def optimizer_delta(before, after):
    before = {**before, 'optimizer_convolution_rule': before.get('optimizer_convolution_rule', 'polar')}
    after = {**after, 'optimizer_convolution_rule': after.get('optimizer_convolution_rule', 'polar')}
    delta = {key: [before.get(key), after.get(key)] for key in before.keys() | after.keys()
             if before.get(key) != after.get(key)}
    allowed = [
        {'optimizer_svd_backend': ['native', 'newton_schulz']},
        {'optimizer_convolution_rule': ['polar', 'frobenius']},
        {'optimizer_convolution_rule': ['polar', 'quintic']},
    ]
    require(delta in allowed, 'undeclared global recipe delta')
    require(after.get('optimizer_family') == 'dualnorm' and after.get('optimizer_convolution') == 'per_offset',
            'optimized rule requires full per-offset DualNorm')
    rule = after['optimizer_convolution_rule']
    backend = after.get('optimizer_svd_backend', 'native')
    require((rule == 'polar' and backend == 'newton_schulz') or (rule != 'polar' and backend == 'native'),
            'optimizer rule and SVD backend disagree')
    return delta, dict(convolution_rule=rule, svd_backend=backend,
                       convolution_update_version=('per_offset_polar_v1' if rule == 'polar'
                                                   else f'per_offset_{rule}_hybrid_v1'))


def cohort_bindings(bindings, commit, *, published):
    """Validate reporter-added origins before comparing the scientific packet."""
    require(isinstance(commit, str) and len(commit) == 40 and set(commit) <= set('0123456789abcdef'),
            'invalid frozen source commit')
    fields = {'source_origin_commit', 'recorded_source_origin_commits'}
    origins = {key: bindings[key] for key in fields if key in bindings}
    verified = {'source_origin_commit': commit, 'recorded_source_origin_commits': [commit]}
    # The independently graded actual row precedes _publication_provenance:
    # its source reader has no Git origin, so it records None (or no field).
    # A nonempty origin claim must contain both exact publication annotations.
    allowed = [verified] if published else [{}, {'source_origin_commit': None}, verified]
    require(origins in allowed, 'source origin annotations differ from frozen commit or are partial')
    return {key: value for key, value in bindings.items() if key not in fields}


def quality_proof(root, source_report, candidate):
    report, pointers = _published_report(root, source_report)
    require(report.get('publication_scope') == 'frozen_source', 'quality must retain an independently frozen source')
    selected = [row for row in report['rows'] if row['candidate_id'] == candidate and row.get('attempt_ids')]
    require(len(selected) == 1, 'choose one complete candidate/runtime row')
    row = deepcopy(selected[0])
    _validate_published_row(root, report, row)
    proof = report.get('comparison_policy_projection', {})
    actual = proof.get('actual_qualification', {})
    require(proof.get('actual_qualification_sha256') == stable_hash(actual), 'actual executed-policy proof changed')
    actual_rows = [item for item in actual.get('rows', []) if item['candidate_revision'] == row['candidate_revision']]
    require(len(actual_rows) == 1, 'missing unique actual executed-policy row')
    measured = actual_rows[0]
    commit = report['frozen_source']['commit']
    require(cohort_bindings(measured['bindings'], commit, published=False)
            == cohort_bindings(row['bindings'], commit, published=True),
            'actual executed-policy scientific bindings differ from projected row')
    require(all(measured.get(key) == row.get(key) for key in
                ('candidate_id', 'candidate_revision', 'cohort', 'attempt_ids', 'runtime_cohort')),
            'actual executed-policy cohort differs from projected row')
    view = read_json(root / 'configs/forge/views/discriminator_stability.json')
    require(actual.get('policy_fingerprint') == stable_hash(view), 'actual qualification uses another declared policy')
    require(measured['status'] == 'PASS' and measured['qualified_tier'] == 3, 'complete ordinary qualification has not passed')
    require(not measured.get('pending_readout'), 'conclude the ordinary quality study before selection')
    require(statuses(measured) == statuses(row), 'comparison projection changed numerical verdicts')
    inventory = read_json(root / 'reports/forge/technique-inventory.json')
    incumbent = [item for item in inventory['rows'] if item.get('trainer_family') == 'bispf']
    require(len(incumbent) == 1, 'missing unique incumbent BISPF row')
    incumbent = incumbent[0]
    required = {a['task'] for a in view['assignments'] if a['importance'] == 'required'}
    old_passes = {name for name, state in statuses(incumbent).items() if state == 'PASS'}
    require(all(statuses(row).get(name) == 'PASS' for name in required | old_passes), 'a counted gate or incumbent pass regressed')
    require(row['bindings']['task_contracts'] == incumbent['bindings']['task_contracts'], 'task contracts differ from incumbent')
    before = inventory['recipe_contracts'][incumbent['bindings']['recipe_sha256']]
    after = report['recipe_contracts'][row['bindings']['recipe_sha256']]
    require(stable_hash(before) == incumbent['bindings']['recipe_sha256']
            and stable_hash(after) == row['bindings']['recipe_sha256'], 'quality recipe contract hash differs')
    delta, optimizer = optimizer_delta(before, after)
    row['trainer_family'] = 'bispf'
    return row, dict(source_commit=report['frozen_source']['commit'], source_digest=row['bindings']['source_digest'],
                     candidate_revision=row['candidate_revision'], actual_policy_revision=actual['view_revision'],
                     recipe_sha256=row['bindings']['recipe_sha256'], native_recipe_sha256=incumbent['bindings']['recipe_sha256'],
                     recipe_comparison_sha256=recipe_comparison_hash(after),
                     native_recipe_comparison_sha256=recipe_comparison_hash(before), optimizer_contract=optimizer,
                     incumbent_passes=sorted(old_passes), counted_passes=sorted(required), recipe_delta=delta,
                     source_report_sha256=pointers['json_sha256'])


def performance_proof(root, manifests, candidate, quality):
    expected = [a['task'] for a in read_json(root / 'configs/forge/views/dcgan_performance.json')['assignments']]
    records, pointers = {}, []
    for name in manifests:
        path = Path(name)
        if not path.is_absolute():
            path = root / path
        manifest = read_json(path)
        performance._validate_manifest(root, manifest)
        pointers.append(dict(path=str(path), sha256=file_hash(path)))
        for entry in manifest['entries']:
            if entry['candidate_id'] not in {candidate, NATIVE, R1R2}:
                continue
            receipt = performance.validate_receipt(read_json(root / entry['receipt']))
            key = entry['candidate_id'], entry['task_id']
            require(key not in records, 'duplicate performance candidate/task cell')
            require(performance.performance_status(receipt) == 'COMPLETE', 'incomplete performance cell')
            require(receipt['identity']['source']['commit'] == quality['source_commit'], 'performance source differs from qualified source')
            require(receipt['forge_binding']['source_digest'] == quality['source_digest'], 'performance source manifest differs from qualification')
            if entry['candidate_id'] == candidate:
                require(receipt['forge_binding']['candidate_revision'] == quality['candidate_revision'], 'performance candidate revision differs')
                require(receipt['recipe'].get('optimizer_convolution_rule', 'polar')
                        == quality['optimizer_contract']['convolution_rule']
                        and receipt['recipe'].get('optimizer_svd_backend', 'native')
                        == quality['optimizer_contract']['svd_backend'], 'performance optimizer rule differs from qualification')
            records[key] = receipt
    require(set(records) == {(c, t) for c in (candidate, NATIVE, R1R2) for t in expected}, 'measure all six workloads for candidate and both controls')
    require(len({r['identity']['global_recipe_sha256'] for (c, _), r in records.items() if c == candidate}) == 1,
            'performance uses several candidate recipes')
    observations = {}
    for task in expected:
        fast, native, r1 = [records[c, task] for c in (candidate, NATIVE, R1R2)]
        def comparison(receipt):
            identity = receipt['identity']
            return dict(source=identity['source']['sha256'], runtime=identity['runtime_sha256'],
                        hardware=performance.hardware_key(receipt), task=identity['task_sha256'],
                        initialization=identity['initialization_sha256'],
                        model=performance._initial_model_for_comparison(receipt), batches=identity['batch_sequence_sha256'])
        require(comparison(fast) == comparison(native) == comparison(r1), 'performance controls differ in workload, initialization, data, runtime or physical GPU')
        latency = [r['metrics']['update_ms_median'] for r in (fast, native, r1)]
        observations[task] = dict(optimized_ms=latency[0], native_ms=latency[1], r1r2_ms=latency[2],
                                 observed_native_speedup=latency[1] / latency[0], observed_r1r2_ratio=latency[0] / latency[2])
    require(all(observations[t]['observed_native_speedup'] >= 8 for t in ('dcgan_performance_r64', 'dcgan_performance_r256')),
            'declared observed 8x speedup target missed')
    require(observations['dcgan_performance_r256']['observed_r1r2_ratio'] <= 2, 'declared observed <=2x R1/R2 target missed')
    clear = all(r['hardware'].get('contention_status') == 'clear' for r in records.values())
    return dict(manifests=pointers, observations=observations, certified_paired_speed_ranking=clear,
                timing_scope='observed complete-update medians; unknown contention remains indicative')


def wide_proof(root, attempts, candidate, quality):
    cells, pointers = {}, []
    require(len(attempts) == len(set(attempts)) == 4, 'provide all four native/optimized wide attempts')
    for attempt in attempts:
        identifier(attempt, 'wide attempt')
        compact = project_receipt(root, attempt)
        directory = root / 'reports/forge/attempts' / attempt
        envelope = read_json(directory / 'request.json')
        request = envelope.get('request', envelope)
        result = read_json(directory / 'result.json')
        name = request['candidate']['id']
        require(name in {candidate, NATIVE} and request['protocol']['seed'] == 0, 'wide attempt uses another candidate or seed')
        expected_recipe = (quality['recipe_comparison_sha256'] if name == candidate
                           else quality['native_recipe_comparison_sha256'])
        require(recipe_comparison_hash(request['candidate']['resolved_recipe']) == expected_recipe,
                'wide diagnostic changed the global recipe')
        if name == candidate:
            require(request['candidate_revision'] == quality['candidate_revision'], 'wide diagnostic candidate revision differs')
        require(request['source']['origin_commit'] == quality['source_commit'] and request['source']['digest'] == quality['source_digest'],
                'wide diagnostic source differs from qualified source')
        require(request['view'].get('evidence_scope') == 'research_diagnostic' and request['view']['id'] == 'bispf_fastpath_quality',
                'wide attempt has another evidence scope')
        for row in result['task_results']:
            task = row['task_id']
            require(task in WIDE and (name, task) not in cells, 'unexpected or duplicate wide task')
            require(task_fingerprint(request['tasks'][task]) == task_fingerprint(
                read_json(root / 'configs/forge/tasks' / (task + '.json'))), 'wide task differs from its declared contract')
            evidence, cost = row['evidence'], row['cost']
            require(row['gate_status'] in {'PASS', 'FAIL'}, 'wide diagnostic incomplete or invalid')
            require(cost.get('completed_steps') == 600 and len(evidence['observations']) == 24, 'wide diagnostic has an incomplete budget')
            require(evidence['guards']['all_finite'] and evidence['guards']['unintended_rng_deviations'] == 0, 'nonfinite or shifted RNG wide run')
            require(all(cost['optimizer_updates'].get(k) == 600 for k in ('generator', 'discriminator', 'prior')), 'wide optimizer budget incomplete')
            gif = Path(request['queue_root']) / request['campaign_id'] / attempt / 'training.gif'
            require(gif.is_file(), 'missing actual-training wide GIF')
            cells[name, task] = dict(status=row['gate_status'], task_contract=task_fingerprint(request['tasks'][task]),
                                    initialization=row['initialization'], rng=row['rng'], metrics=row['metrics'],
                                    training_gif=dict(path=str(gif), sha256=file_hash(gif)))
        pointers.append(dict(attempt=attempt, canonical_result_sha256=compact['provenance']['canonical_result_hash'],
                             original_files=compact['provenance']['original_files']))
    require(set(cells) == {(c, t) for c in (candidate, NATIVE) for t in WIDE}, 'wide diagnostic matrix incomplete')
    outcomes = {}
    for task in WIDE:
        native, fast = cells[NATIVE, task], cells[candidate, task]
        require(all(native[k] == fast[k] for k in ('task_contract', 'initialization', 'rng')), 'wide architecture, initialization or streams differ')
        require(native['status'] != 'PASS' or fast['status'] == 'PASS', 'wide native pass regressed')
        outcomes[task] = dict(native=native, optimized=fast)
    require(cells[candidate, WIDE[0]]['status'] == 'PASS', 'require positive optimized wide stripes evidence')
    return dict(attempts=pointers, outcomes=outcomes, qualification_reuse=False,
                scope='architecture portability; ties preserve native passes; shared failures remain visible; no natural-image quality claim')


def propose(root, *, source_report, candidate, manifests, wide_attempts):
    root = Path(root).resolve()
    identifier(candidate, 'candidate')
    row, quality = quality_proof(root, source_report, candidate)
    speed = performance_proof(root, manifests, candidate, quality)
    wide = wide_proof(root, wide_attempts, candidate, quality)
    current = root / CURRENT_SELECTION
    card = read_json(current)
    require(card.get('default_adoption') is False, 'selection must preserve public-default adoption policy')
    archive = CURRENT_SELECTION.parent / 'history' / ('family-current-' + file_hash(current) + '.json')
    pin = family_row_pin(row, selection_kind='current_measurement', measurement_views=['discriminator_stability'],
                         reason='Preserve every incumbent pass with one globally configured optimized BISPF recipe. Actual executed-policy qualification, older comparison-policy tiers, exercised wide-channel diagnostics and observed performance targets retain their separate source-bound proofs. No public-default adoption or natural-image quality claim.')
    require(sum(p['trainer_family'] == 'bispf' for p in card['selections']) == 1, 'missing unique current BISPF pin')
    card['selections'] = [pin if p['trainer_family'] == 'bispf' else p for p in card['selections']]
    card['previous_selection'] = dict(path=archive.as_posix(), sha256=file_hash(current))
    proposal = dict(schema='bispf_optimized_selection_proposal_v1', candidate=candidate, ready=True,
                    selection_applied=False, default_adoption=False, quality=quality, performance=speed, wide_quality=wide,
                    original_selection_sha256=file_hash(current), exact_archive_path=archive.as_posix(), proposed_selection=card)
    proposal['input_digest'] = stable_hash(proposal)
    return proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--performance-manifest', type=Path, action='append', required=True)
    parser.add_argument('--wide-attempt', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    proposal = propose(args.root, source_report=args.source_report, candidate=args.candidate,
                       manifests=args.performance_manifest, wide_attempts=args.wide_attempt)
    atomic_json(args.output, proposal)
    print(json.dumps(dict(output=str(args.output), ready=True, selection_applied=False, input_digest=proposal['input_digest'])))


if __name__ == '__main__':
    main()
