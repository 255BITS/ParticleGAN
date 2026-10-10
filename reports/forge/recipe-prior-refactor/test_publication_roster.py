"""Reporting discovery must retain exact study controls and immutable source reads."""
import ast
import importlib.util
import inspect
import json
from pathlib import Path
import textwrap

import pytest
from experiments.forge import planning, sources

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('prior_roster_publication_test', ROOT / 'reports/forge/regenerate_technique_inventory.py')
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)
REGISTRATION = ROOT / 'reports/forge/recipe-prior-refactor/ordinary-registration.json'
registered = json.loads(REGISTRATION.read_text())
IDS = {plan['candidate_id'] for plan in registered['families'].values()}
BASELINE = json.loads((REGISTRATION.parent / 'baseline.json').read_text())
CONTROLS = [leader['selection']['candidate_id'] for leader in BASELINE['leaders']
            if leader['selection']['family'] not in {'atlas', 'e22'}]


def reader_program():
    function = ast.parse(inspect.getsource(publisher._frozen_report)).body[0]
    return next(node.value.right.value for node in ast.walk(function)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'program' for target in node.targets))


@pytest.mark.parametrize('control', CONTROLS)
def test_scoped_enumeration_preserves_actual_original_control_lookup(monkeypatch, control):
    expected = planning.load_idea(ROOT, control)
    monkeypatch.setattr(planning, 'declaration_paths', planning.declaration_paths)
    monkeypatch.setattr(planning, 'load_idea', planning.load_idea)
    program = reader_program()
    fragment = textwrap.dedent(program[program.index('        original_declaration_paths ='):program.index('    def question_request(')])
    namespace = dict(planning=planning, root=ROOT, json=json,
                     recorded_candidates=IDS, declaration_roster={'candidate_ids': sorted(IDS)})
    exec(fragment, namespace)
    assert {path.stem for path in planning.declaration_paths(ROOT)} == IDS
    assert planning.load_idea(ROOT, control) == expected
    assert {path.stem for path in planning.declaration_paths(ROOT)} == IDS


@pytest.fixture
def descriptor_probe(monkeypatch):
    digest = next(iter(registered['families'].values()))['source_digest']
    monkeypatch.setattr(publisher, '_resolve_commit', lambda root, commit: commit)
    monkeypatch.setattr(publisher, '_receipt_manifests', lambda root, commit: {digest: {}})
    records = [{'request': {'candidate': {'id': candidate}}} for candidate in IDS - {'atlas', 'e22'}]
    monkeypatch.setattr(publisher, '_recorded_studies', lambda root, manifests: (records, {}))
    class Extracted(Exception):
        pass
    def stop_after_validation(*args, **kwargs):
        raise Extracted
    monkeypatch.setattr(publisher, '_extract_source', stop_after_validation)
    return records, digest, Extracted


def test_exact_executed_registration_reaches_extraction(descriptor_probe, tmp_path):
    _, _, extracted = descriptor_probe
    with pytest.raises(extracted):
        publisher._frozen_report(ROOT, 'a0493ca3c213d559523eadc76e244ec9f9c1f81f',
            view_id='discriminator_stability', execution_backend='cuda', temporary=tmp_path,
            declaration_registration=REGISTRATION)


@pytest.mark.parametrize('mutation', ['omitted_measured', 'wrong_source', 'wrong_backend'])
def test_descriptor_rejects_missing_measured_or_source_cohort(descriptor_probe, monkeypatch, tmp_path, mutation):
    records, _, _ = descriptor_probe
    backend = 'cuda'
    if mutation == 'omitted_measured':
        records[0]['request']['candidate']['id'] = 'omitted-original-measurement'
    elif mutation == 'wrong_source':
        monkeypatch.setattr(publisher, '_receipt_manifests', lambda root, commit: {'wrong-source': {}})
    else:
        backend = 'cpu'
    with pytest.raises(ValueError):
        publisher._frozen_report(ROOT, 'a0493ca3c213d559523eadc76e244ec9f9c1f81f',
            view_id='discriminator_stability', execution_backend=backend, temporary=tmp_path,
            declaration_registration=REGISTRATION)


@pytest.mark.parametrize('conflict', ['--advance-policy', '--refresh-publication', '--ownership-migration'])
def test_independent_report_cli_cannot_mutate_current_policy(conflict):
    arguments = ['--output-prefix', '/tmp/independent-evidence', conflict]
    if conflict == '--ownership-migration':
        arguments.append('map.json')
    with pytest.raises(SystemExit) as error:
        publisher.main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize('behavior', ['copy_and_generator_key', 'other_root_bypass'])
def test_immutable_source_cache_preserves_values_and_bypasses_other_roots(monkeypatch, tmp_path, behavior):
    calls = []
    def original(root, extras=()):
        calls.append((Path(root).resolve(), tuple(extras)))
        return {'files': {'source.py': str(len(calls))}, 'digest': str(len(calls))}
    monkeypatch.setattr(sources, 'inspect_source', original)
    program = reader_program()
    fragment = program[program.index('from copy import deepcopy'):program.index('# A read-only board')]
    namespace = dict(Path=Path, root=tmp_path)
    exec(fragment, namespace)
    cached = namespace['immutable_source_inspection']
    if behavior == 'copy_and_generator_key':
        first = cached(tmp_path, (p for p in ['b.py', 'a.py']))
        first['files']['source.py'] = 'caller mutation'
        second = cached(tmp_path, ['a.py', 'b.py'])
        assert second['files']['source.py'] == '1'
        assert calls == [(tmp_path.resolve(), ('a.py', 'b.py'))]
    else:
        other = tmp_path / 'other-root'
        assert cached(other)['digest'] != cached(other)['digest']
        assert namespace['source_inspection_counts']['other_root_bypasses'] == 2
        assert not namespace['source_inspection_cache']


def test_missing_fallback_preserves_controls_and_restores_lookup_after_error():
    old_paths, old_lookup = planning.declaration_paths, planning.load_idea
    expected = old_lookup(ROOT, CONTROLS[0])
    with pytest.raises(RuntimeError, match='stop read-only reduction'):
        with publisher._missing_declaration_enumeration(ROOT, IDS):
            assert {p.stem for p in planning.declaration_paths(ROOT)} == IDS
            assert planning.load_idea(ROOT, CONTROLS[0]) == expected
            assert {p.stem for p in planning.declaration_paths(ROOT)} == IDS
            raise RuntimeError('stop read-only reduction')
    assert planning.declaration_paths is old_paths and planning.load_idea is old_lookup


def test_fallback_inspection_binds_saved_bytes_restores_aliases_and_copies_packets(tmp_path, monkeypatch):
    from experiments.forge.contracts import file_hash, stable_hash
    path = tmp_path / 'source.py'
    path.write_text('saved_source = 1\n')
    files = {'source.py': file_hash(path)}
    manifest = dict(files=files, digest=stable_hash(files))
    calls = []
    def original(root, extra_paths=()):
        calls.append((root, tuple(extra_paths)))
        return dict(files=dict(files), digest=stable_hash(files), origin_commit='exact-recorded')
    monkeypatch.setattr(sources, 'inspect_source', original)
    old_planning = planning.inspect_source
    with publisher._unchanged_fallback_source_inspection(tmp_path, {manifest['digest']: manifest}) as counts:
        packet = planning.inspect_source(tmp_path, iter(['source.py']))
        packet['files']['source.py'] = 'tampered copy'
        assert planning.inspect_source(tmp_path, iter(['source.py']))['files'] == files
        assert counts['hits'] == counts['misses'] == 1
    assert sources.inspect_source is original and planning.inspect_source is old_planning
    assert len(calls) == 1
    path.write_text('changed_source = 2\n')
    with pytest.raises(ValueError, match='scientific source changed'):
        with publisher._unchanged_fallback_source_inspection(tmp_path, {manifest['digest']: manifest}):
            pytest.fail('changed source entered cache scope')


@pytest.mark.parametrize('tamper', [None, 'source', 'gate'])
def test_archived_solution_base_is_exact_old_whole_row_without_current_credit(tamper):
    """Navigation cannot fabricate a canonical recipe or fill a current gate."""
    from copy import deepcopy
    import subprocess
    from experiments.forge.trainer_families import scientific_row_hash
    migration = publisher._ownership_migration_helper()
    old = json.loads(subprocess.check_output(['git', 'show',
        'a0493ca3c213d559523eadc76e244ec9f9c1f81f:reports/forge/technique-inventory.json'], cwd=ROOT))
    original = next(row for row in old['rows'] if row['trainer_family'] == 'bcap-pure')
    row = deepcopy(original)
    if tamper == 'source':
        row['bindings']['source_digest'] = '0' * 64
    elif tamper == 'gate':
        row['tasks'][0]['status'] = 'PASS'
    data = json.loads((REGISTRATION.parent / 'ownership-migration.json').read_text())
    card = {'ownership_migration': {'declaration': {
        'path': 'reports/forge/recipe-prior-refactor/ownership-migration.json',
        'sha256': publisher.file_hash(REGISTRATION.parent / 'ownership-migration.json')},
        'archived_only_selections': data['archived_only_selections']}}
    current = {'rows': [{'trainer_family': 'bcap-dualnorm'}]}
    reports = [(None, {'json_sha256': original['publication_key']}, old, {'original': row})]
    if tamper is not None:
        with pytest.raises(ValueError, match='one exact verified original archived row'):
            migration.restore_archived_solution_rows(ROOT, current, card, reports,
                validate_row=publisher._validate_published_row)
        assert current['rows'][:1] == [{'trainer_family': 'bcap-dualnorm'}]
        assert len(current['rows']) == 1
        return
    restored = migration.restore_archived_solution_rows(ROOT, current, card, reports,
        validate_row=publisher._validate_published_row)
    assert scientific_row_hash(current['rows'][-1]) == scientific_row_hash(original)
    assert current['rows'][-1]['bindings'] == original['bindings']
    assert current['rows'][-1]['tasks'] == original['tasks']
    assert current['rows'][-1]['selection']['selection_kind'] == 'archived_base_presentation_only'
    assert not original['attempt_ids'] and original['status'] == 'BLOCKED'
    assert restored[0]['scientific_row_sha256'] == scientific_row_hash(original)
    assert restored[0]['qualification_input'] is False
    assert current['rows'][:1] == [{'trainer_family': 'bcap-dualnorm'}]
