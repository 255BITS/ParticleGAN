"""Bind completed software checks and preservation assertions; launches no training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from reproduce import SOURCES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]

    def git(*arguments):
        return subprocess.check_output(['git', *arguments], cwd=root)

    def identity(path):
        data = path.read_bytes()
        return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

    batches = []
    unique = set()
    for name in ('combined-existing', 'combined-hosts', 'combined-new', 'combined-strict',
                 'combined-final-assertions', 'combined-boundary'):
        path = args.archive / 'logs' / (name + '.xml')
        cases = ET.parse(path).getroot().findall('.//testcase')
        assert all(not any(case.find(tag) is not None for tag in ('failure', 'error', 'skipped'))
                   for case in cases), name
        unique.update((case.attrib['classname'], case.attrib['name']) for case in cases)
        batches.append({'suite': name, 'passed': len(cases), 'junit': identity(path),
                        'stdout': identity(path.with_suffix('.log'))})
    assert len(unique) == 939
    preserved = []
    for role, commit in SOURCES.items():
        assert subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=root).returncode == 0
        if role == 'develop':
            continue
        name = {'moment_tether': 'covariance', 'initial_mmd': 'mmd'}.get(role, role)
        paths = ['reports/forge/bcap-prior-search/' + name,
                 'reports/forge/configuration-search/bcap-prior-' + name + '-search-v1.json',
                 'configs/forge/searches/bcap-prior-' + name + '-search-v1.json']
        for path in paths:
            assert git('ls-tree', commit, '--', path).strip(), (role, path)
        assert not git('diff', '--name-only', commit, '--', *paths).strip(), role
        preserved.append({'feature': role, 'source': commit, 'paths': paths, 'unchanged': True})
    canonical = ['reports/forge/technique-inventory.json', 'reports/forge/technique-inventory.md',
                 'configs/forge/selections/family-current-v1.json']
    for path in canonical:
        assert git('ls-tree', SOURCES['develop'], '--', path).strip(), path
    assert not git('diff', '--name-only', SOURCES['develop'], '--', *canonical).strip()
    modified_reports = set(git('diff', '--name-only', '--diff-filter=M', SOURCES['develop'],
                               '--', 'reports/forge').decode().splitlines())
    assert modified_reports == {'reports/forge/EXPERIMENT_MEMORY.md',
                                'reports/forge/compilation.json',
                                'reports/forge/publication-records.json'}
    compilation = json.loads((root / 'reports/forge/compilation.json').read_text())
    original_compilation = json.loads(git('show', SOURCES['develop'] + ':reports/forge/compilation.json'))
    retained_compilation = {name: compilation[name] for name in (
        'qualification_refresh', 'automation_digest', 'operational_lifecycle_digest')}
    assert all(value == original_compilation[name] for name, value in retained_compilation.items())
    check = json.loads((args.archive / 'logs/compile-check.log').read_text())
    assert check['status'] == 'CURRENT' and check['reducer_current']
    validation = json.loads((args.archive / 'logs/forge-validate-final.log').read_text())
    assert validation['family_documentation']['status'] == 'valid'
    task_paths = git('diff', '--name-only', SOURCES['develop'], '--', 'configs/forge/tasks').decode().splitlines()
    tasks = []
    for relative in task_paths:
        value = json.loads((root / relative).read_text())
        original = json.loads(git('show', SOURCES['develop'] + ':' + relative))
        for name, digest in value['evaluation']['sources'].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, (relative, name)
        for record in (value, original):
            record['evaluation'].pop('sources', None)
            record['evaluation'].pop('evaluator_revision', None)
        assert value == original, relative
        tasks.append(relative)
    assert len(tasks) == 11
    changed = git('diff', '--name-only', SOURCES['develop'], '--', 'particlegan', 'experiments',
                  'benchmarks', 'tests', 'docs', 'configs/forge/tasks').decode().splitlines()
    changed += ['tests/test_prior_feature_integration.py',
                'reports/forge/prior-integration-review/probe.py',
                'reports/forge/prior-integration-review/reproduce.py',
                'reports/forge/prior-integration-review/validate_receipt.py']
    receipt = {
        'recommendation': 'safe_after_explicit_integration_fixes',
        'scope': 'bounded public-API software compatibility; no scientific qualification',
        'develop': SOURCES['develop'], 'reviewed_feature_heads': SOURCES,
        'initial_union_before_compatibility_fixes': 'f60609e9fa0b9a4e23cb636ec51ce683c4b45bd4',
        'archive': str(args.archive), 'protocol_seed': 0,
        'new_scientific_experiments': 0, 'new_queue_admissions': 0,
        'archived_gate_rescoring': False, 'default_adoption': False,
        'remote_push_or_merge': False,
        'measurements': [{'name': 'distinct_passing_software_tests', 'score': len(unique), 'target': 939},
                         {'name': 'combined_compatibility_cases', 'score': 47, 'target': 47},
                         {'name': 'current_task_contracts_unchanged', 'score': len(tasks), 'target': 11},
                         {'name': 'original_feature_publications_unchanged', 'score': len(preserved), 'target': 5}],
        'batches': batches, 'passing_executions_including_rechecks': sum(b['passed'] for b in batches),
        'pre_fix_findings': [
            {'case': 'old positive Coulomb checkpoint without SIGReg packet',
             'exception': 'ValueError: active prior regularizer requires checkpoint state'},
            {'case': 'positive global coefficient on non-sampled host without module',
             'exception': "AttributeError: 'NoneType' object has no attribute 'state_dict'"}],
        'corrected_software_fixture_attempt': {
            'cause': 'recipe_field_active task argument requires keyword',
            'junit': identity(args.archive / 'logs/combined-boundary-test-signature-error.xml'),
            'stdout': identity(args.archive / 'logs/combined-boundary-test-signature-error.log')},
        'frozen_fixtures': {role: identity(args.archive / 'fixtures' / (role + '.pt')) for role in SOURCES},
        'preserved_feature_publications': preserved,
        'canonical_selection_and_leaderboard_unchanged': canonical,
        'preserved_qualification_automation_lifecycle': retained_compilation,
        'summaries_only_memory_status': check['status'],
        'publication_checks': {name: identity(args.archive / 'logs' / (name + '.log'))
                               for name in ('forge-validate-final', 'compile-summaries-final', 'compile-check')},
        'current_source_metadata_only_tasks': tasks,
        'tested_source_files': {name: identity(root / name) for name in sorted(set(changed))},
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print('PASS: 939 distinct tests; all five publications and 11 task contracts preserved')


if __name__ == '__main__':
    main()
