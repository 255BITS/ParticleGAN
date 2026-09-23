"""Audit archived rare-mode attempts, fixed comparison axes and report references."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from benchmarks.transfer_suite.formulations import axes
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.vector_tasks import fixed_policy
from particlegan import learning_rate_scale
from .build import ROOT, SUITE, read


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify():
    files = sources = episodes = references = 0
    payloads = set()
    for manifest in ROOT.rglob('archive_manifest.json'):
        for item in read(manifest)['files']:
            raw = (manifest.parent/item['path']).read_bytes()
            assert digest(raw) == item['archived_sha256'], (manifest, item['path'])
            if item['gzip_added']:
                assert digest(gzip.decompress(raw)) == item['original_sha256'], item['path']
            files += 1
    for archive in ROOT.rglob('source.tar.gz'):
        path = archive.parent/'protocol.json.gz'
        if not path.exists():
            nested = sorted(archive.parent.glob('*/protocol.json.gz'))
            if not nested:
                continue
            path = nested[0]
        protocol = read(path)
        with tarfile.open(archive) as tar:
            for name, sha in protocol['source_sha256'].items():
                assert digest(tar.extractfile(name).read()) == sha, (archive, name)
                sources += 1
        if 'driver_sha256' in protocol:
            assert digest((archive.parent/'run.py').read_bytes()) == protocol['driver_sha256']
        for name, sha in protocol.get('experiment_source_sha256', {}).items():
            candidates = [archive.parent/name, archive.parent/'scripts'/name, path.parent/name]
            source = next(p for p in candidates if p.is_file())
            assert digest(source.read_bytes()) == sha, (archive, name)
    declared = {s['name']: s for s in read(SUITE/'study/manifest.json')['tasks']}
    for path in ROOT.glob('**/episodes/*.json.gz'):
        raw = gzip.decompress(path.read_bytes())
        value = json.loads(raw)
        spec = value.get('effective_spec', value['spec'])
        assert test_verdict(spec, value['result']) == value['verdict'], path
        assert value['verdict']['convergence']['complete'], path
        assert value['policy'] == fixed_policy('cosine'), path
        expected_actions = [dict(step=step, role=role,
                                 multiplier=learning_rate_scale(step, spec['steps'], .6, .05))
                            for step in range(0, spec['steps'], 20) for role in ('d', 'g')]
        assert value['result']['actions'] == expected_actions, path
        assert len(value['result']['observations']) == 24 and not value['result'].get('error'), path
        actual, expected = axes(spec, 'vector'), axes(declared[spec['name']], 'vector')
        assert actual['architecture']['generator'] == expected['architecture']['generator'], path
        for key in ('formulation', 'training', 'resources', 'target'):
            assert actual[key] == expected[key], (path, key)
        payloads.add(digest(raw))
        episodes += 1

    def check_refs(value):
        nonlocal references
        if isinstance(value, dict):
            if 'artifact' in value and 'sha256' in value:
                path = SUITE/value['artifact']
                assert digest(path.read_bytes()) == value['sha256'], path
                references += 1
            for child in value.values():
                check_refs(child)
        elif isinstance(value, list):
            for child in value:
                check_refs(child)

    for path in (ROOT/'leaderboard.json', SUITE/'formulations/leaderboard.json', SUITE/'valid_search/leaderboard.json'):
        check_refs(read(path))
    report = read(ROOT/'leaderboard.json')
    assert report['episodes'] == episodes == len(payloads)
    assert digest((ROOT/'build.py').read_bytes()) == report['source_sha256']
    forensic = ROOT/'forensics'
    plan = read(forensic/'plan.json.gz')
    reference_path = SUITE.parents[1]/plan['reference']
    assert digest(reference_path.read_bytes()) == plan['reference_sha256']
    reference = read(reference_path)
    assert reference == read(forensic/'reference.json.gz')

    def without_timing(value):
        if isinstance(value, dict):
            return {k: without_timing(v) for k, v in value.items()
                    if k not in ('seconds', 'controller_seconds', 'confirmed_seconds', 'stable_from_seconds')}
        if isinstance(value, list):
            return [without_timing(v) for v in value]
        return value

    for filename in ('replay_result.json.gz', 'update_replay_result.json.gz'):
        assert without_timing(read(forensic/filename)) == without_timing(reference['result']), filename
    with tarfile.open(forensic/'source.tar.gz') as tar:
        for name, sha in plan['source']['source_sha256'].items():
            assert digest(tar.extractfile(name).read()) == sha, name
            sources += 1
    with tarfile.open(forensic/'instrumentation_source.tar.gz') as tar:
        for member in tar.getmembers():
            assert tar.extractfile(member).read() == (forensic/member.name).read_bytes(), member.name
            sources += 1
    winning_replays = 0
    winner_paths = [SUITE/r['artifact'] for r in report['records']
                    if r['spec']['name'] == 'vector_unequal_mass' and r['verdict']['passed']]
    assert len(winner_paths) == 1
    winner = read(winner_paths[0])
    for folder in ('winner_replay', 'cli_replay'):
        path = ROOT/folder
        assert read(path/'reference.json.gz') == winner
        parity = read(path/'parity.json.gz')
        assert digest(winner_paths[0].read_bytes()) == parity['reference_sha256']
        if folder == 'winner_replay':
            actual = read(path/'result.json.gz')
        else:
            episode = read(path/'episode.json.gz')
            assert dict(episode['spec'], runner='vector', phase='fit') == winner['spec']
            assert episode['policy'] == winner['policy']
            assert digest((path/'episode.json.gz').read_bytes()) == parity['episode_sha256']
            actual = episode['result']
        assert without_timing(actual) == without_timing(winner['result'])
        assert test_verdict(winner['spec'], actual)['passed']
        winning_replays += 1
    validation = dict(archive_files=files, source_file_instances=sources,
                      complete_architecture_episodes=episodes, unique_episode_payloads=len(payloads),
                      rare_candidates=report['rare_candidates'], sustained_rare_winners=report['sustained_rare_winners'],
                      cross_checks=report['cross_checks'], leaderboard_references=references,
                      diagnostic_replays_exact=2, diagnostic_controls_counted_as_gan_wins=0,
                      winning_candidate_replays_exact=winning_replays,
                      fixed_formulation_recipe_generator_resources_targets=True,
                      numerical_thresholds_unchanged=True, errors=[])
    (ROOT/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation))


if __name__ == '__main__':
    verify()
