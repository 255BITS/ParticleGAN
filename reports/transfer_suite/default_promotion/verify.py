"""Audit default promotion without retraining or modifying historical evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from particlegan import get_recipe
from benchmarks.transfer_suite.protocol import test_verdict

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify():
    replay = ROOT/'replay'
    manifest = read(replay/'archive_manifest.json')
    for item in manifest['files']:
        raw = (replay/item['path']).read_bytes()
        assert digest(raw) == item['archived_sha256']
        if item['gzip_added']:
            assert digest(gzip.decompress(raw)) == item['original_sha256']
    protocol = read(replay/'protocol.json.gz')
    with tarfile.open(replay/'source.tar.gz') as archive:
        for name, sha in protocol['source_sha256'].items():
            assert digest(archive.extractfile(name).read()) == sha, name
    parity = read(replay/'parity.json.gz')
    reference_path = REPO/parity['reference']
    assert digest(reference_path.read_bytes()) == parity['reference_sha256']
    reference = read(reference_path)
    result = read(replay/'result.json.gz')
    assert len(result['observations']) == len(reference['result']['observations']) == 24
    for actual, expected in zip(result['observations'], reference['result']['observations']):
        assert {k: v for k, v in actual.items() if k != 'seconds'} == {
            k: v for k, v in expected.items() if k != 'seconds'}
    assert result['actions'] == reference['result']['actions']
    assert result['live'] == reference['result']['live'] and result['ema'] == reference['result']['ema']
    verdict = test_verdict(reference['spec'], result)
    assert verdict == read(replay/'verdict.json.gz')
    assert verdict['convergence']['complete'] and verdict['convergence']['passing_suffix'] == 6 and verdict['passed']
    recipe = get_recipe(num_particles=256, batch_size=128, total_steps=1200)
    assert json.loads(json.dumps(recipe.to_dict())) == read(replay/'recipe.json.gz')
    before, after = read(ROOT/'previous-recipes.json'), read(ROOT/'current-recipes.json')
    for name, value in before.items():
        if name not in ('gan', '100gaussians'):
            assert after[name] == value, name
        assert after[name] == json.loads(json.dumps(get_recipe(name).to_dict())), name
    assert dict(after['gan_legacy'], name='gan') == before['gan']
    wheel_audit = read(ROOT/'wheel-audit.json')
    assert wheel_audit['default_recipe'] == after['gan']
    assert wheel_audit['resume_training_state_exact']
    assert wheel_audit['completed_steps'] == 4
    assert '193 passed' in (ROOT/'tests.log').read_text()
    assert '1 passed' in (ROOT/'history-test.log').read_text()
    validation = dict(archive_files=len(manifest['files']), numerical_source_files=len(protocol['source_sha256']),
                      public_default_recipe_trainer_critic_exact=True, live_passing_suffix=6,
                      live_confirmation_step=verdict['convergence']['confirmed_step'],
                      named_domain_recipes_unchanged=True, legacy_recipe_exact=True,
                      focused_tests=194, installed_wheel_resume_training_state_exact=True,
                      errors=[])
    (ROOT/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation))


if __name__ == '__main__':
    verify()
