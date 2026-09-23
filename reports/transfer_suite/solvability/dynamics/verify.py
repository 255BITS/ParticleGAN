"""Verify retained evidence without importing torch or rerunning training."""
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
sha = lambda value: hashlib.sha256(value).hexdigest()
inventory = json.loads((ROOT / 'inventory.json').read_text())
gzip_count = 0
for item in inventory['files']:
    data = (ROOT / item['path']).read_bytes()
    assert len(data) == item['bytes'], item['path']
    assert sha(data) == item['sha256'], item['path']
    if item['path'].endswith('.gz'):
        original = gzip.decompress(data)
        assert sha(original) == item['original_sha256'], item['path']
        assert len(original) == item['original_bytes'], item['path']
        gzip_count += 1
manifest = json.loads((ROOT / 'source_manifest.json').read_text())
with tarfile.open(fileobj=io.BytesIO((ROOT / 'source.tar.gz').read_bytes()), mode='r:gz') as archive:
    members = archive.getmembers()
    assert len(members) == len(manifest)
    assert {member.name for member in members} == set(manifest)
    for member in members:
        assert member.isfile(), member.name
        assert sha(archive.extractfile(member).read()) == manifest[member.name], member.name
raw_bytes = gzip.decompress((ROOT / 'results.json.gz').read_bytes())
raw = json.loads(raw_bytes)
export = json.loads(gzip.decompress((ROOT / 'episodes.json.gz').read_bytes()))
assert export['raw_results_sha256'] == sha(raw_bytes)
assert raw['protocol']['source_sha256'] == manifest == export['protocol']['source_sha256']
assert len(raw['rows']) == len(export['episodes']) == inventory['episodes'] == 66
assert export['attempts'] == 66
assert raw['source_commit'] == export['source_commit'] == inventory['source_commit']
originals = {spec['name']: spec for spec in raw['original_tasks']}


def passes(metrics, bounds):
    for key, op, bound in bounds:
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return False
        assert op in ('<=', '>=')
        if op == '<=' and value > bound or op == '>=' and value < bound:
            return False
    return True


observations = 0
witnesses = set()
for index, (original, row) in enumerate(zip(raw['rows'], export['episodes'])):
    assert row['episode_index'] == index
    assert original['card'] == row['candidate']
    assert original['spec'] == row['spec']
    assert original['result'] == row['result']
    assert row['original_spec'] == originals[original['task']]
    bounds = row['spec']['thresholds']
    assert bounds == row['original_spec']['thresholds']
    curve = row['result']['observations']
    assert len(curve) == 24
    assert [point['step'] for point in curve] == [math.ceil(i * row['spec']['steps'] / 24) for i in range(1, 25)]
    observations += len(curve)
    suffix = 0
    for point in reversed(curve):
        if not passes(point, bounds):
            break
        suffix += 1
    final = passes(row['result']['live'], bounds)
    ema = passes(row['result']['ema'], bounds)
    verdict = row['verdict']
    assert final == verdict['final_pass']
    assert ema == verdict['ema_pass']
    assert suffix == verdict['convergence']['passing_suffix']
    assert (final and suffix >= 5) == verdict['sustained_pass']
    if verdict['sustained_pass']:
        witnesses.add(row['original_spec']['name'])
    if row['original_spec']['reg_arm'] == 'a_r1r2':
        assert row['spec']['reg_arm'] == 'a_r1r2'
        assert row['spec']['reg_coeff'] == row['original_spec']['reg_coeff']
    assert row['spec']['d_lr_mult'] == row['original_spec']['d_lr_mult']
assert observations == inventory['live_observations'] == 1584
ranking = {spec['name'] for spec in raw['original_tasks'] if spec['split'] == 'development' and spec['tier'] == 'ranking'}
assert len(ranking) == 6 and ranking <= witnesses
last = export['episodes'][-1]
assert last['candidate']['name'] == 'slow_g_every2_budget5'
assert last['result']['update_counts'] == {'d': 6000, 'g': 3000}
assert last['verdict']['convergence']['passing_suffix'] == 11
assert last['spec']['d_lr_mult'] == .75
for name, control in raw['target_positive_controls'].items():
    assert control['pass'] and passes(control['metrics'], originals[name]['thresholds'])
assert len(raw['target_positive_controls']) == 7
wall = sum(row['result']['seconds'] for row in raw['rows'])
assert math.isclose(wall, inventory['recorded_wall_seconds'], abs_tol=1e-9)
result = {'verified': True, 'episodes': 66, 'live_observations': observations,
          'gzip_roundtrips': gzip_count, 'source_members': len(manifest),
          'trained_ranking_witnesses': sorted(ranking), 'scoring_positive_controls': 7,
          'recorded_wall_seconds': wall, 'results_sha256': sha(raw_bytes)}
(ROOT / 'verification.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps(result, indent=2))
