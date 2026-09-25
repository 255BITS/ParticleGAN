from pathlib import Path
from collections import Counter
import gzip, hashlib, json, shutil, sys

PUB = Path('/ml2/hypergan/ParticleGAN-epsilon-gan-followup')
BATCH = Path('/ml2/hypergan/gan-attempts/formulations-20260925T005324Z')
RUN = BATCH / 'adversarial_mobility/20260925T005324Z-2154605'
SRC = RUN / 'repo/reports/toy100/adversarial-mobility-attempt'
C = SRC / 'candidates/direct_particle_response'
OUT = PUB / 'reports/toy100/direct-particle-base'
OUT.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
decl = json.loads((C / 'declaration.json').read_text())
for name, h in decl['file_hashes'].items():
    assert sha(C / name) == h
    shutil.copy2(C / name, OUT / name)
shutil.copy2(C / 'declaration.json', OUT / 'original-declaration.json')
manifest_path = SRC / 'prepared/prepared-sources.json'
assert sha(manifest_path) == decl['source_manifest_sha256']
manifest = json.loads(manifest_path.read_text())
for name, h in manifest['cuda'].items():
    assert sha(SRC / 'prepared/repos/cuda' / name) == h, name
shutil.copy2(manifest_path, OUT / 'prepared-sources.json')
sys.path.insert(0, str(SRC / 'prepared/repos/cuda'))
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from benchmarks.transfer_suite.public_default_verification import load_declaration, declared_spec
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.locked_shared.hosts.unipolar import SCALES
from benchmarks.locked_shared.hosts.mid_scale_identity import EVAL_SCALES

config = json.loads((OUT / 'config.json').read_text()) | {'device': 'cuda:0'}
recipe, _, _ = declared_recipe(config)
jobs, profile = load_declaration()
rows = []
for p in sorted(C.glob('*/result.json')):
    r = json.loads(p.read_text())
    task = r['task']
    expected, _, _ = declared_spec(next(j for j in jobs if j['spec']['name'] == task), profile, recipe)
    assert r['spec'] == expected and r['config'] == config
    assert test_verdict(expected, r['result']) == r['verdict']
    assert r['status'] == r['verdict']['status']
    assert r['worker_sha256'] == decl['file_hashes']['probe.py']
    fixture = next(root / 'initialization-fixtures' / task / 'initial-values.pt'
                   for root in (PUB / 'reports/toy100/dimension-rms-base',
                                PUB / 'reports/toy100/cpu-recipe-gpu-port', SRC)
                   if (root / 'initialization-fixtures' / task / 'initial-values.pt').exists())
    assert sha(fixture) == r['initialization_fixture_sha256']
    fp = fixture.with_name('result.json.gz')
    init_bytes = gzip.decompress(fp.read_bytes()) if fp.exists() else fixture.with_name('result.json').read_bytes()
    init = json.loads(init_bytes)
    assert init['proof']['adam_calls'] == 0
    assert init['proof']['initial_optimizers'] == r['proof']['initial_optimizers']
    dest = OUT / 'initialization-fixtures' / task
    dest.mkdir(parents=True)
    shutil.copy2(fixture, dest / 'initial-values.pt')
    (dest / 'result.json.gz').write_bytes(gzip.compress(init_bytes, mtime=0))
    assert r['proof']['adam_calls'] == 2 * expected['steps']
    assert all(o['calls'] == expected['steps'] and o['device'] == 'cuda:0'
               and o['state_devices'] == ['cuda:0'] and o['parameter_dtypes'] == ['torch.float32']
               for o in r['proof']['optimizers'].values())
    scale_count = len(SCALES) if task == 'unipolar' else len(EVAL_SCALES) if task == 'mid_scale_identity' else 1
    assert r['regularizer_receipt']['calls'] == expected['steps'] * scale_count
    assert r['regularizer_receipt']['extra_critic_forwards'] == 0
    assert r['response_receipt']['calls'] == (expected['steps'] if task == 'two_pole' else 0)
    assert r['response_receipt']['history_devices'] == (['cuda:0'] if task == 'two_pole' else [])
    refpath = PUB / 'reports/toy100/dimension-rms-base/results' / (task + '.json.gz')
    compared = refpath.exists()
    if compared:
        ref = json.loads(gzip.decompress(refpath.read_bytes()))
        assert r['result']['actions'] == ref['result']['actions']
        assert r['randomness'] == ref['randomness']
    dest = OUT / 'results' / (task + '.json.gz')
    dest.parent.mkdir(exist_ok=True)
    dest.write_bytes(gzip.compress(p.read_bytes(), mtime=0))
    shutil.copy2(p.parent / 'audit.json', dest.with_suffix('').with_suffix('.audit.json'))
    rows.append(dict(gate=task, status=r['status'], metrics=r['result']['live'],
                     passing_suffix=r['verdict']['convergence']['passing_suffix'], steps=expected['steps'],
                     adam_calls=r['proof']['adam_calls'], regularizer_calls=r['regularizer_receipt']['calls'],
                     response_calls=r['response_receipt']['calls'], parent_rng_actions_compared=compared,
                     artifact=str(dest.relative_to(OUT))))
assert len(rows) == 16 and Counter(r['status'] for r in rows) == {'PASS': 15, 'FAIL': 1}
for name in ('check_direct.py', 'check_response.py', 'direct-mechanism-check.json', 'mechanism-check.json',
             'capture_initialization.py', 'fixture-captures.jsonl', 'final-audit.json', 'commands.jsonl'):
    shutil.copy2(SRC / name, OUT / name)
shutil.copy2(C / 'unipolar/audit-before-count-correction.json', OUT / 'unipolar-original-audit-error.json')
shutil.copy2(RUN / 'result.md', OUT / 'original-attempt-report.md')
shutil.copy2(RUN / 'tests.jsonl', OUT / 'original-tests.jsonl')
shutil.copy2(Path(__file__), OUT / 'promotion-review.py')
(OUT / 'audit.json').write_text(json.dumps(dict(status='PASS', candidate='direct_particle_response',
    source_files=len(manifest['cuda']), executed_gates=16, passes=15, failures=1, not_run=6,
    cuda_adam_updates=sum(r['adam_calls'] for r in rows),
    checks=['exact candidate/config/source hashes', 'canonical frozen specs and sustained verdicts',
            'zero-update CPU initialization fixtures', 'CUDA FP32 model and Adam state',
            'original optimizer and host-specific regularizer counts', 'direct-particle structural scope',
            'parent RNG and scheduled actions for seven retained controls'], rows=rows), indent=2) + '\n')
round_rows = []
for lane in json.loads((BATCH / 'batch.json').read_text()):
    run = sorted(Path(lane['directory']).glob('20*'))[-1]
    unique = {}
    for line in (run / 'tests.jsonl').read_text().splitlines():
        x = json.loads(line)
        if x['candidate'] == 'regression' or x['status'] == 'SKIPPED':
            continue
        artifact = Path(x.get('artifact', ''))
        if artifact.is_file() and artifact.name == 'result.json':
            unique[str(artifact)] = x
    for artifact, x in unique.items():
        r = json.loads(Path(artifact).read_text())
        assert r['verdict'] == test_verdict(r['spec'], r['result'])
        round_rows.append(dict(lane=lane['lane'], candidate=x['candidate'], gate=x['gate'], status=r['status'],
                               metrics=r['result']['live'], passing_suffix=r['verdict']['convergence']['passing_suffix'],
                               original_artifact=artifact))
assert len(round_rows) == 34
(OUT / 'round-summary.json').write_text(json.dumps(dict(proposals=9, executed_training_gates=34,
    counts=dict(Counter(r['status'] for r in round_rows)), rows=round_rows), indent=2) + '\n')
print(json.dumps(dict(published=str(OUT), selected=dict(Counter(r['status'] for r in rows)),
                      previous_round=dict(Counter(r['status'] for r in round_rows)))))
