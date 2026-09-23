"""Verify and package an already-completed bounded architecture search."""
import gzip,hashlib,json,tarfile
from pathlib import Path
import sys
sys.path.insert(0,'/ml2/hypergan/ParticleGAN-pr36-valid-d')
import torch
from lib.toy_models import SimpleMLPDiscriminator
from benchmarks.transfer_suite.protocol import test_verdict
out=Path('/tmp/pr36-valid-data-d-architecture')
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
plan=json.loads((out/'plan.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
assert hashlib.sha256((out/'run.py').read_bytes()).hexdigest()==protocol['driver_sha256']
complete=json.loads((out/'completed.json').read_text());records=json.loads((out/'index.json').read_text())['records']
assert complete['episodes']==len(records)==45
assert complete['errors']==0 and complete['all_complete']
with tarfile.open(out/'source.tar.gz') as archive:
    for name,digest in protocol['source_sha256'].items():
        assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==digest
for row in records:
    raw=gzip.decompress((out/row['artifact']).read_bytes())
    assert hashlib.sha256(raw).hexdigest()==row['uncompressed_sha256']
    payload=json.loads(raw)
    assert payload['original_spec']['thresholds']==payload['spec']['thresholds']
    assert test_verdict(row['spec'],payload['result'])==row['verdict']
    changed={k for k in payload['spec'] if payload['spec'][k]!=payload['original_spec'].get(k)}
    assert changed<= {'d_hidden','d_layers','fourier'}
    assert row['spec']['steps']==payload['original_spec']['steps']
for row in json.loads((out/'references.json').read_text()):
    assert hashlib.sha256((out/row['artifact']).read_bytes()).hexdigest()==row['artifact_sha256']
rows=[]
for card in plan['cards']:
    group=[r for r in records if r['candidate']['name']==card['name']]
    changes=card['overrides'];torch.manual_seed(0)
    params=sum(p.numel() for p in SimpleMLPDiscriminator(2,changes['d_hidden'],changes['d_layers'],changes['fourier']).parameters())
    row={'name':card['name'],'overrides':changes,'d_parameters':params,'sustained':sum(r['verdict']['passed'] for r in group),'attempted':len(group),'complete_all_six':len(group)==6,'final_pass':sum(all(m['status']=='PASS' for m in r['verdict']['metrics']) for r in group),'seconds':sum(r['seconds'] for r in group),'tasks':{r['task']:{'status':r['verdict']['status'],'convergence':r['verdict']['convergence'],'live':r['live'],'artifact':r['artifact']} for r in group}}
    rows.append(row)
write(out/'leaderboard.json',{'scope':'D architecture only, one fixed formulation and task settings; no new seeds or longer training. Six valid data cases only.','rows':rows,'validation':complete})
with (out/'README.md').open('a') as doc:
    doc.write('\n## Completed validation\n\nAll45 new episodes completed24 observations without errors. Source/episode hashes, unchanged thresholds/settings/budgets, and every sustained verdict were independently rechecked.\n\n')
    doc.write('| Fully evaluated D architecture | D parameters | Sustained /6 | Final /6 | Episode seconds |\n|---|---:|---:|---:|---:|\n')
    for row in rows:
        if row['complete_all_six']:
            doc.write(f'| {row["name"]} | {row["d_parameters"]} | {row["sustained"]}/6 | {row["final_pass"]}/6 | {row["seconds"]:.2f} |\n')
    doc.write('\nSelection used only the three hard valid data toys, followed by full-six regression checks of the three best predeclared screen scores. Each row is one discriminator setting shared across tasks. Screening failures and incomplete candidates remain visible. This is architecture search on inspected development data, not a new formulation, a seed study, or held-out generalization evidence. Wall times include evaluations under concurrent machine load; they do not establish a speedup.\n\n[Machine leaderboard](leaderboard.json), [selected cards and screen scores](selected.json).\n')
index={}
for file in sorted(out.rglob('*')):
    if file.is_file() and file.name!='archive_manifest.json':
        index[str(file.relative_to(out))]={'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'bytes':file.stat().st_size}
write(out/'archive_manifest.json',index)
print(json.dumps([{'name':r['name'],'sustained':r['sustained'],'attempted':r['attempted'],'final_pass':r['final_pass']} for r in rows if r['complete_all_six']],indent=2))
