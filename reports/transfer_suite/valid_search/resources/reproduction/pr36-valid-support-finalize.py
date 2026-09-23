import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from benchmarks.transfer_suite.protocol import test_verdict

STAGES=['screen','validation','combined','combined-validation','beta999','architecture']
repo=Path.cwd()
output=Path('/tmp/pr36-valid-support-artifacts')
output.mkdir(exist_ok=False)
manifest=[]; records=[]

def preserve(path,destination):
 payload=path.read_bytes(); raw=None
 if path.suffix=='.json':
  raw=payload;destination=destination.with_suffix('.json.gz');payload=gzip.compress(raw,mtime=0)
 elif path.name.endswith('.json.gz'):raw=gzip.decompress(payload)
 target=output/destination;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
 record=dict(path=str(destination),sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
 if raw is not None:
  record.update(original_sha256=hashlib.sha256(raw).hexdigest(),original_bytes=len(raw))
  assert gzip.decompress(payload)==raw
 manifest.append(record)

for stage in STAGES:
 directory=Path('/tmp/pr36-valid-support-'+stage)
 index=json.loads((directory/'index.json').read_text())
 plan=json.loads((directory/'plan.json').read_text())
 expected=sum(len(c.get('tasks',plan['tasks'])) for c in plan['candidates'])
 assert len(index['records'])==expected,(stage,len(index['records']),expected)
 protocol=json.loads((directory/'protocol.json').read_text())
 with tarfile.open(directory/'source.tar.gz','r:gz') as tar:
  for name,digest in protocol['source_sha256'].items():
   assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==digest
 for record in index['records']:
  payload=gzip.decompress((directory/record['artifact']).read_bytes())
  assert hashlib.sha256(payload).hexdigest()==record['uncompressed_sha256']
  episode=json.loads(payload); spec=record['spec']; original=record['original_spec']; result=episode['result']
  assert spec['thresholds']==original['thresholds'] and spec['steps']==original['steps']
  assert all(spec.get(k,default)==wanted for k,default,wanted in [
   ('loss_type','logistic','logistic'),('gan_mode','rp','rp'),('reg_arm',None,'b_cap'),
   ('reg_coeff',None,3.),('reg_kappa',None,1.25),('prior_reg',None,.05),('d_every',None,1),('g_every',None,1)])
  assert all(spec.get(k)==original.get(k) for k in ['kind','means','covariances','masses','identifiable','noise','turns','radius_min','radius_max','hidden','layers','z_dim'])
  if stage in ('screen','validation'):
   assert set(record['candidate']['overrides'])=={'particles','batch'}
  verdict=test_verdict(spec,result)
  assert verdict==record['verdict']
  assert len(result['observations'])==24 and verdict['convergence']['complete'] and not result.get('error')
  records.append({**record, 'stage':stage, 'artifact':stage+'/'+record['artifact']})
 for path in sorted(directory.rglob('*')):
  if path.is_file():preserve(path,Path(stage)/path.relative_to(directory))
 log=directory.with_suffix('.log')
 preserve(log,Path('logs')/log.name)
 assert hashlib.sha256((directory/'plan.json').read_bytes()).hexdigest()

assert len(records)==57
pure=json.loads(Path('/tmp/pr36-valid-support-report/index.json').read_text())
for path in sorted(Path('/tmp/pr36-valid-support-report/reused').glob('*.json.gz')):
 preserve(path,Path('reused')/path.name)
assert len(list((output/'reused').glob('*.json.gz')))==9
for name,path in [('baseline-source.tar.gz',repo/'reports/transfer_suite/study/source.tar.gz'),
                  ('p1024-source.tar.gz',repo/'reports/transfer_suite/solvability/vectors/screen/source.tar.gz')]:
 preserve(path,Path('reused')/name)
for path in [Path('/tmp/pr36-valid-support-report.py'),Path(__file__)]:preserve(path,Path('reproduction')/path.name)

hard=['vector_unequal_mass','vector_unequal_width','vector_overlap']
all_tasks=['vector_two_broad','vector_unequal_mass','vector_unequal_width','vector_anisotropic','vector_overlap','vector_spiral']
follow={}
for record in records:
 if record['stage'] in ('screen','validation'):continue
 follow.setdefault(record['candidate']['name'],{})[record['spec']['name']]=record
full=dict((r['spec']['name'],r) for r in pure['baseline'])

lines=['# Fixed-formulation support and resource search','',
       '**The best pure resource change passes4/6 valid data tasks versus3/6 for the original setup:** '
       '512 particles with batch128 fixes unequal-width mixtures while preserving broad, anisotropic and spiral cases. '
       'It does not solve the unequal-mass or sustained-overlap failures. No shared all-pass data configuration was found.', '',
       'The original resource grid is12 shared cards: particle counts512/1024/2048/4096 crossed with batches128/256/512. '
       'Only particles and batch change in that grid. All loss terms remain Rp logistic, b_cap coefficient3, kappa1.25, '
       'prior regularization0.05, no particle L2. The G/D networks, original LRs, Adam(0,.99), prior LR and budgets '
       'remain unchanged. Gaussian tasks keep1200 steps; spiral keeps its original1600. '
       'Evaluation uses live weights, all24 observations and at least five final passing checks; EMA is separate.', '',
       'There are57 newly executed complete episodes and9 byte-identical reused episodes (six original baselines and '
       'three prior1024/128 hard-task runs). Reuse was accepted only after all relevant numerical source hashes matched. '
       'Targets and thresholds never changed; there are no seed sweeps, image reruns or imposed dynamics tests.', '',
       '## Pure particle/batch results','']
resource_md=Path('/tmp/pr36-valid-support-report/README.md').read_text()
lines.append(resource_md[resource_md.index('| Shared particles / batch'):resource_md.index('Every failure remains')].strip())
lines+=['','Covariance columns report the **mean of component-relative covariance errors** for the named dataset, '
        'not only the rare component. Some failures are caused by outliers in a more common component. '
        'No resource card sustains the unequal-mass task; the closest final mean covariance error is0.9129 '
        '(512/256) against the unchanged≤0.85 gate. More particles are not a monotonic fix.', '',
        '## Separate optimizer and architecture combinations','',
        'After the resource grid was frozen, parent-requested combinations were declared as separate stages. '
        'The coordinated recipe uses G LR0.00075, D LR0.0015, prior LR0.0225 and Adam(0,.999). '
        'The plain-beta stage changes only beta2 to.999 plus the stated resource count/batch. '
        'The architecture stage keeps original Adam/LRs and uses D width128, three layers and four Fourier bands. '
        'The adversarial/regularization formulation and all gates remain fixed in every stage.', '',
        '| Shared combination | Hard sustained /3 | Full-data sustained /6 | Unequal mass | Unequal width | Overlap |',
        '| --- | ---: | --- | --- | --- | --- |']
def cell(record):
 if record is None:return '—'
 v=record['verdict'];return f"{v['status']} (tail {v['convergence']['passing_suffix']})"
for name,group in follow.items():
 passed=sum(group[t]['verdict']['passed'] for t in hard)
 count=f"{sum(r['verdict']['passed'] for r in group.values())}/6" if len(group)==6 else 'not promoted'
 lines.append(f'| {name} | {passed}/3 | {count} | '+' | '.join(cell(group.get(t)) for t in hard)+' |')
lines+=['', 'The coordinated512/128 card fixes unequal width and overlap but fails unequal mass and anisotropic, '
        'so it reaches4/6 rather than improving the full-data count. Its unequal-mass mean covariance error is3.3056: '
        'the rare component error is only0.5134, while the13% component error is10.2514. '
        'The plain-beta combinations add no hard-task pass beyond overlap and were not promoted. '
        'The final D/resource combination also passes only overlap; unequal-width mean covariance error reaches33.016. '
        'It fails the predeclared≥2/3 promotion rule. The favorable effects of separate changes do not add reliably.', '',
        '## Cost, validation and reproduction','',
        f'New episode wall time totals{sum(r["seconds"] for r in records):.1f}s, with zero numerical errors. '
        'These are single CPU observations under shared load, not replicated speed comparisons. '
        'The particle sweep uses2×–16× the original trainable support rows; batch256/512 processes2×/4× as many samples '
        'per update at the same update budget. Validation retains each original task budget.', '',
        'All57 new episode hashes and all source bundles were checked; each recorded verdict was recomputed from '
        'its complete live curve and unchanged thresholds. No repository code was edited: the existing '
        '`benchmarks.transfer_suite.solvability_search` runner at commit '
        '`a11c5304cde01c7fdc96e8a49a5a576b8cb8ebff` performed every new run. '
        'No new behavioral tests were necessary for this data-only research run.', '',
        'Reproduce any stage by extracting its plan and using a fresh output directory:', '',
        '```bash','python -u -m benchmarks.transfer_suite.solvability_search \\',
        '  --plan plan.json --output /tmp/new-valid-support-run','```','',
        'Every stage directory contains its exact plan, runtime/source protocol, source.tar.gz, per-episode '
        'curves/actions/live/EMA metrics and index. JSON payloads are gzip-compressed with original-byte '
        'SHA256s in archive_manifest.json. Original failures remain included. Reproduction scripts and logs '
        'are retained. All cases are inspected development data; no universal or production default is established.', '']
text='\n'.join(lines)
for a,b in [('passes4/6','passes 4/6'),('versus3/6','versus 3/6'),('batch128','batch 128'),('is12','is 12'),('counts512','counts 512'),('batches128','batches 128'),('coefficient3','coefficient 3'),('kappa1.25','kappa 1.25'),('regularization0.05','regularization 0.05'),('Adam(0,.99)','Adam (0,.99)'),('keep1200','keep 1200'),('original1600','original 1600'),('all24','all 24'),('are57','are 57'),('and9','and 9'),('prior1024','prior 1024'),('is0.9129','is 0.9129'),('unchanged≤','unchanged ≤'),('LR0.00075','LR 0.00075'),('LR0.0015','LR 0.0015'),('LR0.0225','LR 0.0225'),('Adam(0,.999)','Adam (0,.999)'),('to.999','to .999'),('width128','width 128'),('coordinated512','coordinated 512'),('reaches4/6','reaches 4/6'),('is3.3056','is 3.3056'),('only0.5134','only 0.5134'),('the13%','the 13%'),('is10.2514','is 10.2514'),('reaches33.016','reaches 33.016'),('predeclared≥','predeclared ≥'),('totals','totals '),('uses2×','uses 2×'),('batch256/512','batch 256/512'),('processes2×/4×','processes 2×/4×'),('All57','All 57')]:text=text.replace(a,b)
(output/'README.md').write_text(text)
summary=dict(new_records=records,reused_resource_records=[r for r in pure['records'] if r.get('source','').startswith('reused')],
             baseline= pure['baseline'],new_episode_count=57,reused_episode_count=9,new_seconds=sum(r['seconds'] for r in records),
             source_revision='a11c5304cde01c7fdc96e8a49a5a576b8cb8ebff',production_changes=False)
raw=(json.dumps(summary,indent=2)+'\n').encode();(output/'summary.json.gz').write_bytes(gzip.compress(raw,mtime=0))
manifest.append(dict(path='summary.json.gz',sha256=hashlib.sha256((output/'summary.json.gz').read_bytes()).hexdigest(),original_sha256=hashlib.sha256(raw).hexdigest()))
(output/'archive_manifest.json').write_text(json.dumps(dict(files=manifest,validation=dict(new_complete_episodes=57,reused_exact_episodes=9,errors=0,source_bundles_verified=len(STAGES),verdicts_recomputed=57)),indent=2))
print('Verified and archived',len(manifest),'files at',output)
