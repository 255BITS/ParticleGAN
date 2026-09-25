"""Package and verify the completed, immutable three-stage architecture study."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from benchmarks.transfer_suite.compare_defaults import effective_spec, ema_verdict
from benchmarks.transfer_suite.shared_discriminator_search import recipe
from benchmarks.transfer_suite.shared_variants import architecture_spec
from benchmarks.transfer_suite.protocol import test_verdict

source=Path('/tmp/pr38-shared-discriminator')
out=Path('/tmp/pr38-shared-discriminator-handoff')
repo=Path('/ml2/hypergan/ParticleGAN-shared-discriminator')
if out.exists():raise FileExistsError(out)
out.mkdir()
records=[]
source_members=0
for stage in ['screen','cross','completion']:
 shutil.copytree(source/stage,out/stage)
 shutil.copy2(source/(stage+'.log'),out/(stage+'.log'))
 protocol=json.loads((out/stage/'protocol.json').read_text())
 with tarfile.open(out/stage/'source.tar.gz') as tar:
  contents={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
  assert set(contents)==set(protocol['source_sha256'])
  for name,sha in protocol['source_sha256'].items():
   assert hashlib.sha256(contents[name]).hexdigest()==sha
  source_members=len(contents)
 for row in json.loads((out/stage/'index.json').read_text())['records']:
  raw=gzip.decompress((out/stage/row['artifact']).read_bytes())
  assert hashlib.sha256(raw).hexdigest()==row['uncompressed_sha256']
  payload=json.loads(raw)
  assert payload['source_sha256']==protocol['source_sha256']
  assert payload['spec']==effective_spec(architecture_spec(payload['original_spec'],payload['discriminator_variant']),recipe())
  assert payload['recipe']==json.loads(json.dumps(recipe().to_dict()))
  assert payload['verdict']==test_verdict(payload['spec'],payload['result'])
  assert payload['ema_verdict']==ema_verdict(payload['spec'],payload['result'])
  assert len(payload['result']['observations'])==24
  receipts={r['role']:(r['lr'],r['betas']) for r in payload['applied']}
  assert receipts=={'g':(.00425,[0.,.99]),'d':(.00425,[0.,.99]),'prior':(.0085,[0.,.99])}
  row=deepcopy(row);row['artifact']=stage+'/'+row['artifact'];records.append(row)
assert len(records)==28
(out/'index.json').write_text(json.dumps(dict(records=records),indent=2)+'\n')
for name in ['reproduce.py','reproduce-v1.py','seal.py','overlap-replay.log','overlap-replay-timing-comparison.log']:
 shutil.copy2(source/name,out/name)
raw=(source/'overlap-replay.json').read_bytes()
(out/'overlap-replay.json.gz').write_bytes(gzip.compress(raw,mtime=0))
(out/'tests').mkdir()
shutil.copy2(repo/'tests/test_shared_critic_research.py',out/'tests/test_shared_critic_research.py')
shutil.copy2(repo/'benchmarks/transfer_suite/SHARED_DISCRIMINATOR_RESEARCH.md',out/'PROTOCOL.md')
names=list(dict.fromkeys(r['architecture'] for r in records))
tasks=['vector_two_broad','vector_unequal_mass','vector_unequal_width','vector_anisotropic','vector_overlap','vector_spiral']
lines=['# Shared cap6 architecture results','',
'**Two additional task witnesses; rare mass and unequal width remain unresolved.** '
'One unchanged formulation and optimizer recipe throughout; architecture selection per case is explicit. '
'The existing reference profile is 15/19. Combining its retained successes with the new anisotropic and overlap '
'architecture witnesses supports 17/19, pending the primary importer. This bundle alone is a six-data study, '
'not a fresh run of all19 or a single-discriminator universal pass.','',
'28 search episodes, 672 live observations, '+f"{sum(r['seconds'] for r in records):.3f} total single-CPU episode seconds. "
'One additional exact overlap replay verifies portability. All use seed0; there are no seed sweeps, resource changes, '
'target-derived features or objective additions. Five search episodes pass; all23 failures remain.','',
'Live PASS requires 24 recorded checkpoints and at least5 passing checkpoints at the end. '
'Cells show live status and final passing suffix; missing cases stay untested.','',
'| Architecture | D parameters | '+ ' | '.join(t.removeprefix('vector_') for t in tasks)+' |',
'| --- | ---: | '+' | '.join('---' for _ in tasks)+' |']
for name in names:
 rows=[r for r in records if r['architecture']==name]
 count=next(a['parameters'] for a in rows[0]['applied'] if a['role']=='d')
 cells=[]
 for task in tasks:
  row=next((r for r in rows if r['spec']['name']==task),None)
  cells.append('not tested' if row is None else f"[{row['verdict']['status']} {row['verdict']['convergence']['passing_suffix']}/24]({row['artifact']})")
 lines.append(f'| {name} | {count} | '+' | '.join(cells)+' |')
lines+=['',
'The completed raw-SiLU128 profile passes only anisotropic (1/6); it loses broad and spiral. '
'The completed raw-Softplus96 profile passes overlap, broad and spiral (3/6). '
'Explicit per-case architecture support is therefore essential to the17/19 accounting. '
'Raw-SiLU128 width ends within all bounds but has only one passing checkpoint; its overlap ends within bounds '
'but has suffix4. Quadratic-Tanh anisotropic also has suffix4. All remain FAIL.','',
'Raw-SiLU128 anisotropic passes with HQ.98364, covariance error.30926 and minimum eigen ratio.36505 (suffix6). '
'The smaller additive raw/Fourier critic also passes anisotropic (5796D parameters, suffix8), '
'but its final minimum eigen ratio.16660 is closer to the.15 bound. '
'Raw-Softplus96 overlap passes with SW.10615, mean error.10610 and covariance error.32536 (suffix10).','',
'Rare-mode minimum variance still fails for every tested card, including nonperiodic critics. '
'These results do not establish periodic features as the sole cause under the higher shared learning rates. '
'No optimizer setting was selected separately by toy.','',
'EMA is recorded independently in every artifact. Raw-SiLU128 EMA passes anisotropic, overlap and spiral, '
'while its live model passes only anisotropic. Raw-Softplus96 live spiral passes but EMA fails. '
'EMA cannot rescue live failures.','',
'[Frozen methodology and card definitions](PROTOCOL.md) · [All episode metadata](index.json) · '
'[Stage1 matrix](screen/README.md) · [Stage2 matrix](cross/README.md) · [Stage3 matrix](completion/README.md) · '
'[SHA256 inventory](inventory.json)','',
'Reproduction from a checkout uses the module command in PROTOCOL.md and the retained stage plan.json files. '
'For a standalone exact replay, extract that stage source.tar.gz, install the versions recorded in protocol.json, '
'and run from outside a repository:', '',
'```sh','mkdir /tmp/shared-d-source',
'tar -xzf cross/source.tar.gz -C /tmp/shared-d-source',
'PYTHONPATH=/tmp/shared-d-source python reproduce.py \\',
'  cross/episodes/shared_c6__raw_softplus96_l3__vector_overlap.json.gz --output /tmp/overlap-replay.json',
'```','',
'The replay checks source hashes before training and exact numerical curves, actions and optimizer receipts '
'afterward (timings excluded). The first replay comparator incorrectly compared nested convergence timestamps; '
'its original script/log are retained. The corrected recursive comparator verified the already-retained replay '
'without another training run. The retained overlap-replay.json.gz/log show this was executed from an extracted '
'source bundle. Reference artifact paths are provenance labels; replay uses the canonical original_spec retained '
'in the episode and does not read historical report files.','',
'Code commit: b13c69491a3f3195b8d3f4a214dadc79ba00d0e5, based on78c872236b70f6e5527169db7143559536e052a1. '
'The parent-owned shared_variants.py is included byte-for-byte in source archives but excluded from that commit. '
'Focused tests:11 passed. Numerical sources were frozen before all28 episodes and unchanged throughout.','']
(out/'README.md').write_text('\n'.join(lines))
validation=dict(search_episodes=28,extra_exact_replays=1,live_observations=672,source_members_per_archive=source_members,
                source_manifests_verified=3,optimizer_receipts_verified=28,verdicts_recomputed=28,
                same_recipe=True,source_bytes_unchanged=True,seconds=sum(r['seconds'] for r in records))
(out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
inventory=[]
for path in sorted(out.rglob('*')):
 if not path.is_file():continue
 data=path.read_bytes();item=dict(path=str(path.relative_to(out)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
 if path.suffix=='.gz':
  raw=gzip.decompress(data)
  assert gzip.decompress(gzip.compress(raw,mtime=0))==raw
  item.update(uncompressed_bytes=len(raw),uncompressed_sha256=hashlib.sha256(raw).hexdigest())
 inventory.append(item)
(out/'inventory.json').write_text(json.dumps(dict(files=inventory),indent=2)+'\n')
print(json.dumps(validation))
print('inventory_sha256',hashlib.sha256((out/'inventory.json').read_bytes()).hexdigest())
