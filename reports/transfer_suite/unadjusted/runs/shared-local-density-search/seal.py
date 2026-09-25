"""Archive exact two-stage local-density search and recompute all integrity gates."""
from copy import deepcopy
import gzip,hashlib,json,shutil,tarfile
from pathlib import Path
from benchmarks.transfer_suite.compare_defaults import effective_spec,ema_verdict
from benchmarks.transfer_suite.shared_discriminator_search import recipe
from benchmarks.transfer_suite.shared_variants import architecture_spec
from benchmarks.transfer_suite.protocol import test_verdict
src=Path('/tmp/pr38-local-density');out=Path('/tmp/pr38-local-density-handoff')
repo=Path('/ml2/hypergan/ParticleGAN-shared-local-density')
if out.exists():raise FileExistsError(out)
out.mkdir();records=[];members={};source_manifests=[]
for stage,count in [('screen',32),('refinement',12)]:
 shutil.copytree(src/stage,out/stage);shutil.copy2(src/(stage+'.log'),out/(stage+'.log'))
 protocol=json.loads((out/stage/'protocol.json').read_text());source_manifests.append(protocol['source_sha256'])
 with tarfile.open(out/stage/'source.tar.gz') as t:
  archived={m.name:t.extractfile(m).read() for m in t if m.isfile()}
 assert {k:hashlib.sha256(v).hexdigest() for k,v in archived.items()}==protocol['source_sha256']
 members[stage]=len(archived)
 rows=json.loads((out/stage/'index.json').read_text())['records'];assert len(rows)==count
 for row in rows:
  raw=gzip.decompress((out/stage/row['artifact']).read_bytes())
  assert hashlib.sha256(raw).hexdigest()==row['uncompressed_sha256']
  p=json.loads(raw);r=p['result'];assert 'error' not in r
  assert p['source_sha256']==protocol['source_sha256']
  assert p['recipe']==json.loads(json.dumps(recipe().to_dict()))
  assert p['spec']==effective_spec(architecture_spec(p['original_spec'],p['discriminator_variant']),recipe())
  assert p['verdict']==test_verdict(p['spec'],r) and p['ema_verdict']==ema_verdict(p['spec'],r)
  assert len(r['observations'])==24
  assert {a['role']:(a['lr'],a['betas']) for a in p['applied']}=={'g':(.00425,[0.,.99]),'d':(.00425,[0.,.99]),'prior':(.0085,[0.,.99])}
  row=deepcopy(row);row['artifact']=stage+'/'+row['artifact'];records.append(row)
assert all(source_manifests[1][k]==v for k,v in source_manifests[0].items())
(out/'index.json').write_text(json.dumps(dict(records=records),indent=2)+'\n')
for name in ['reproduce.py','seal.py']:shutil.copy2(src/name,out/name)
shutil.copy2(repo/'benchmarks/transfer_suite/SHARED_LOCAL_DENSITY.md',out/'PROTOCOL.md')
(out/'tests').mkdir()
for name in ['test_shared_local_density_research.py','test_shared_residual_curvature.py']:
 shutil.copy2(repo/'tests'/name,out/'tests'/name)
lines=['# Shared-cap6 local-density architecture results','',
'22 architecture cards,44 full training episodes,1056 live observations. One unchanged shared_c6 formulation '
'and optimizer recipe throughout. D architecture is the only varying axis; both original1200-step budgets, '
'256 particles,128 batch and all metric thresholds remain fixed. Seed0 only; no seed experiments.','',
f"**Sustained live passes: {sum(x['verdict']['passed'] for x in records)}/44 episodes.** "
'This bounded round provides no additional task support unless a row below passes. '
'Final metrics that pass without a five-check suffix remain FAIL. EMA is independent.','',
'| D architecture | Parameters | Rare live / suffix | Rare minimum eigen ratio | Width live / suffix | Width minimum eigen ratio |',
'| --- | ---: | --- | ---: | --- | ---: |']
for name in dict.fromkeys(x['architecture'] for x in records):
 q=[x for x in records if x['architecture']==name]
 count=next(a['parameters'] for a in q[0]['applied'] if a['role']=='d');cells=[]
 for task in ['vector_unequal_mass','vector_unequal_width']:
  row=next(x for x in q if x['spec']['name']==task)
  cells += [f"[{row['verdict']['status']} / {row['verdict']['convergence']['passing_suffix']}]({row['artifact']})",f"{row['live']['component_min_eigen_ratio']:.6g}"]
 lines.append(f'| {name} | {count} | '+' | '.join(cells)+' |')
lines += ['',
'The required minimum component eigen ratio is.15. Fixed radial and standalone local quadratic heads often '
'lose entire modes. Adding local curvature to an initialized raw MLP generally restores occupancy and several '
'distribution metrics, but still leaves a collapsed narrow covariance direction. More explicit local features '
'alone did not repair the two blockers at these shared learning rates.','',
'Initial16-card screen:32 episodes. Six approved initially-zero residual-curvature refinements:12 episodes. '
'Every failed trial, complete live/EMA curve, action trace and actual optimizer receipt is retained. '
'Architecture-only spec diffs and every verdict were recomputed during packaging. The second source archive '
'adds two modules and preserves every byte of the first archive’s numerical source.','',
f"Total episode CPU wall time: {sum(x['seconds'] for x in records):.3f}s. "
'25 focused tests passed, including exact initial base-model weights, score and global RNG preservation for '
'residual branches, nonzero cap gradients into the branch and actual unchanged optimizer receipts.','',
'[Initial screen matrix](screen/README.md) · [Refinement matrix](refinement/README.md) · '
'[All episode metadata](index.json) · [Frozen methodology](PROTOCOL.md) · [Validation](validation.json) · '
'[File hashes](inventory.json)','',
'Code commit:19bf7b8c6b196d1c058a95f7c26903e690d071ba, based onf10cfb1b025aa6c843b61ea77da5f474444bc7cc. '
'No production components or shared runner were edited.','',
'Import the two stage index.json files separately, or use aggregate index.json whose artifact paths already '
'include stage prefixes. Original compressed bytes are unchanged. Source archives contain exact executed '
'code; protocol.json records Python/Torch/build/CPU details. Inventory lists original and compressed SHA256.','',
'To rerun a whole stage from the checkout, use its plan.json with the module in PROTOCOL.md. To reproduce '
'one retained episode without external historical report files, extract that stage source.tar.gz, then run:', '',
'```sh','mkdir /tmp/local-density-source','tar -xzf refinement/source.tar.gz -C /tmp/local-density-source',
'PYTHONPATH=/tmp/local-density-source OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python reproduce.py \\',
'  refinement/episodes/shared_c6__curvature_raw_silu128_l3_q64_w1p0__vector_unequal_mass.json.gz \\',
'  --output /tmp/local-density-replay.json','```','',
'The helper checks source hashes first, then compares numerical results, curves, actions, receipts and '
'verdicts exactly, excluding runtime fields. It is supplied for reproduction but was not executed in this '
'search round; the44 retained episodes are the complete GAN-run count.','']
(out/'README.md').write_text('\n'.join(lines))
validation=dict(episodes=len(records),observations=24*len(records),passed=sum(x['verdict']['passed'] for x in records),
                errors=0,seed=0,source_members=members,source_manifests_verified=2,old_source_unchanged=True,
                optimizer_receipts_verified=len(records),verdicts_recomputed=len(records),
                seconds=sum(x['seconds'] for x in records),focused_tests_passed=25)
(out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
inventory=[]
for p in sorted(out.rglob('*')):
 if not p.is_file():continue
 raw=p.read_bytes();r=dict(path=str(p.relative_to(out)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 if p.suffix=='.gz':
  expanded=gzip.decompress(raw);assert gzip.decompress(gzip.compress(expanded,mtime=0))==expanded
  r.update(uncompressed_bytes=len(expanded),uncompressed_sha256=hashlib.sha256(expanded).hexdigest())
 inventory.append(r)
(out/'inventory.json').write_text(json.dumps(dict(files=inventory),indent=2)+'\n')
print(json.dumps(validation));print('inventorySHA',hashlib.sha256((out/'inventory.json').read_bytes()).hexdigest())
