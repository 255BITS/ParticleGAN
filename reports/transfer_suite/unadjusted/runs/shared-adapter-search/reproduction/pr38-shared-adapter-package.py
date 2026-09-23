"""Verify and package every attempt without rewriting original experiment bytes."""
import gzip,hashlib,json,shutil,tarfile
from pathlib import Path
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.shared_adapter_search import clean,canonical
from benchmarks.transfer_suite.relative_step_adapter import mechanism
SRC=Path('/tmp/pr38-shared-adapter-v2');OUT=Path('/tmp/pr38-shared-adapter-artifacts')
read=lambda p:json.loads(gzip.decompress(p.read_bytes()) if p.name.endswith('.gz') else p.read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
index=read(SRC/'index.json');rows=index['records'];assert len(rows)==34
selection=read(SRC/'selection.json');selected=selection['selected']['name'];selected_rows=[r for r in rows if r['adapter_candidate']==selected];assert len(selected_rows)==19
OUT.mkdir(exist_ok=False)
manifest=[]
def copy_tree(source,destination):
 for p in sorted(source.rglob('*')):
  if not p.is_file():continue
  relative=p.relative_to(source);raw=p.read_bytes();added=p.suffix=='.json';q=destination/relative
  if added:q=q.with_suffix(q.suffix+'.gz')
  q.parent.mkdir(parents=True,exist_ok=True);payload=gzip.compress(raw,mtime=0) if added else raw;q.write_bytes(payload)
  manifest.append(dict(path=str(q.relative_to(OUT)),original_path=str(p),original_sha256=sha(raw),archived_sha256=sha(payload),gzip_added=added))
validation=[];total=0
for name,source in [('metadata_tuple_attempt',Path('/tmp/pr38-shared-adapter')),('metadata_timestamp_attempt',Path('/tmp/pr38-shared-adapter-v1')),('study',SRC)]:
 protocol=read(source/'protocol.json')
 with tarfile.open(source/'source.tar.gz') as archive:
  for path,digest in protocol['source_sha256'].items():assert sha(archive.extractfile(path).read())==digest,(name,path)
 old_rows=read(source/'index.json')['records']
 for row in old_rows:
  p=source/row['artifact'];raw=gzip.decompress(p.read_bytes());assert sha(raw)==row['uncompressed_sha256'];value=json.loads(raw)
  assert test_verdict(value['spec'],value['result'])==value['verdict']
  assert value['verdict']['convergence']['complete'] and len(value['result']['observations'])==24
  assert value['mechanism']==mechanism(value['mechanism']['fraction'])
  assert value['adapter']['mechanism']==value['mechanism']
  for tr in value['adapter']['trace']:
   assert 0<tr['factor']<=1
   if value['mechanism']['fraction'] is not None:
    assert tr['applied_rms'] <= value['mechanism']['fraction']*max(tr['parameter_rms'],.1)+2e-7,(name,tr)
  if row['stage']=='control':
   ref=Path('reports/transfer_suite/unadjusted/runs/round0-2/episodes')/f"lr00425_prior2__{row['spec']['name']}.json.gz"
   target=read(ref)
   assert clean(value['result'])==clean(target['result'])
   assert canonical(value['recipe'])==target['recipe']
   assert canonical(value['spec'])==target['spec']
   assert value['applied']==target['applied']
  total+=1
 validation.append(dict(stage=name,episodes=len(old_rows),all_sources_raw_bytes_and_recomputed_verdicts_valid=True))
 copy_tree(source,OUT/name)
 log=source.with_suffix('.log');target=OUT/name/'run.log';target.write_bytes(log.read_bytes())
 manifest.append(dict(path=str(target.relative_to(OUT)),original_path=str(log),original_sha256=sha(log.read_bytes()),archived_sha256=sha(target.read_bytes()),gzip_added=False))
assert total==38
for p in [Path(__file__),Path('tests/test_relative_step_adapter.py'),Path('benchmarks/transfer_suite/relative_step_adapter.md')]:
 q=OUT/'reproduction'/p.name;q.parent.mkdir(exist_ok=True);q.write_bytes(p.read_bytes());manifest.append(dict(path=str(q.relative_to(OUT)),original_path=str(p),original_sha256=sha(p.read_bytes()),archived_sha256=sha(q.read_bytes()),gzip_added=False))
baseline=[r for r in read(Path('reports/transfer_suite/unadjusted/runs/round0-2/index.json'))['records'] if r['recipe']['name']=='lr00425_prior2']
base_by={r['spec']['name']:r for r in baseline}
lines=['# One shared relative-step adapter: results','',
 f'**Selected candidate `{selected}` sustains {sum(r["verdict"]["passed"] for r in selected_rows)}/19, versus15/19 for the unchanged Adam baseline.** '
 'This is an altered optimizer mechanism applied with one equation and one scalar fraction everywhere. There is no overall PASS unless all19live tests pass. No production default changes.',
 '', 'The baseline remains G/D LR.00425, particleLR.0085, Adam(0,.99), Rp logistic, b_cap3/κ1.25, spread.05, no particleL2, delayed cosinehold60%/floor5%. All data, auxiliary objectives, architecture profile, particle counts, batches and budgets are unchanged. Seed0only.',
 '', 'Each ordinary Adam proposal delta is multiplied by min(1, fraction×max(RMS(parameter_before),.1)/(RMS(delta)+1e−12)). The fraction is the only candidate knob; .01,.025,.05 were declared before results. Adam moments receive unchanged raw gradients. The equation never receives role names, task names, target labels, metrics or time; roles only label logs. Its interaction with the unchanged schedule occurs through the Adam proposal.',
 '', '| Candidate | Screen live /6 | Attempted /19 | Full live /19 |', '| --- | ---: | ---: | --- |']
for name in [c['name'] for c in read(SRC/'plan.json')['candidates']]:
 rr=[r for r in rows if r['adapter_candidate']==name];screen=[r for r in rr if r['stage']=='screen'];lines.append(f"| {name} | {sum(r['verdict']['passed'] for r in screen)}/6 | {len(rr)}/19 | {str(sum(r['verdict']['passed'] for r in rr))+'/19' if len(rr)==19 else 'INCOMPLETE'} |")
lines+=['', 'Selection was frozen: screen pass count, then mean final normalized bound shortfall, then candidate name. Exactly one candidate completes the remaining13tasks without further tuning. Incomplete candidates cannot claim19-task coverage.', '', '| Task | Baseline | Selected live | Final suffix | Selected EMA |', '| --- | --- | --- | ---: | --- |']
for row in selected_rows:
 name=row['spec']['name'];lines.append(f"| {name} | {base_by[name]['verdict']['status']} | {row['verdict']['status']} | {row['verdict']['convergence']['passing_suffix']} | {row['ema_verdict']['status']} |")
roles={}
for row in selected_rows:
 for name,s in row['attenuation'].items():
  out=roles.setdefault(name,{k:0 for k in s});
  for k,v in s.items():
   if k=='minimum_factor':out[k]=min(out[k] if out[k] else 1.,v)
   else:out[k]+=v
lines+=['', '## Measured attenuation and overhead','', '| Reported role | Tensor-updates attenuated | Mean factor | Minimum factor |', '| --- | ---: | ---: | ---: |']
for role,s in roles.items():lines.append(f"| {role} | {s['attenuated']/s['tensors']:.2%} | {s['factor_sum']/s['tensors']:.4f} | {s['minimum_factor']:.5f} |")
seconds=sum(r['seconds'] for r in selected_rows);overhead=sum(r['adapter_seconds'] for r in selected_rows)
lines += ['', f'Selected full-suite wall time sums to {seconds:.2f}s; measured adaptation/instrumentation overhead is {overhead:.2f}s ({overhead/seconds:.2%}). This includes cloning, norm calculation and tracing, and excludes ordinary Adam.step time. It is a concurrent CPU measurement, not a production performance comparison. Role aggregates weight each tensor-update equally; they are descriptive, never inputs to the rule.',
 '', '## Validation and retained failures','',
 'All3final identity controls match archived24live/EMA observations, actions, recipes, actualgroupLRs/betas and specs exactly apart from runtime timestamps. Two initial control attempts stopped on reporting comparisons (tuple/list serialization, then created_at); their numerics also match exactly after canonicalization. They remain intact with original source archives and failure logs. No numerical training error occurred.',
 '', f'All{total}completed episodes are retained:31new candidate episodes and7identity control episodes across attempts. Four analytical contract tests verify identity parity, exact capped direction/relativebound, unchangedAdam moments, and role independence; invalid cards failclosed. Source snapshots, all raw payload hashes, recomputed live verdicts and all traced relative-stepbounds validate. EMA is separate.',
 '', 'The source includes a reusable mechanism and frozen study CLI; importing the rows must preserve the mechanism card alongside the base recipe. Do not label these rows plainAdam or combine their passes with another recipe. The bounded negative or mixed result does not rule out other generic adapters.',
 '', '```bash', '/tmp/pr38-default-env/bin/python -u -m benchmarks.transfer_suite.shared_adapter_search \\', '  --output /tmp/shared-adapter-replay > /tmp/shared-adapter-replay.log 2>&1', '```', '', '[Incremental leaderboard](study/README.md) · [All episodes and settings](study/index.json.gz) · [Frozen selection](study/selection.json.gz) · [Exact sources](study/source.tar.gz) · [Raw tensor traces](study/episodes/) · [Artifact manifest](archive_manifest.json).','']
(OUT/'README.md').write_text('\n'.join(lines))
(OUT/'validation.json').write_text(json.dumps(dict(stages=validation,all_episodes=total,candidate_episodes=31,identity_control_episodes=7,selected=selected,full_suite_passes=sum(r['verdict']['passed'] for r in selected_rows),no_numerical_errors=True),indent=2)+'\n')
for name in ['README.md','validation.json']:
 p=OUT/name;manifest.append(dict(path=name,original_path='generated',original_sha256=sha(p.read_bytes()),archived_sha256=sha(p.read_bytes()),gzip_added=False))
(OUT/'archive_manifest.json').write_text(json.dumps(dict(files=manifest,original_bytes_retained=True),indent=2)+'\n')
print('ARCHIVED',OUT,len(manifest),'files',total,'episodes')
