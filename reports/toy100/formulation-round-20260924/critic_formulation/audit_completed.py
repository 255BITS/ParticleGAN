"""Check new formulation receipts against retained controls, without training."""
import datetime,gzip,hashlib,json,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent;checkout=root.parents[2];sys.path.insert(0,str(root/'prepared/repos/cuda'))
from benchmarks.transfer_suite.protocol import test_verdict
started=time.perf_counter();digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((root/'prepared/prepared-sources.json').read_text());source_errors=[]
for name,expected in manifest['cuda'].items():
 p=root/'prepared/repos/cuda'/name
 if not p.is_file() or digest(p)!=expected:source_errors.append(name)
assert not source_errors,source_errors[:5]
base_config=json.loads((checkout/'configs/toy100/constraints_simple_regularization.json').read_text());rows=[]
for candidate in sorted((root/'candidates').iterdir()):
 declaration_path=candidate/'declaration.json'
 if not declaration_path.exists():continue
 declaration=json.loads(declaration_path.read_text());config=json.loads((candidate/'config.json').read_text())
 assert config==base_config|declaration['changed_fields']
 assert digest(candidate/'probe.py')==declaration['probe_sha256']
 assert digest(candidate/'config.json')==declaration['config_sha256']
 assert digest(root/'prepared/prepared-sources.json')==declaration['prepared_sources_sha256']
 commands=[json.loads(line) for line in (candidate/'commands.jsonl').read_text().splitlines()] if (candidate/'commands.jsonl').exists() else []
 assert all(c['started']>declaration['declared_at'] for c in commands)
 for path in sorted((candidate/'runs').glob('*/result.json')):
  record=json.loads(path.read_text());task=record['task'];assert record['status']!='ERROR'
  recalculated=test_verdict(record['spec'],record['result']);assert recalculated==record['verdict']
  assert record['worker_sha256']==declaration['probe_sha256']
  fixture=checkout/'reports/toy100/cpu-recipe-gpu-port/initialization-fixtures'/task/'initial-values.pt'
  assert record['initialization_fixture_sha256']==digest(fixture)
  reference_path=checkout/'reports/toy100/cpu-recipe-gpu-port/runs/cuda_cpu_init'/(task+'.json.gz')
  reference=json.loads(gzip.decompress(reference_path.read_bytes()))
  assert record['proof']['initial_optimizers']==reference['proof']['initial_optimizers']
  assert record['randomness']['sha256']==reference['randomness']['sha256']
  proof=record['proof'];budget=record['spec']['steps'];optimizers=list(proof['optimizers'].values())
  assert proof['adam_calls']==2*budget and len(optimizers)==2
  assert all(p['calls']==budget and p['device']=='cuda:0' for p in optimizers)
  assert record['backend']=='cuda' and not record['cpu_random']
  assert record['environment']['CUDA_VISIBLE_DEVICES']=='GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69'
  assert record['environment']['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
  rows.append(dict(candidate=candidate.name,gate=task,status=record['status'],source_bound=True,frozen_verdict_recomputed=True,cpu_fixture_matched=True,native_random_stream_matched=True,adam_calls=proof['adam_calls'],cuda=True))
result=dict(status='PASS',audited_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_files_verified=len(manifest['cuda']),runs=len(rows),receipts=rows)
artifact=root/'receipt-audit.json';artifact.write_text(json.dumps(result,indent=2)+'\n')
ledger=dict(candidate='regression',gate='formulation_receipt_audit',status='PASS',seconds=time.perf_counter()-started,metrics={k:v for k,v in result.items() if k!='receipts'},artifact=str(artifact))
with (checkout.parent/'tests.jsonl').open('a') as stream:stream.write(json.dumps(ledger)+'\n')
print(json.dumps({k:v for k,v in ledger.items() if k!='metrics'}|{'runs':len(rows),'source_files_verified':len(manifest['cuda'])}))
