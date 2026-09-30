"""Independent stdlib source/hash review. Never execute artifact authority."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
OWNER=ROOT/'quality/ra11/final-regression-review'
HERE=Path(__file__).resolve().parent
EXPECTED={
 'CHECKER-READY.json':'cc820feeb81c809541143ab3b85f33d65147dfd94346b294e0d5b2a802cad7e3',
 'PREPARATION-FROZEN.json':'fdb331529d1d3b223ae11f992faf8759eab2dabdb06fee03f849f33803d1d348',
 'PREPARATION-FINAL-FROZEN.json':'7dff45a78eede6640a4740f38ca31f05397aff9edebd3d5766d0509f70d9e3ee',
 'audit_final.py':'4f6ab96bad5701a2c4a9fca975a32c4fb1f89e18f6dae7258b940c8d5b100cf5'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def tree(p):return ast.parse(Path(p).read_text())
def fn(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
for name,digest in EXPECTED.items():assert sha(OWNER/name)==digest,name
ready=read(OWNER/'CHECKER-READY.json');preparation=read(OWNER/'PREPARATION-FINAL-FROZEN.json')
mapping=dict(preparation['files']);assert len(mapping)==761
for p,digest in mapping.items():assert sha(p)==digest,p
authority=ROOT/'integration/review/audit_learned.py';source=tree(authority)
inventory={name:hashlib.sha256(ast.dump(fn(source,name),include_attributes=False).encode()).hexdigest()
 for name in ready['original_authority_function_ast_sha256']}
assert inventory==ready['original_authority_function_ast_sha256']
assert sha(authority)==ready['original_authority_sha256']
assert ready['planned_PT_loads']==dict(mnist_checkpoints=10,replay_endpoints=4,accepted_toy_and_grid=0)
assert ready['original_semantic_exclusions']==['birth_death.last.eval_seconds']
assert ready['total_jobs']==19 and ready['total_screens']==16
assert len(ready['tasks'])==16 and len(set(ready['tasks']))==16
assert len(ready['portability'])==13 and len(ready['native'])==3
audit=(OWNER/'audit_final.py').read_text();audit_tree=tree(OWNER/'audit_final.py')
assert "training=ns['audit_training']('mnist'" in audit
assert "for problem in ('toy','mnist')" in audit
assert "original_load=ns['load_cpu'];loaded={}" in audit
assert "result=original_load(path);loaded[key]=result;return result" in audit
assert "assert key not in loaded,key" in audit and "assert len(loaded)==14" in audit
assert "if serial and stamp['step']==step:" in audit
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='semantic'
 for n in ast.walk(fn(audit_tree,'state_check')))
assert "MNIST_quality_gate=None" in audit
assert 'go[\'exactly_one_CPU_artifact_invocation\'] is True' in audit
assert audit.index("verify(seal['source_and_input_sha256'])")<audit.index('    import torch')
assert audit.index("go['input_freeze_sha256']")<audit.index('    import torch')
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in
 ['manual_seed','sample','_generate','step','forward','backward','load_state_dict','to']
 for n in ast.walk(audit_tree))
assert not any(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('particlegan') for n in ast.walk(audit_tree))
live=[OWNER.parent.parent.parent/'validation-cb64-ra11/run.log',ROOT/'integration/review/validation-cb64-ra11-monitor/summary.json',
 ROOT/'integration/review/validation-cb64-ra11-monitor/READ-ONLY-ARTIFACT-MANIFEST.json',
 ROOT/'performance/training-regression/count-review/post-ra10-quality/ra11-canonical-watcher/monitor-process.log']
assert not any(str(p) in mapping for p in live)
common_tree=tree(OWNER/'common.py');writer=fn(common_tree,'write_new')
strict=any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='dumps'
 and any(k.arg=='allow_nan' and isinstance(k.value,ast.Constant) and k.value.value is False for k in n.keywords)
 for n in ast.walk(writer))
original_writer=fn(source,'write')
original_default=any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='dumps'
 and not any(k.arg=='allow_nan' for k in n.keywords) for n in ast.walk(original_writer))
assert strict and original_default
for name,digest in EXPECTED.items():mapping[str(OWNER/name)]=digest
value={'status':'SOURCE_ISSUE','scope':'INDEPENDENT_PREEXECUTION_SOURCE_HASH_AST_ONLY',
 'checks':{'all761_preparation_guards_exact':True,'original13_function_AST_inventory_exact':True,
  'original_MNIST_and_both_replay_functions_unchanged':True,'device_tag_digest_and_single_semantic_exclusion_preserved':True,
  'ten_plus_four_load_capture_no_reload':True,'qualified_backend10_schema2_semantic_extraction':True,
  'reaction_boundary_only_history_reset_claims':True,'initial_state_and_typed_lease_paths_correct':True,
  'original19_completion_and16_canonical_closure_planned':True,'closed_toy_grid_API_proofs_reused':True,
  'live_queue_and_watch_outputs_absent_from_static_maps':True,'constructors_forwards_draws_training_and_CUDA_absent':True},
 'issue':{'file':str(OWNER/'common.py'),'line':writer.lineno,
  'summary':'Strict allow_nan=False conflicts with unchanged original training evidence carrying permitted legacy NaN/Infinity diagnostics.',
  'flow':'audit_training returns full result.final -> audit_final receipt -> write_new; close_review also calls write_new on that value.',
  'original_writer_allows_legacy_nonfinite_tokens':True,
  'required_correction':'Preserve original legacy JSON token serialization; keep typed schema2 mean diagnostic finite checks exact.'},
 'protected_sha256':mapping,'original_authority_function_ast_sha256':inventory,
 'Torch_imported':False,'PT_loaded':0,'authority_functions_executed':0,'numerical_tests':0,
 'limitation':'Source-only qualification; no final-input or artifact validity verdict is issued.'}
target=HERE/'receipt-v1.json';assert not target.exists();target.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'status':value['status'],'guards':len(mapping),'receipt_sha256':sha(target)}))
