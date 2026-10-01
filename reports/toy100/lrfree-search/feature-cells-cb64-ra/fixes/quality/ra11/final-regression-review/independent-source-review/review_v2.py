"""Source-only rereview of the versioned legacy JSON serialization correction."""
import ast
import copy
import hashlib
import json
from pathlib import Path
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
OWNER=ROOT/'quality/ra11/final-regression-review';V2=OWNER/'preparation-attempt2'
HERE=Path(__file__).resolve().parent
EXPECTED={'CHECKER-READY.json':'e44ae3946b78d45a61e281fc5c52d66d6b152cbb0a8dbb62390162056d5cd280',
 'PREPARATION-FINAL-FROZEN.json':'1cd2bbea316e5e08267786816e172b8c86ce53a6cc9c25546ac5918f31321450'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def tree(p):return ast.parse(Path(p).read_text())
def dump(t):return ast.dump(t,include_attributes=False)
for name,digest in EXPECTED.items():assert sha(V2/name)==digest,name
prep=read(V2/'PREPARATION-FINAL-FROZEN.json');mapping=dict(prep['files']);assert len(mapping)==773
for p,d in mapping.items():assert sha(p)==d,p
v1=read(HERE/'receipt-v1.json');assert v1['status']=='SOURCE_ISSUE'
for p,d in v1['protected_sha256'].items():assert sha(p)==d,p
for name in ['audit_final.py','launch_once.py','seal_inputs.py']:assert (OWNER/name).read_bytes()==(V2/name).read_bytes(),name
assert sha(V2/'audit_final.py')=='4f6ab96bad5701a2c4a9fca975a32c4fb1f89e18f6dae7258b940c8d5b100cf5'
common=tree(V2/'common.py');inverse=copy.deepcopy(common)
assert len([n for n in inverse.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FINAL_AREA' for t in n.targets)])==1
inverse.body=[n for n in inverse.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FINAL_AREA' for t in n.targets))]
writer=next(n for n in inverse.body if isinstance(n,ast.FunctionDef) and n.name=='write_new')
dumps=next(n for n in ast.walk(writer) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='dumps')
assert not any(k.arg=='allow_nan' for k in dumps.keywords)
dumps.keywords.append(ast.keyword(arg='allow_nan',value=ast.Constant(value=False)))
assert dump(inverse)==dump(tree(OWNER/'common.py'))
class PublicationInverse(ast.NodeTransformer):
 def visit_Name(self,n):
  if n.id=='FINAL_AREA':n.id='HERE'
  return n
assert dump(PublicationInverse().visit(tree(V2/'close_review.py')))==dump(tree(OWNER/'close_review.py'))
prepare=tree(V2/'prepare.py');extra=[n for n in prepare.body if isinstance(n,ast.For) and 'REVISION.json' in ast.unparse(n)]
assert len(extra)==1
prepare.body=[n for n in prepare.body if n is not extra[0]]
assert dump(prepare)==dump(tree(OWNER/'prepare.py'))
assert (V2/'PLAN.md').read_text().startswith((OWNER/'PLAN.md').read_text())
ready=read(V2/'CHECKER-READY.json');first=read(OWNER/'CHECKER-READY.json')
for key in ['original_authority_function_ast_sha256','original_authority_sha256','planned_PT_loads',
 'original_semantic_exclusions','tasks','portability','native','total_jobs','total_screens','supplement','quality_scope']:
 assert ready[key]==first[key],key
for name,digest in EXPECTED.items():mapping[str(V2/name)]=digest
for name in ['receipt-v1.json','FROZEN-V1.json','review_v1.py','review-v1-attempt1.log']:
 mapping[str(HERE/name)]=sha(HERE/name)
checks=dict(v1['checks']);checks.update(legacy_JSON_serialization_restored=True,
 v1_sources_and_issue_evidence_preserved=True,whole_common_AST_inverse_exact=True,
 whole_close_publication_AST_inverse_exact=True,whole_prepare_provenance_AST_inverse_exact=True,
 audit_launch_input_sealer_bytes_exact=True,finite_mean_and_last_validators_unchanged=True)
value={'status':'PASS','scope':'INDEPENDENT_SOURCE_ONLY_PREEXECUTION_V2_REVIEW','checks':checks,
 'checker_ready_sha256':EXPECTED['CHECKER-READY.json'],'preparation_final_sha256':EXPECTED['PREPARATION-FINAL-FROZEN.json'],
 'helper_sha256':sha(V2/'audit_final.py'),'protected_sha256':mapping,
 'original_authority_function_ast_sha256':ready['original_authority_function_ast_sha256'],
 'issue_resolution':'Legacy JSON default NaN/Infinity semantics restored; typed finite schema2 mean validation exact.',
 'final_input_freeze_and_root_GO_still_required':True,'Torch_imported':False,'PT_loaded':0,
 'authority_functions_executed':0,'model_forwards':0,'numerical_tests':0,
 'limits':['No final numerical artifacts interpreted. This qualifies checker source, not artifact validity or quality.',
 'Four replay endpoints are checked at1010; historical1008 IDs grant no current reset/inheritance claim.',
 'Closed toy/grid/API proofs are reused without loads or repeated numerical tests.']}
target=HERE/'receipt.json';assert not target.exists();target.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({'status':'PASS','guards':len(mapping),'receipt_sha256':sha(target)}))
