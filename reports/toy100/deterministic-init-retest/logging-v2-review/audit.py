"""Stdlib logger-only regression contract; never imports package or Torch."""
from pathlib import Path
import ast,hashlib,json
E=Path(__file__).resolve().parents[1];B=E/'logging-v2-screen';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=(E/'mode_hold_harness.py').read_text();new=(B/'mode_hold_harness.py').read_text();tree=ast.parse(new);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='diagnostic_value');span=ast.get_source_segment(new,fn);assert new.replace(span+'\n\n\n','').replace('diagnostic_value(v) for k, v in stats.items()','float(v) for k, v in stats.items()')==old
scope={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'diagnostic_value','exec'),scope);f=scope['diagnostic_value']
for x in [0,1,-2.,3.25]:assert f(x)==float(x)
class Scalar:
 def __float__(self):return 3.5
x={'nested':[Scalar(),{'pass':True,'why':None,'label':'trial'}],'tuple':(1,2.5)}
y=f(x);assert y=={'nested':[3.5,{'pass':True,'why':None,'label':'trial'}],'tuple':[1.,2.5]};assert isinstance(x['nested'][0],Scalar);json.dumps(y,allow_nan=False)
manifest=json.loads((B/'manifest.json').read_text());files=manifest.get('files',manifest)
for n,h in files.items():assert sha(B/n)==h,n
for name in ['mode_hold_contract.py','init_contract.py','preflight.py']:
 assert (B/name).read_bytes()==(E/name).read_bytes()
a=json.loads((E/'protocol.json').read_text());b=json.loads((B/'protocol.json').read_text());print('protocol_changed_keys',[k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)])
out={'status':'PASS_LOGGING_ONLY_REVIEW','manifest_sha256':sha(B/'manifest.json'),'old_harness_sha256':sha(E/'mode_hold_harness.py'),'new_harness_sha256':sha(B/'mode_hold_harness.py'),'exact_harness_diff':'one diagnostic_value recursive serializer added; one stats comprehension float(v) call replaced','checks':['Existing numeric scalar encoding unchanged','Nested dict/list/tuple and optional/string/bool leaves preserved','Input tree not mutated','Complete learner, initializer, sampling, scorer, serial transaction source unchanged','Copied constructor/protocol helper source unchanged','No Torch/model/CPUconstructor/training/GPU execution'], 'limits':['Original RP12–RP15 step1 loggingERROR artifacts remain errors, not quality failures or discarded attempts.','Four fresh declared reruns needed; no previous step1 state reused or quality inferred.']}
(B.parent/'logging-v2-review/source-audit.json').write_text(json.dumps(out,indent=2)+'\n');(B.parent/'logging-v2-review/source-audit.md').write_text('''# Logging v2 independent review\n\nPASS: the complete worker diff adds one recursive diagnostic serializer and replaces one float(v) call in the post-update logging comprehension. Numeric scalar losses retain their exact former float encoding. Nested dict/list/tuple diagnostics and optional/string/bool leaves serialize without changing the input. Learner updates, initialization, data streams, scoring and serial transaction source are unchanged; constructor helpers are byte-identical. A small stdlib contract check exercised nested values and a float-compatible scalar without importing Torch.\n\nThe original RP12–RP15 step1 loggingERROR records remain preserved and have no quality score. Fresh runs are appropriate after this reporting-only repair. No repeated initialization, training or GPU work was performed. Source seal and worker hashes are in source-audit.json.\n''');print(json.dumps(out,indent=2))
