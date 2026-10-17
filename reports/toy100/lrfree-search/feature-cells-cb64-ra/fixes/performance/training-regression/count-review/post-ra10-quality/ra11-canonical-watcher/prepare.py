"""Stdlib-only preparation; preserve the qualified monitor and original checks."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
LANE=ROOT/'validation-cb64-ra11'
BASE=ROOT.parent/'feature-cells-cuda-retest-20260929'
MONITOR=ROOT/'integration/review/monitor_validation.py'
OUTPUT=ROOT/'integration/review/validation-cb64-ra11-monitor'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
dump=lambda node:ast.dump(node,include_attributes=False)
mapping={}
def pin(path,digest=None):
    path=Path(path).resolve();actual=sha(path)
    assert digest is None or actual==digest,str(path)
    assert str(path) not in mapping or mapping[str(path)]==actual
    mapping[str(path)]=actual
    return actual

assert not OUTPUT.exists() and not (HERE/'WATCHER-READY.json').exists()
assert pin(MONITOR)=='4ee4cae810342b675a39b269112ca907be3ffe88ae9cadab157e4933c487328b'
freeze=json.loads((LANE/'source-freeze.json').read_text())
source_freeze_sha=pin(LANE/'source-freeze.json')
for name,digest in freeze['local_sources'].items():pin(LANE/name,digest)
for name,digest in freeze['external_sources'].items():pin(name,digest)
root_ready=ROOT/'quality/ra11/READY.json';root_ready_sha=pin(root_ready)
ready=json.loads(root_ready.read_text())
assert ready['backend_schema']==10 and ready['trainer_schema']==5
assert ready['package_sha256']==freeze['package_sha256']=='1b54cb00461df0ad89fcad59bba1e3012bf94a71ac887ecf708aafdfadc18a93'
assert ready['config_sha256']==freeze['config_sha256']=='b3656ea7494413106484c556e53877900dd5ccebf2597b3f80b7d0e7d5ba4437'
peer=ROOT/'quality/ra11/lane-review'
for name in ('receipt.json','FROZEN.json'):pin(peer/name)
assert json.loads((peer/'receipt.json').read_text())['status']=='PASS'
peer_freeze=json.loads((peer/'FROZEN.json').read_text())
for key,values in peer_freeze.items():
    if isinstance(values,dict) and values and all(isinstance(p,str) and p.startswith('/') and isinstance(h,str) and len(h)==64 for p,h in values.items()):
        for path,digest in values.items():pin(path,digest)
screen_freeze=LANE/'screens/source-freeze.json';pin(screen_freeze)
for item in json.loads(screen_freeze.read_text())['files'].values():pin(item['path'],item['sha256'])
declaration_path=LANE/'screens/DECLARED-API-EXPECTATION.json';pin(declaration_path)
declaration=json.loads(declaration_path.read_text())
collector=ast.parse((LANE/'screens/collect.py').read_text())
original=ast.parse((BASE/'screens/collect.py').read_text().replace('CB64-RA','CB64-RA11'))
assert declaration['adapted_ast_sha256']==hashlib.sha256(dump(collector).encode()).hexdigest()
assert declaration['original_ast_sha256']==hashlib.sha256(dump(original).encode()).hexdigest()
options=[node for node in ast.walk(collector) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected_options' for t in node.targets)]
assert len(options)==1
generation=next(k.value for k in options[0].value.keywords if k.arg=='evaluation_generate')
assert isinstance(generation,ast.Constant) and generation.value=='indexed'
generation.value='plain'
assert dump(collector)==dump(original)
assert declaration['declared_before_numerical_execution'] and declaration['all_other_collector_checks_exact'] and declaration['quality_gates_unchanged']
current=ast.parse((LANE/'screens/lane.py').read_text());baseline=ast.parse((BASE/'screens/lane.py').read_text())
def assignment(tree,name):
    return next(node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets))
tasks=[]
for name in ('PORTABILITY','NATIVE'):
    assert dump(assignment(current,name))==dump(assignment(baseline,name))
    tasks.extend(ast.literal_eval(assignment(current,name).value))
assert len(tasks)==16
assert dump(assignment(current,'TASKS'))==dump(assignment(baseline,'TASKS'))
for tree in (current,baseline):
    assert next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='task_plan')
assert dump(next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name=='task_plan'))==dump(next(n for n in baseline.body if isinstance(n,ast.FunctionDef) and n.name=='task_plan'))
pin(BASE/'screens/collect.py');pin(BASE/'screens/lane.py')
old=ROOT/'performance/sampler-regression/cpu-plan-review/post-ra9-quality/ra10-lane-review'
for name in ('start_monitor.py','MONITOR-START.json','MONITOR-START-FROZEN.json'):pin(old/name)
for name in ('prepare.py','start_monitor.py','README.md'):pin(HERE/name)
value=dict(status='READY_SOURCE_ONLY',utc=datetime.now(timezone.utc).isoformat(),
    validation=str(LANE),output=str(OUTPUT),monitor=str(MONITOR),monitor_sha256=sha(MONITOR),
    root_ready_sha256=root_ready_sha,source_freeze_sha256=source_freeze_sha,
    source_and_input_sha256=mapping,tasks=tasks,total_screens=16,portability_screens=13,native_screens=3,
    collector_exception='collect.expected_options.evaluation_generate: plain -> indexed; declared before lane freeze',
    original_quality_checks_unchanged=True,write_redirection_only=True,Torch_imported=False,
    PT_objects_loaded=0,numerical_execution=False,watcher_started=False,
    command=['/tmp/pr38-default-env/bin/python','-B',str(HERE/'start_monitor.py')],
    environment=dict(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),
    lifetime='Collect all16 if root continues original queue; root stops only this owned PID on quality failure.')
(HERE/'WATCHER-READY.json').write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=value['status'],ready_sha256=sha(HERE/'WATCHER-READY.json'),protected=len(mapping))))
