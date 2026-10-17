"""Seal sources and pure gate AST inventory without reading run metrics/PT."""
from __future__ import annotations
import __future__
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules
assert not (HERE/'HELPERS-FROZEN.json').exists()
spec=importlib.util.spec_from_file_location('ra10_grid_source_preparation',HERE/'audit_grid.py')
audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
sha=audit.sha; read=audit.read
inputs={}


def add(name,digest=None):
    path=Path(name).resolve(); digest=digest or sha(path)
    assert sha(path)==digest,str(path)
    if str(path) in inputs: assert inputs[str(path)]==digest,str(path)
    inputs[str(path)]=digest


ready_path=ROOT/'quality/ra10/READY.json'
add(ready_path,audit.READY_SHA); ready=read(ready_path)
assert ready['backend_schema']==9 and ready['trainer_schema']==5
assert ready['package_sha256']==audit.PACKAGE_SHA and ready['config_sha256']==audit.CONFIG_SHA
for name,h in ready['numerical_source_sha256'].items(): add(name,h)
freeze_path=audit.LANE/'source-freeze.json'; add(freeze_path,audit.LANE_SHA)
frozen=read(freeze_path)
assert len(frozen['local_sources'])==18 and len(frozen['external_sources'])==127
for name,h in frozen['local_sources'].items(): add(audit.LANE/name,h)
for name,h in frozen['external_sources'].items(): add(name,h)
screen=read(audit.LANE/'screens/source-freeze.json')
for item in screen['files'].values(): add(item['path'],item['sha256'])
lane_review=audit.LANE_REVIEW
add(lane_review/'receipt.json','5404526292d32aa7f462992314888e1ff61b0d5f55aa8c2c871fb93416053c54')
add(lane_review/'FROZEN.json','4bf756afb872a5389b7cdaa9c8291b1431786fa30925a430ca1662e6c80805c4')
add(lane_review/'MONITOR-START.json','dfcee3d1662f74d0057de38fa55ff3534fcba56040f9597f4180fe50acff2c38')
add(lane_review/'MONITOR-START-FROZEN.json','c289ae13650b70e25978b38c0a779fbbfa117c3ee6bf73212638f731b0df709a')
for part in ('source_sha256','source_and_input_sha256'):
    for name,h in read(lane_review/'FROZEN.json')[part].items(): add(name,h)
assert read(lane_review/'receipt.json')['status']=='PASS'
owned=read(lane_review/'MONITOR-START.json')
assert owned['pid']==1030112 and owned['startticks']==167460340
assert owned['output']==str(audit.MONITOR) and owned['source_freeze_sha256']==audit.LANE_SHA
assert owned['monitor_sha256']=='4ee4cae810342b675a39b269112ca907be3ffe88ae9cadab157e4933c487328b'
add(audit.REFERENCE,'2787a69bfeb42b5d48e02c0105f21351e4d9dffd32b693efe0625c72c08abce8')
add(ROOT/'performance/sampler-regression/cpu-plan-review/post-ra8-quality/ra9-grid-final-review/audit_completed.py')
fixture=read(audit.HARNESS/'tasks/native100_fixture.json')
for name,h in fixture['host_source_sha256'].items(): add(Path(fixture['frozen_repo'])/name,h)
assert read(audit.LANE/'screens/DECLARED-API-EXPECTATION.json')['exact_change']==(
    'collect.expected_options.evaluation_generate: plain -> indexed')
package=ROOT/'pkg-CB64-RA10/particlegan'
actual={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*.py'))}
assert actual==ready['package_source_sha256'] and len(actual)==30
digest=hashlib.sha256()
for name in sorted(actual): digest.update(name.encode()+b'\0'+(package/name).read_bytes()+b'\0')
assert digest.hexdigest()==audit.PACKAGE_SHA
# Check the inherited canonical run has no diagnostic-only dry-run activation.
assert 'self.dry_run = False' in (package/'birth_death.py').read_text()
assert 'dry_run' not in (audit.HARNESS/'screen.py').read_text()
nodes=audit.pure_nodes()
compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
    '<selected original pure ASTs, not executed>','exec',flags=__future__.annotations.compiler_flag)
node_map=[]
for node in nodes:
    name=node.name if isinstance(node,ast.FunctionDef) else ','.join(t.id for t in node.targets)
    node_map.append(dict(name=name,ast_sha256=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()))
helpers={str(HERE/name):sha(HERE/name) for name in ('PROTOCOL.md','audit_grid.py','prepare_audit.py','seal_completed.py')}
for name in helpers:
    if name.endswith('.py'): compile(Path(name).read_text(),name,'exec')
assert 'torch' not in sys.modules
receipt=dict(status='PASS',scope='SOURCE_ONLY_CHECKER_PREPARATION',utc=datetime.now(timezone.utc).isoformat(),
    variant='CB64-RA10',ready_sha256=audit.READY_SHA,source_freeze_sha256=audit.LANE_SHA,
    package_sha256=audit.PACKAGE_SHA,config_sha256=audit.CONFIG_SHA,
    checked_source_inputs=len(inputs),helpers=len(helpers),pure_ASTs=node_map,
    bound_owned_canonical_monitor=dict(pid=owned['pid'],startticks=owned['startticks'],output=owned['output']),
    source_only=True,Torch_imported=False,PT_objects_loaded=0,models_instantiated=0,model_forwards=0,
    metrics_or_quality_verdicts_measured=0,numerical_gate_functions_executed=0,
    write_scope=str(HERE),runtime_status='NOT_STARTED_AWAIT_PARENT_AUTHORIZATION')
audit.write('PREPARATION-RECEIPT.json',receipt)
helpers[str(HERE/'PREPARATION-RECEIPT.json')]=sha(HERE/'PREPARATION-RECEIPT.json')
seal=dict(status='PASS',scope='PRE_RUNTIME_SOURCE_ONLY_HELPER_INPUT_SEAL',
    utc=datetime.now(timezone.utc).isoformat(),source_and_input_sha256=inputs,helper_sha256=helpers,
    package_sha256=audit.PACKAGE_SHA,config_sha256=audit.CONFIG_SHA,
    ready_sha256=audit.READY_SHA,source_freeze_sha256=audit.LANE_SHA)
audit.write('HELPERS-FROZEN.json',seal)
print(json.dumps(dict(status='PASS',source_inputs=len(inputs),helpers=len(helpers),
    helper_freeze_sha256=sha(HERE/'HELPERS-FROZEN.json'))),flush=True)
