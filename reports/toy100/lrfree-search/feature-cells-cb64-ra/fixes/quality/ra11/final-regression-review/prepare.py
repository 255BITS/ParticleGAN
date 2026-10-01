"""Source-only preparation; no Torch import, artifact interpretation or monitor write."""
import ast
from common import *

assert not (HERE/'PREPARATION-FROZEN.json').exists()
mapping={}
assert pin(mapping,ORIGINAL)=='683acf472985084a911edd6763d63d411772563d7fcb71df4a547d2a502347ba'
assert pin(mapping,ROOT/'integration/review/monitor_validation.py')=='4ee4cae810342b675a39b269112ca907be3ffe88ae9cadab157e4933c487328b'
source=read(LANE/'source-freeze.json');pin(mapping,LANE/'source-freeze.json')
assert source['package_sha256']==PACKAGE_SHA and source['config_sha256']==CONFIG_SHA
for path,digest in source['local_sources'].items():pin(mapping,LANE/path,digest)
for path,digest in source['external_sources'].items():pin(mapping,path,digest)
for path in PROOF_SEALS:
    pin(mapping,path);pin_absolute_maps(mapping,read(path))
for path in (ROOT/'quality/ra11/READY.json',ROOT/'quality/ra11/COMPOSITION.json',
             ROOT/'quality/results/CB64-RA11.json',ROOT/'quality/results/CB64-RA11-toy.json',
             ROOT/'quality/results/RA11-regressions-launch.json',WATCHER/'MONITOR-START.json',
             ROOT/'integration/review/ra11-toy-artifact-audit/audit_completed.py'):
    pin(mapping,path)
assert read(ROOT/'quality/results/CB64-RA11.json')['both_quality_gates_pass']
inventory={node.name:ast_sha(node) for node in nodes(ORIGINAL,AUTHORITY_NAMES)}
tree=ast.parse((LANE/'screens/lane.py').read_text())
def assignment(name):
    return next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
portability=ast.literal_eval(assignment('PORTABILITY').value);native=ast.literal_eval(assignment('NATIVE').value)
assert len(portability)==13 and len(native)==3
tasks=portability+native
declaration=read(LANE/'screens/DECLARED-API-EXPECTATION.json')
assert declaration['declared_before_numerical_execution'] and declaration['all_other_collector_checks_exact'] and declaration['quality_gates_unchanged']
for path in sorted(HERE.iterdir()):
    if path.suffix in ('.py','.md'):pin(mapping,path)
ready=dict(status='READY_SOURCE_ONLY',utc=now(),validation=str(LANE),package=str(PACKAGE),
    final_input_freeze_required=True,root_GO_required=True,source_and_input_sha256=mapping,
    original_authority_function_ast_sha256=inventory,original_authority_sha256=sha(ORIGINAL),
    tasks=tasks,portability=portability,native=native,total_jobs=19,total_screens=16,
    planned_PT_loads=dict(mnist_checkpoints=10,replay_endpoints=4,accepted_toy_and_grid=0),
    original_semantic_exclusions=['birth_death.last.eval_seconds'],
    supplement='Qualified backend10/trainer5/schema2 typed output-moment and lease/population/lineage state checks',
    quality_scope='Original outcomes retained; MNIST has no newly introduced pass gate; valid negative regressions do not become INVALID.',
    source_adapter='None for original learned authority; only predeclared collector plain -> indexed expectation.',
    Torch_imported=False,PT_loaded=0,model_forwards=0,new_training_updates=0,new_quality_emissions=0,
    mutable_queue_and_watch_outputs_pinned=False,
    future_command=['/tmp/pr38-default-env/bin/python','-B',str(HERE/'launch_once.py'),'--input-freeze',str(HERE/'completed-inputs/INPUTS-FROZEN.json'),'--root-go',str(HERE/'ROOT-GO.json')])
write_new(HERE/'CHECKER-READY.json',ready)
pin(mapping,HERE/'CHECKER-READY.json')
write_new(HERE/'PREPARATION-FROZEN.json',dict(status='SOURCE_ONLY_PREPARATION_FROZEN',utc=now(),files=mapping))
print(json.dumps(dict(status=ready['status'],ready_sha256=sha(HERE/'CHECKER-READY.json'),freeze_sha256=sha(HERE/'PREPARATION-FROZEN.json'),guards=len(mapping))))
