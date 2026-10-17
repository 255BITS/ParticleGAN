"""Independent RA8 owner/composition/source and scalar semantic-law audit.

Pure Python; no torch import, model forward, gradient, optimizer or sample.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
OWNER = ROOT/'performance/sampler-regression/cpu-plan-review/post-ra7-quality/paired-average'
BASE = ROOT/'pkg-CB64-RA7/particlegan'
parse = argparse.ArgumentParser()
parse.add_argument('--feature-sha',required=True)
parse.add_argument('--training-sha',required=True)
parse.add_argument('--owner-ready-sha',required=True)
args = parse.parse_args()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
dump = lambda x: ast.dump(x,include_attributes=False)
verified = {}
def check(path,expected):
    path = Path(path)
    assert sha(path)==expected,str(path)
    verified[str(path)] = expected

assert not (HERE/'receipt.json').exists()
check(OWNER/'READY.json',args.owner_ready_sha)
ready = read(OWNER/'READY.json')
assert ready['status']=='CPU_QUALIFIED_PRIVATE_PAIRED_AVERAGE_GPU_PENDING'
assert ready['quality_verdict'] is None and not ready['default_package_promoted']
proposed = OWNER/'pkg-PAIR-AVERAGE/particlegan'
check(proposed/'feature_cells.py',args.feature_sha)
check(proposed/'training.py',args.training_sha)
check(BASE/'feature_cells.py','eee5469b420d9c750d9ad015172af58052e8aab161d93d4f16fc323be12f9245')
check(BASE/'training.py','8961dc8596230bc0ff5f153606694aa21745e77f5577a0fc93664d9107d5b005')
check(ROOT/'quality/ra7/READY.json','f300089ce4fece3a812b890d4060e0567e833dddd236e5a955993f65ef15e1dd')
owner_map = ready.get('package_source_sha256',ready.get('source_sha256'))
assert isinstance(owner_map,dict) and len(owner_map)==29
for name,expected in owner_map.items():
    check(proposed/name,expected)
def package_identity(directory):
    digest=hashlib.sha256()
    for name in sorted(owner_map):
        digest.update(('particlegan/'+name).encode()+b'\0')
        digest.update((directory/name).read_bytes()+b'\0')
    return digest.hexdigest()
assert package_identity(BASE)=='671404988209f615aefd1aec32ab1ef0a51807a9b07514c9bb35a662a6154c7c'
owner_package_identity=package_identity(proposed)
assert ready['package_sha256']==owner_package_identity
for key in ('helper_source_sha256','numerical_source_sha256','local_source_sha256','input_sha256','artifact_sha256'):
    for name,expected in ready.get(key,{}).items():
        check(name,expected)
if (OWNER/'FROZEN.json').exists():
    check(OWNER/'FROZEN.json','6573df9f118a2746159d3a096ea608d1a9fed67745c2c04342919c971900cec6')
    frozen = read(OWNER/'FROZEN.json')
    for name,expected in frozen.get('files',{}).items(): check(name,expected)
    verified[str(OWNER/'FROZEN.json')]=sha(OWNER/'FROZEN.json')
owner_receipt=read(OWNER/'receipt.json')
check(OWNER/'receipt.json',ready['owner_receipt_sha256'])
assert owner_receipt['status']=='PASS' and owner_receipt['matched_plan_actions_numerical_state_and_RNG_exact']
neutral=read(OWNER/'neutrality-receipt.json')
assert neutral['status']=='PASS' and neutral['cpu_only'] and not neutral['cuda_initialized']
assert neutral['quality_verdict'] is None and all(neutral['checks'].values())
assert all(neutral[k]==0 for k in ('new_optimizer_steps','new_training_steps','new_emissions','new_seed_experiments'))
for key in ('source_sha256','input_sha256'):
    for name,expected in neutral[key].items():check(name,expected)
verified[str(OWNER/'neutrality-receipt.json')]=sha(OWNER/'neutrality-receipt.json')
reaction_pair=[read(OWNER/name) for name in ('reaction-RA7.json','reaction-proposal.json')]
geometry=read(OWNER/'gate-receipt.json')
assert geometry['status']=='PASS' and geometry['cpu_only'] and not geometry['cuda_initialized']
assert geometry['quality_verdict'] is None and all(geometry['checks'].values())
assert all(geometry[k]==0 for k in ('new_optimizer_steps','new_training_steps','new_emissions','new_seed_experiments'))
assert [r['step'] for r in geometry['records']]==[500,1000,2000]
assert [r['eligible'] for r in geometry['records']]==[False,False,True]
for key in ('source_sha256','input_sha256'):
    for name,expected in geometry[key].items():check(name,expected)
verified[str(OWNER/'gate-receipt.json')]=sha(OWNER/'gate-receipt.json')
reaction_summary=[]
for filename,record in zip(('reaction-RA7.json','reaction-proposal.json'),reaction_pair):
    assert record['status']=='PASS' and record['cpu_only'] and not record['cuda_initialized']
    assert record['quality_verdict'] is None
    assert all(record[k]==0 for k in ('new_optimizer_steps','new_training_steps','new_emissions','new_seed_experiments'))
    for key in ('source_sha256','input_sha256'):
        for name,expected in record[key].items():check(name,expected)
    verified[str(OWNER/filename)]=sha(OWNER/filename)
assert reaction_pair[0]['input_sha256']==reaction_pair[1]['input_sha256']
assert [r['step'] for r in reaction_pair[0]['records']]==[1250,2000]
for a,b in zip(reaction_pair[0]['records'],reaction_pair[1]['records']):
    assert a['step']==b['step'] and all(a['checks'].values()) and all(b['checks'].values())
    for key in ('plan_sha256','numerical_state_sha256','original_semantic_event_sha256'):
        assert a[key]==b[key],(a['step'],key)
    assert b['paired_average']['step']==b['step']+1
    reaction_summary.append(dict(step=b['step'],plans_actions_numeric_state_old_event_exact=True,
        copy_moves=b['copies'],novel_moves=b['novel'],joint_rows=b['paired_average']['coherent_rows'],
        required=b['paired_average']['required'],geometry_eligible=b['paired_average']['eligible']))

base_fc,fc = (ast.parse(p.read_text()) for p in (BASE/'feature_cells.py',proposed/'feature_cells.py'))
base_tr,tr = (ast.parse(p.read_text()) for p in (BASE/'training.py',proposed/'training.py'))
def find_class(tree,name):
    return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
def method(cls,name):
    return next(n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
fc_class = find_class(fc,'FeatureCellBirthDeath')
new_methods = {'paired_average_eligible','_record_paired_average','_check_paired_average_state','check_paired_average_step'}
assert {n.name for n in fc_class.body if isinstance(n,ast.FunctionDef)}-{n.name for n in find_class(base_fc,'FeatureCellBirthDeath').body if isinstance(n,ast.FunctionDef)}==new_methods
assert len([n for n in fc.body if isinstance(n,ast.FunctionDef) and n.name=='paired_average_geometry'])==1
inverse_fc = deepcopy(fc)
inverse_fc.body=[n for n in inverse_fc.body if not(isinstance(n,ast.FunctionDef) and n.name=='paired_average_geometry')]
inverse_class=find_class(inverse_fc,'FeatureCellBirthDeath')
inverse_class.body=[n for n in inverse_class.body if not(isinstance(n,ast.FunctionDef) and n.name in new_methods)]
class UndoDeclaredFeatureSplice(ast.NodeTransformer):
    def visit_Assign(self,node):
        target=ast.unparse(node.targets[0]) if len(node.targets)==1 else ''
        if target=='BACKEND_SCHEMA':
            assert isinstance(node.value,ast.Constant) and node.value.value==7
            node.value=ast.Constant(value=6)
        if target in ('self.paired_average',"last['paired_average']","last['paired_average_forward_rows']"):
            return None
        return self.generic_visit(node)
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call):
            function=ast.unparse(node.value.func)
            if function in ('self._record_paired_average','self._check_paired_average_state'):
                return None
            if function=='self.settings.update' and any(k.arg=='paired_average_policy' for k in node.value.keywords):
                assert {k.arg for k in node.value.keywords}=={'paired_average_policy','paired_average_expiry'}
                return None
        return self.generic_visit(node)
    def visit_Call(self,node):
        if ast.unparse(node.func)=='result.update':
            node.keywords=[k for k in node.keywords if k.arg not in ('paired_average','paired_average_age_real_rows')]
        return self.generic_visit(node)
    def visit_Set(self,node):
        node.elts=[v for v in node.elts if not(isinstance(v,ast.Constant) and v.value=='paired_average')]
        return self.generic_visit(node)
UndoDeclaredFeatureSplice().visit(inverse_class)
assert dump(inverse_fc)==dump(base_fc),'unexpected feature/training action AST change'

inverse_tr=deepcopy(tr)
trainer_class=find_class(inverse_tr,'GANTrainer')
base_trainer=find_class(base_tr,'GANTrainer')
serving=method(trainer_class,'_serve_settled')
assert ast.unparse(serving.body[1].test)=="self.birth_death is not None and hasattr(self.birth_death, 'paired_average_eligible')"
assert ast.unparse(serving.body[1].body[0])=='return self.birth_death.paired_average_eligible(self.completed_steps)'
serving.body.pop(1)
serving.body[0]=deepcopy(method(base_trainer,'_serve_settled').body[0])
class UndoStepValidator(ast.NodeTransformer):
    removed=0
    def visit_If(self,node):
        if ast.unparse(node.test)=="hasattr(self.birth_death, 'check_paired_average_step')":
            assert len(node.body)==1 and ast.unparse(node.body[0])=="self.birth_death.check_paired_average_step(state['birth_death'], steps)"
            self.removed+=1
            return None
        return self.generic_visit(node)
undo=UndoStepValidator()
undo.visit(method(trainer_class,'_load_state_dict'))
assert undo.removed==1 and dump(inverse_tr)==dump(base_tr),'unexpected trainer numeric/EMA/fallback AST change'

# Execute only scalar law methods, not constructors, imported Torch or models.
selected=[deepcopy(method(fc_class,n)) for n in ('paired_average_eligible','_check_paired_average_state','check_paired_average_step')]
for n in selected:n.decorator_list=[]
ns=dict(math=math,Q=.05)
exec(compile(ast.Module(body=selected,type_ignores=[]),'<reviewed-scalar-laws>','exec'),ns)
policy='ema_support_inside_current_real_group_Q_v1'
stamp=dict(schema=1,policy=policy,snapshot=1,step=2000,rows=1024,required=973,
    cells=64,rank=8,groups=25,calibration_rows=512,chart_valid=True,duplicate_ok=True,
    finite_rows=1024,same_group_rows=1018,ema_eligible_rows=991,coherent_rows=985,eligible=True)
self=SimpleNamespace(N=1024,settings=dict(rank=8,cells=64,paired_average_policy=policy),
    paired_average=stamp,snapshot_serial=1,snapshot=None,fill=1024,rows_since_eval=0)
def state_of(s):
    return dict(paired_average=s,snapshot_serial=s['snapshot'],
        last=dict(step=s['step'],snapshot=s['snapshot'],cells=64,metric_rank=8,calibration_rows=512,
                  mass_topology=dict(groups=25),duplicate_fraction=0.,paired_average=deepcopy(s)))
ns['_check_paired_average_state'](self,state_of(stamp))
negative=[]
for name,changes in (
    ('boolean-schema',dict(schema=True)),
    ('above-marginal-joint',dict(coherent_rows=1024)),
    ('impossible-empty-intersection',dict(same_group_rows=1024,ema_eligible_rows=1024,coherent_rows=0,eligible=False)),
    ('nonfinite-with-positive-counts',dict(finite_rows=1023,eligible=False)),
    ('invalid-chart-positive-rank',dict(chart_valid=False,eligible=False)),
    ('mismatched-cells',dict(cells=32)),
    ('mismatched-rank',dict(rank=4)),
    ('mismatched-groups',dict(groups=24)),
):
    bad=deepcopy(stamp);bad.update(changes)
    try:ns['_check_paired_average_state'](self,state_of(bad))
    except ValueError:negative.append(name)
    else:raise AssertionError('accepted impossible stamp: '+name)
lease=[]
for age,step,expected in ((0,2000,True),(1023,2007,True),(1024,2008,False),(0,1999,False),(0,2000.,False)):
    self.rows_since_eval=age
    actual=ns['paired_average_eligible'](self,step)
    assert actual is expected
    lease.append(dict(age_real_rows=age,query_step=step,eligible=actual))
try:ns['check_paired_average_step'](self,state_of(stamp),1999)
except ValueError:negative.append('future-trainer-step')
else:raise AssertionError('accepted future stamp')

composition_path=ROOT/'quality/ra8/COMPOSITION.json'
composition=read(composition_path)
verified[str(composition_path)]=sha(composition_path)
candidate=ROOT/'pkg-CB64-RA8/particlegan'
assert set(composition['source_sha256'])==set(owner_map)
for name,expected in owner_map.items():
    check(candidate/name,expected)
    assert composition['source_sha256'][name]==expected
    if name not in ('feature_cells.py','training.py'):
        assert (candidate/name).read_bytes()==(BASE/name).read_bytes()
assert composition['package_sha256']==package_identity(candidate)==owner_package_identity
config=ROOT/'configs/overrides-CB64-RA8.json'
expected_config='08359ed4406148faf6915144ef4265d930829de53775eeafc4f17a11b5667b4c'
check(config,expected_config)
assert config.read_bytes()==(ROOT/'configs/overrides-CB64-RA7.json').read_bytes()
assert composition['config_sha256']==expected_config
for path,expected in verified.items():assert sha(path)==expected,path
receipt=dict(status='VALID',review_status='PASS',scope='RA8 source composition and scalar empirical-geometry state law',
    owner_ready_sha256=args.owner_ready_sha,feature_sha256=args.feature_sha,training_sha256=args.training_sha,
    package_sha256=owner_package_identity,
    composition_sha256=sha(composition_path),module_count=29,other_modules_exact_RA7=27,
    config_byte_exact_RA7=True,full_feature_AST_inverse_exact_RA7=True,full_training_AST_inverse_exact_RA7=True,
    meaningful_malformed_stamp_controls=negative,scalar_FIFO_turnover_and_step_controls=lease,
    reviewed_owner_matched_reactions=reaction_summary,
    reviewed_owner_geometry_checks=len(geometry['checks']),
    reviewed_owner_observational_neutrality_checks=len(neutral['checks']),
    reviewed_owner_malformed_controls=geometry['malformed_controls'],
    saved_clean_geometry_eligibility=[dict(step=r['step'],joint_rows=r['coherent_rows'],eligible=r['eligible']) for r in geometry['records']],
    chart_absent_after_load_is_supported=True,backend_schema=7,trainer_schema=5,
    training_population_law_and_all_action_count_noise_optimizer_gradient_paths_unchanged=True,
    semantic_timings_added=False,model_forwards=0,gradients=0,optimizer_calls=0,new_seeds=0,cuda_initialized=False,
    limits=['Empirical current clean geometry; not stationarity, emitted support equivalence or quality.',
            'Lease ages by real FIFO rows; G/D may change before the next reaction.',
            'No numerical quality run or replay is claimed by this audit.'],verified_hashes=verified)
(HERE/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
files={str(p):sha(p) for p in sorted(HERE.iterdir()) if p.is_file()}
(HERE/'FROZEN.json').write_text(json.dumps(dict(status='VALID',files=files,reviewed_hashes=verified),indent=2)+'\n')
print(json.dumps(dict(status='VALID',review_status='PASS',receipt_sha256=sha(HERE/'receipt.json'),freeze_sha256=sha(HERE/'FROZEN.json'))))
