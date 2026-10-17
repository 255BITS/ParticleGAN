"""Independent stdlib-only resolution source/scalar-state review."""
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
OWNER=ROOT/'performance/sampler-regression/cpu-plan-review/post-ra8-quality/resolution'
BASE=ROOT/'pkg-CB64-RA8/particlegan'
PKG=OWNER/'pkg-RESOLUTION/particlegan'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
dump=lambda n:ast.dump(n,include_attributes=False)
assert not (HERE/'static-receipt.json').exists()
assert sha(OWNER/'SOURCE-FROZEN.json')=='06649bc07d660c2c54f2927c6b39c0a8c995a2f5f4f7877182ebdf83f9e65294'
seal=read(OWNER/'SOURCE-FROZEN.json')
print('seal keys',list(seal))
maps=seal.get('file_sha256',seal.get('source_and_input_sha256',seal.get('source_sha256')))
assert isinstance(maps,dict)
maps=dict(maps)
maps.update({str(p):sha(p) for p in (OWNER/'SOURCE-FROZEN.json',Path(__file__),HERE/'failed-inspection1.txt')})
for p,d in maps.items():assert sha(p)==d,p
sources={str(p.relative_to(PKG)):sha(p) for p in sorted(PKG.rglob('*.py'))}
assert len(sources)==29
unchanged=[]
for name,d in sources.items():
    if name!='feature_cells.py':assert sha(BASE/name)==d;unchanged.append(name)
oldconfig=read(ROOT/'configs/overrides-CB64-RA8.json');newconfig=read(OWNER/'config.json')
assert set(oldconfig)==set(newconfig)
delta={k:[oldconfig[k],newconfig[k]] for k in oldconfig if oldconfig[k]!=newconfig[k]}
assert delta=={'birth_death_cells':[64,128]}
old=ast.parse((BASE/'feature_cells.py').read_text());new=ast.parse((PKG/'feature_cells.py').read_text())
oc={v.name:v for v in old.body if isinstance(v,ast.ClassDef)};nc={v.name:v for v in new.body if isinstance(v,ast.ClassDef)}
methods=lambda c:{v.name:v for v in c.body if isinstance(v,(ast.FunctionDef,ast.AsyncFunctionDef))}
ofit=methods(oc['FeatureCellSnapshot'])['fit'];nfit=methods(nc['FeatureCellSnapshot'])['fit']
oninit=methods(oc['FeatureCellBirthDeath'])['__init__'];nninit=methods(nc['FeatureCellBirthDeath'])['__init__']
ostamp=methods(oc['FeatureCellBirthDeath'])['_check_paired_average_state'];nstamp=methods(nc['FeatureCellBirthDeath'])['_check_paired_average_state']
inverse=deepcopy(new)
inverse.body=[n for n in inverse.body if not (isinstance(n,ast.FunctionDef) and n.name=='_fit_cell_count')
              and not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CELL_RESOLUTION_POLICY' for t in n.targets))]
ic={v.name:v for v in inverse.body if isinstance(v,ast.ClassDef)}
fcells=[n for n in methods(ic['FeatureCellSnapshot'])['fit'].body if isinstance(n,ast.Assign) and len(n.targets)==1 and ast.unparse(n.targets[0])=='self.cells']
assert len(fcells)==1 and ast.unparse(fcells[0].value)=='_fit_cell_count(cells, len(x), self.rank)'
fcells[0].value=deepcopy(next(n.value for n in ofit.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='self.cells'))
assert dump(methods(ic['FeatureCellSnapshot'])['fit'])==dump(ofit)
ibd=ic['FeatureCellBirthDeath']
initial=methods(ibd)['__init__']
policy_nodes=[n for n in initial.body if isinstance(n,ast.Expr) and ast.unparse(n.value)=='self.settings.update(resolution_policy=CELL_RESOLUTION_POLICY)']
assert len(policy_nodes)==1
initial.body.remove(policy_nodes[0]);assert dump(initial)==dump(oninit)
for n in ibd.body:
    if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='BACKEND_SCHEMA':
        assert isinstance(n.value,ast.Constant) and n.value.value==8;n.value=ast.Constant(value=7)
ibd.body=[deepcopy(ostamp) if isinstance(n,ast.FunctionDef) and n.name=='_check_paired_average_state' else n for n in ibd.body]
assert dump(inverse)==dump(old),'unscoped source delta'
helper=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='_fit_cell_count')
policy=next(n.value.value for n in new.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='CELL_RESOLUTION_POLICY')
assert policy=='even_fit_average_rows_per_effective_rank_floor1_v1'
ns=dict(Q=.05,Fraction=Fraction,math=math)
exec(compile(ast.Module(body=[helper,nstamp],type_ignores=[]),str(PKG/'feature_cells.py'),'exec'),ns)
cap=ns['_fit_cell_count'];check=ns['_check_paired_average_state']
cap_controls=[(128,512,8,64),(128,10000,8,128),(128,513,8,64),(128,3,2,1),
              (128,512,0,128),(64,512,8,64),(128,512,1,128),(128,400,8,50)]
for request,rows,rank,expected in cap_controls:assert cap(request,rows,rank)==expected

def fixture(n,rank,snapshot=1):
    k=cap(128,(n+1)//2,rank) if snapshot else 0
    stamp=dict(schema=1,policy='ema_support_inside_current_real_group_Q_v1',snapshot=snapshot,step=8 if snapshot else 0,
        rows=n,required=n-math.floor(.05*n),cells=k,rank=rank if snapshot else 0,groups=1 if snapshot else 0,
        calibration_rows=n//2 if snapshot else 0,chart_valid=bool(rank and snapshot),duplicate_ok=bool(snapshot),
        finite_rows=n if snapshot else 0,same_group_rows=n if rank and snapshot else 0,
        ema_eligible_rows=n if rank and snapshot else 0,coherent_rows=n if rank and snapshot else 0,
        eligible=bool(rank and snapshot))
    fitted=(n+1)//2;q=Fraction('0.05');ordinal=(fitted*(q.denominator-q.numerator)+q.denominator-1)//q.denominator
    partition=dict(rule='even_fit_score_order_statistic',q=.05,fitted_rows=fitted,ordinal=ordinal,ties='inside',categories=2*k)
    last=dict(step=stamp['step'],snapshot=snapshot,paired_average=stamp,cells=k,metric_rank=stamp['rank'],
        calibration_rows=stamp['calibration_rows'],duplicate_fraction=0.,mass_topology={'groups':stamp['groups']},
        count_categories=2*k,count_multiplicity=3*k+2,count_cutoff=.05/(3*k+2),count_partition=partition) if snapshot else {}
    state=dict(paired_average=stamp,snapshot_serial=snapshot,last=last)
    self=SimpleNamespace(N=n,settings={'cells':128,'rank':8,'paired_average_policy':stamp['policy']},paired_average=stamp)
    return self,state

accepted=[]
for n,rank,serial in ((1024,8,1),(20000,8,1),(1025,8,1),(1024,0,1),(1024,0,0)):
    self,state=fixture(n,rank,serial);check(self,state)
    accepted.append(dict(rows=n,rank=rank,snapshot=serial,cells=state['paired_average']['cells'],eligible=state['paired_average']['eligible']))
self,state=fixture(1024,8)
malformed=[]
mutations={'float_fitted_rows':lambda s:s['last']['count_partition'].update(fitted_rows=512.),
    'float_ordinal':lambda s:s['last']['count_partition'].update(ordinal=487.),
    'float_categories':lambda s:s['last']['count_partition'].update(categories=128.),
    'integer_q':lambda s:s['last']['count_partition'].update(q=0),
    'extra_partition_key':lambda s:s['last']['count_partition'].update(extra=1),
    'requested_not_actual_cells':lambda s:s['paired_average'].update(cells=128),
    'wrong_family':lambda s:s['last'].update(count_multiplicity=386),
    'wrong_cutoff':lambda s:s['last'].update(count_cutoff=.05/386)}
for name,mutate in mutations.items():
    bad=deepcopy(state);mutate(bad)
    try:check(self,bad)
    except ValueError:malformed.append(name)
    else:raise AssertionError(name+' accepted')
assert len(malformed)==8
# The actual first incompatibility guard rejects backend7 before super-load;
# actual whole-trainer execution is the API owner's separate contract scope.
state_check=methods(nc['FeatureCellBirthDeath'])['check_state']
prefix=deepcopy(state_check);prefix.body=prefix.body[:3]
superclass=ast.parse((BASE/'birth_death.py').read_text())
tensor_decl=next(n for c in superclass.body if isinstance(c,ast.ClassDef) and c.name=='ParticleBirthDeath'
                 for n in c.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='_TENSORS')
tensors=ast.literal_eval(tensor_decl.value)
exec(compile(ast.Module(body=[prefix],type_ignores=[]),str(PKG/'feature_cells.py'),'exec'),ns)
fake=SimpleNamespace(_TENSORS=tensors,BACKEND_SCHEMA=8,settings={'resolution_policy':policy},population_policy={})
base=set(tensors)|{'fill','cursor','rows_since_eval','counters','last','stream'}
extra={'backend','backend_schema','settings','sample_shape','snapshot_serial','population_policy','lineage_neighbors','paired_average'}
oldstate={k:None for k in base|extra};oldstate.update(backend='feature_cells',backend_schema=7,settings=fake.settings,population_policy={})
copy=deepcopy(oldstate)
try:ns['check_state'](fake,oldstate)
except ValueError:pass
else:raise AssertionError('backend7 accepted')
assert oldstate==copy
for p,d in maps.items():assert sha(p)==d,p
out=dict(status='PASS_STATIC_SOURCE_AND_SCALAR_STATE',owner_READY='PENDING',root_composition='PENDING',
    backend_schema=8,trainer_schema=5,policy=policy,config_delta=delta,whole_module_AST_inverse_RA8=True,
    unchanged28_modules=unchanged,changed_only_module='feature_cells.py',source_sha256=sources['feature_cells.py'],
    cap_controls=cap_controls,accepted_scalar_states=accepted,rejected_scalar_metadata=malformed,
    old_backend7_actual_early_guard_rejects_without_mutation=True,
    source_input_preseal=sha(OWNER/'SOURCE-FROZEN.json'),reviewed_sha256=maps,
    Torch_import=False,numerical_inputs_loaded=False,forwards=0,training_updates=0,draws=0,
    quality_claim=None,limitations=['Average resolution cap; no per-cell occupancy guarantee',
        'Full owner reaction and actual trainer API contracts pending; final owner/root freeze must be pinned'])
(HERE/'static-receipt.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status=out['status'],source_sha256=out['source_sha256'],reviewed_files=len(maps),receipt_sha256=sha(HERE/'static-receipt.json'))))
