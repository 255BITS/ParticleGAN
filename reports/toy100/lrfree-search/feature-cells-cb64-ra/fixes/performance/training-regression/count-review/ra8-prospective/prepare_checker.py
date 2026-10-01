"""Derive a new RA8 saved-state watcher from the immutable RA7 auditor."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
BASE=ROOT/'performance/training-regression/count-review/ra7-prospective/audit_checkpoints.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(BASE)=='bbaaad0a8b37163b38818d4bbfb813559d78deae75c5a4cea2fdfe2a34bf1070'
assert not (HERE/'audit_checkpoints.py').exists()
text=BASE.read_text().replace('RA7','RA8').replace('ra7','ra8').replace("'accepted-attempt2'","'accepted-attempt1'")
changes=[]
def replace(before,after):
    global text
    assert text.count(before)==1,(before,text.count(before))
    text=text.replace(before,after)
    changes.append(dict(before=before,after=after))
replace('f300089ce4fece3a812b890d4060e0567e833dddd236e5a955993f65ef15e1dd',
        'f2f117d650dbe8eca58c0313d07b661a25ad953bd6b3b30c99507510ca9e5949')
replace('1724fc603970338b0ae94d58d6055a734e3cd424fcd514a53bb1ca9297426b3d',
        '1292ef86f6b16d8928923fda267fb9f1efc6d8ac746a1f091a01397e032a3cf3')
replace('8961dc8596230bc0ff5f153606694aa21745e77f5577a0fc93664d9107d5b005',
        '7fc5e2f6f64ce452c255babbd9532b293a9fba8e70bb0fb1f8465e1a825c6205')
replace("assert ready['backend_schema']==6", "assert ready['backend_schema']==7")
replace('trainer_schema=5,backend_schema=6', 'trainer_schema=5,backend_schema=7')
replace("assert bd['backend']=='feature_cells' and bd['backend_schema']==6",
        "assert bd['backend']=='feature_cells' and bd['backend_schema']==7")
replace('from pathlib import Path\n', 'from pathlib import Path\nfrom types import SimpleNamespace\n')
replace("    return dict(status='VALID',package_sha256=h.hexdigest(),reviewed_hashes=checked,",
"""    assert h.hexdigest()=='da29f10340ddd0cf4c8452e234496579699bd8ea7e1154da22ffeac18796065c'
    assert (ROOT/'configs/overrides-CB64-RA8.json').read_bytes()==(ROOT/'configs/overrides-CB64-RA7.json').read_bytes()
    assert sha(ROOT/'configs/overrides-CB64-RA8.json')=='08359ed4406148faf6915144ef4265d930829de53775eeafc4f17a11b5667b4c'
    return dict(status='VALID',package_sha256=h.hexdigest(),reviewed_hashes=checked,""")
helper='''
def paired_average_audit(state, step, diagnostics):
    bd=state['birth_death'];stamp=bd['paired_average'];n=len(state['models']['prior']['z'])
    fake=SimpleNamespace(N=n,settings=bd['settings'],paired_average=stamp,
        snapshot_serial=bd['snapshot_serial'],snapshot=None,fill=bd['fill'],rows_since_eval=bd['rows_since_eval'],
        BACKEND_SCHEMA=7,_TENSORS=base_tensor_names,population_policy=bd['population_policy'])
    guard=fingerprint((fake.__dict__,stamp))
    paired_methods['_check_paired_average_state'](fake,bd)
    paired_methods['check_paired_average_step'](fake,bd,step)
    batch=state['recipe']['batch_size']
    assert n%batch==0 and stamp['step']==(step//(n//batch))*(n//batch)
    assert stamp['snapshot']==bd['snapshot_serial']==step//(n//batch)==bd['counters']['evals']
    assert bd['rows_since_eval']==batch*(step-stamp['step'])
    assert bd['fill']==min(n,batch*step)
    assert diagnostics['paired_average']==stamp and diagnostics['paired_average_age_real_rows']==bd['rows_since_eval']
    if bd['last']:assert bd['last']['paired_average']==stamp
    eligible=paired_methods['paired_average_eligible'](fake,step)
    assert eligible==bool(stamp['eligible'] and bd['fill']==n and 0<=bd['rows_since_eval']<n)
    rejected=[]
    for label in ('old-backend6','wrong-geometry-policy','boolean-step','future-step'):
        bad=dict(bd);badstamp=dict(stamp);bad['paired_average']=badstamp
        if label=='old-backend6':bad['backend_schema']=6
        elif label=='wrong-geometry-policy':badstamp['policy']='old'
        elif label=='boolean-step':badstamp['step']=False
        else:
            badstamp['step']=step+1
            bad['last']={**bd['last'],'step':step+1,'paired_average':badstamp}
        try:
            if label=='old-backend6':paired_methods['check_early_backend_schema'](fake,bad)
            elif label=='future-step':paired_methods['check_paired_average_step'](fake,bad,step)
            else:paired_methods['_check_paired_average_state'](fake,bad)
        except ValueError:rejected.append(label)
        else:raise AssertionError('incompatible paired-average state accepted: '+label)
        assert fingerprint((fake.__dict__,stamp))==guard
    assert all(type(v) in (bool,int,str) for v in stamp.values())
    return dict(stamp=dict(stamp),age_real_rows=bd['rows_since_eval'],
        age_steps=step-stamp['step'],eligible_now=eligible,derived_served_model='EMA' if eligible else 'fast',
        chart_absent_semantic_view=True,rejected_atomic_controls=rejected,
        limitation='Empirical anti-blur FIFO lease; no every-update support/stationarity/equivalence or quality certificate.')

'''
replace('\ndef audit(event,continuous):\n', '\n'+helper+'\ndef audit(event,continuous):\n')
replace("    bd=state['birth_death'];last=bd['last'];settings=bd['settings']",
        "    bd=state['birth_death'];last=bd['last'];settings=bd['settings']\n    paired=paired_average_audit(state,step,record['diagnostics']['birth_death'])")
replace("semantic_timing_fields=allowed_times,last_metadata_json_serializable=True,rng_cpu_uint8=True),",
        "semantic_timing_fields=allowed_times,last_metadata_json_serializable=True,rng_cpu_uint8=True,\n            backend7_geometry_stamp_consistent=True,FIFO_step_snapshot_history_exact=True,\n            paired_average_atomic_rejections=paired['rejected_atomic_controls']),")
replace("serving=dict(served_model='EMA' if active else 'fast',derived_from_original_predicate=True,",
        "paired_average=paired,\n        serving=dict(served_model=paired['derived_served_model'],derived_from_frozen_RA8_geometry_predicate=True,")
replace("note='Saved model tensors are fast training iterates; serving is derived by the unchanged predicate.'",
        "note='Saved tensors are FAST training iterates; serving derives from the frozen empirical geometry lease, independently of table stationarity.'")
replace("continuous=importlib.util.module_from_spec(spec);spec.loader.exec_module(continuous)",
'''continuous=importlib.util.module_from_spec(spec);spec.loader.exec_module(continuous)
feature_tree=ast.parse((PACKAGE/'feature_cells.py').read_text())
feature_class=next(n for n in feature_tree.body if isinstance(n,ast.ClassDef) and n.name=='FeatureCellBirthDeath')
names={'paired_average_eligible','_check_paired_average_state','check_paired_average_step'}
definitions=[deepcopy(n) for n in feature_class.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in definitions}==names
early=deepcopy(next(n for n in feature_class.body if isinstance(n,ast.FunctionDef) and n.name=='check_state'))
early.name='check_early_backend_schema';early.body=early.body[:3];early.decorator_list=[]
definitions.append(early)
paired_methods=dict(math=math,Q=.05,torch=torch)
exec(compile(ast.Module(body=definitions,type_ignores=[]),'<frozen-backend7-semantic-checks>','exec'),paired_methods)
base_class=next(n for n in ast.parse((PACKAGE/'birth_death.py').read_text()).body
    if isinstance(n,ast.ClassDef) and n.name=='ParticleBirthDeath')
base_tensor_names=ast.literal_eval(next(n.value for n in base_class.body if isinstance(n,ast.Assign)
    and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_TENSORS'))''')
replace("population=row.get('population'),serving=row.get('serving'),error=row.get('error')",
        "population=row.get('population'),paired_average=row.get('paired_average'),serving=row.get('serving'),error=row.get('error')")
replace("population_trace=[dict(step=r['step'],population=r.get('population'),serving=r.get('serving'),",
        "population_trace=[dict(step=r['step'],population=r.get('population'),paired_average=r.get('paired_average'),serving=r.get('serving'),")
ast.parse(text)
target=HERE/'audit_checkpoints.py';target.write_text(text)
derivation=dict(status='PREPARED',base_path=str(BASE),base_sha256=sha(BASE),new_sha256=sha(target),
    substitutions=changes,scope='RA8 saved checkpoints; same schema5 population/graph/action proof plus schema7 geometry state and serving view.',
    model_forwards=0,gradients=0,optimizer_updates=0,new_emissions=0,new_seeds=0,cuda=False)
(HERE/'DERIVATION.json').write_text(json.dumps(derivation,indent=2)+'\n')
files={str(p):sha(p) for p in (Path(__file__),target,HERE/'DERIVATION.json')}
(HERE/'CHECKER-FROZEN.json').write_text(json.dumps(dict(status='FROZEN_CPU_ONLY_AUDITOR',files=files,
    original_checker={str(BASE):sha(BASE)},prospective_ready_sha256='f2f117d650dbe8eca58c0313d07b661a25ad953bd6b3b30c99507510ca9e5949',
    lane_freeze_sha256='1292ef86f6b16d8928923fda267fb9f1efc6d8ac746a1f091a01397e032a3cf3'),indent=2)+'\n')
print(json.dumps(dict(status='FROZEN_CPU_ONLY_AUDITOR',checker_sha256=sha(target),freeze_sha256=sha(HERE/'CHECKER-FROZEN.json'))))
