"""CPU-only audit of sealed prospective RA9 checkpoints; no training/replay."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
    MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from fractions import Fraction
import sys
import time
import traceback
import torch

torch.set_num_threads(1)
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
VALIDATION = ROOT / 'validation-cb64-ra9'
RUN = VALIDATION / 'learned/training/toy/CB64-RA9'
LOG = VALIDATION / 'logs/learned-toy-CB64-RA9.log'
OUT = Path(__file__).resolve().parent / 'accepted-attempt1'
OUT.mkdir(exist_ok=True)
READY = ROOT / 'quality/ra9/READY.json'
PACKAGE = ROOT / 'pkg-CB64-RA9/particlegan'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--watch', action='store_true')
args = parser.parse_args()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
write = lambda p, value: Path(p).write_text(json.dumps(value, indent=2, allow_nan=True)+'\n')
json_equal = lambda a,b: json.dumps(a,sort_keys=True,allow_nan=True) == json.dumps(b,sort_keys=True,allow_nan=True)


def fingerprint(value):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(str((x.dtype,tuple(x.shape),str(x.device))).encode())
            h.update(x.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x,key=repr):
                h.update(repr(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode())
            for y in x:visit(y)
        else:h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


def verify_sources():
    checked = {str(READY):sha(READY),str(VALIDATION/'source-freeze.json'):sha(VALIDATION/'source-freeze.json')}
    assert checked[str(READY)] == 'ba558fff064f5e5613656a1c1550431091efc9989f4dd34f1d85a58d17d55990'
    assert checked[str(VALIDATION/'source-freeze.json')] == '09e6e11fd7c862d7d98fca9b5209e93be971d034ab0c6553cbd9a18bdeb6bec2'
    ready = read(READY)
    freeze = read(VALIDATION/'source-freeze.json')
    maps = [(ready['numerical_source_sha256'],None),(freeze['local_sources'],VALIDATION),
        (freeze['external_sources'],None)]
    for values,base in maps:
        for name,expected in values.items():
            p = Path(name) if base is None else base/name
            assert sha(p)==expected,str(p)
            checked[str(p)]=expected
    h=hashlib.sha256()
    for name,expected in ready['package_source_sha256'].items():
        p=PACKAGE/name
        assert sha(p)==expected
        h.update(name.encode()+b'\0'+p.read_bytes()+b'\0')
    assert h.hexdigest()==ready['package_sha256']==freeze['package_sha256']
    assert sha(PACKAGE/'continuous.py')=='071ebb1b8b85c2ff4166557499dc2b71b5ec41e62aeb9d6dfc34bedc10160195'
    assert sha(PACKAGE/'training.py')=='7fc5e2f6f64ce452c255babbd9532b293a9fba8e70bb0fb1f8465e1a825c6205'
    trainer=next(n for n in ast.parse((PACKAGE/'training.py').read_text()).body
        if isinstance(n,ast.ClassDef) and n.name=='GANTrainer')
    generate=next(n for n in trainer.body if isinstance(n,ast.FunctionDef) and n.name=='_generate')
    assert generate.args.args[5].arg=='indices' and 'rows' in [a.arg for a in generate.args.kwonlyargs]
    declaration=read(VALIDATION/'screens/DECLARED-API-EXPECTATION.json')
    assert declaration['exact_change']=='collect.expected_options.evaluation_generate: plain -> indexed'
    assert declaration['quality_gates_unchanged'] and declaration['all_other_collector_checks_exact']
    assert declaration['declared_before_numerical_execution']
    assert ready['backend_schema']==8 and ready['trainer_schema']==5
    base_config=read(ROOT/'configs/overrides-CB64-RA6.json')
    current_config=read(ROOT/'configs/overrides-CB64-RA9.json')
    assert base_config.keys()==current_config.keys()
    assert {k for k in current_config if current_config[k]!=base_config[k]}=={'lr','prior_lr_mult','d_lr_mult','birth_death_cells'}
    assert (current_config['lr'],current_config['prior_lr_mult'],current_config['d_lr_mult'])==(.0010625,8.,4.)
    assert current_config['lr']==base_config['lr']/4
    assert current_config['lr']*current_config['prior_lr_mult']==base_config['lr']*base_config['prior_lr_mult']==.0085
    assert current_config['lr']*current_config['d_lr_mult']==base_config['lr']*base_config['d_lr_mult']==.00425
    assert h.hexdigest()=='2d00c77ae0e7545ac253ff81aa729158d69016be82edc117e5597bdd664b86ce'
    assert (ROOT/'configs/overrides-CB64-RA9.json').read_text()==(ROOT/'configs/overrides-CB64-RA8.json').read_text().replace('"birth_death_cells": 64','"birth_death_cells": 128')
    assert sha(ROOT/'configs/overrides-CB64-RA9.json')=='b3656ea7494413106484c556e53877900dd5ccebf2597b3f80b7d0e7d5ba4437'
    return dict(status='VALID',package_sha256=h.hexdigest(),reviewed_hashes=checked,
        trainer_schema=5,backend_schema=8,named_index_api=True,population_sources_byte_exact=True,
        declared_indexed_expectation=True,all_other_gates_unchanged=True)


def events():
    values=[]
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            try:values.append(json.loads(line))
            except json.JSONDecodeError:pass
    return values


def time_fields(value,prefix='trainer'):
    result=[]
    if isinstance(value,dict):
        for key,x in value.items():
            path=prefix+'.'+str(key)
            if any(word in str(key).lower() for word in ('seconds','timing','elapsed','perf_counter')):
                result.append(path)
            result.extend(time_fields(x,path))
    elif isinstance(value,(tuple,list)):
        for index,x in enumerate(value):result.extend(time_fields(x,prefix+f'[{index}]'))
    return result



def paired_average_audit(state, step, diagnostics):
    bd=state['birth_death'];stamp=bd['paired_average'];n=len(state['models']['prior']['z'])
    fake=SimpleNamespace(N=n,settings=bd['settings'],paired_average=stamp,
        snapshot_serial=bd['snapshot_serial'],snapshot=None,fill=bd['fill'],rows_since_eval=bd['rows_since_eval'],
        BACKEND_SCHEMA=8,_TENSORS=base_tensor_names,population_policy=bd['population_policy'])
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
    for label in ('old-backend7','wrong-geometry-policy','boolean-step','future-step'):
        bad=dict(bd);badstamp=dict(stamp);bad['paired_average']=badstamp
        if label=='old-backend7':bad['backend_schema']=7
        elif label=='wrong-geometry-policy':badstamp['policy']='old'
        elif label=='boolean-step':badstamp['step']=False
        else:
            badstamp['step']=step+1
            bad['last']={**bd['last'],'step':step+1,'paired_average':badstamp}
        try:
            if label=='old-backend7':paired_methods['check_early_backend_schema'](fake,bad)
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


def audit(event,continuous):
    step=event['step'];path=RUN/f'checkpoint-{step:04d}.pt'
    # A post-save training_checkpoint event seals the completed torch.save.
    before=sha(path);saved=torch.load(path,map_location='cpu',weights_only=False)
    assert sha(path)==before
    state=saved['trainer'];record=saved['record'];cfg=read(RUN/'config.json')
    assert state['schema']==5 and state['completed_steps']==step==record['step']
    assert saved['data_position']==256*step and saved['receipt_sha256']==sha(RUN/'config.json')
    assert state['device']=='cuda:0' and state['serial_backward'] and cfg['seed']==314159
    assert cfg['package']==read(VALIDATION/'learned/INPUTS.json')['variants']['CB64-RA9']
    assert cfg['steps']==2000 and json_equal(cfg['recipe'],state['recipe'])
    assert cfg['recipe']['serve_average']==4 and cfg['recipe']['z_dim']==128 and cfg['recipe']['num_particles']==1024
    assert (cfg['recipe']['lr'],cfg['recipe']['prior_lr_mult'],cfg['recipe']['d_lr_mult'])==(.0010625,8.,4.)
    assert state['initial_lrs']==[[.0010625,.0085,.0010625],[.00425]]
    applied_rates=[[g['lr'] for g in opt['param_groups']] for opt in state['optimizers']]
    assert applied_rates==record['diagnostics']['lr']
    assert all(json_equal(event[k],v) for k,v in record.items())
    curve=next(r for r in [json.loads(line) for line in (RUN/'metrics.jsonl').read_text().splitlines() if line]
        if r['step']==step)
    assert all(json_equal(record[k],v) for k,v in curve.items())
    if step==0:
        expected=read(VALIDATION/'learned/INPUTS.json')['expected_initial_hashes']['toy']
        for role,field in (('G','initial_generator_sha256'),('D','initial_critic_sha256')):
            h=hashlib.sha256()
            for key,value in state['models'][role].items():h.update(key.encode());h.update(value.contiguous().numpy().tobytes())
            assert h.hexdigest()==cfg[field]==expected[field]
        assert hashlib.sha256(state['models']['prior']['z'].contiguous().numpy().tobytes()).hexdigest()==cfg['initial_prior_sha256']==expected['initial_prior_sha256']
    table=state['lr_settle'][0][1];n=len(state['models']['prior']['z']);required=n-math.floor(.05*n)
    assert table['population_schema']==1 and table['population_q']==.05
    assert table['population_policy']=='two_pair_participation_Q_survival_one_descent_undo_v1'
    mask=table['stationary_rows'];assert mask.shape==(n,) and mask.dtype==torch.bool
    active=table['population_active'];assert active==(table['last_decisive']==-1)
    if active:assert int(mask.sum())>=required and table['stationary_undo_s']==table['s']/continuous.SequentialSettleTest.GAMMA
    else:assert table['stationary_undo_s'] is None
    assert table['counts']['population_expiries']<=table['counts']['stationary']
    tester=continuous.SequentialSettleTest()
    tester.load_state_dict(table,state['models']['prior']['z'].numel())
    assert fingerprint(tester.state_dict())==fingerprint(table)
    guard=fingerprint(tester.state_dict());rejected=[]
    for label in ('old-law','wrong-level','mask-dtype'):
        bad=deepcopy(table)
        if label=='old-law':bad.pop('population_policy')
        elif label=='wrong-level':bad['population_q']=.051
        else:bad['stationary_rows']=bad['stationary_rows'].float()
        try:tester.load_state_dict(bad,state['models']['prior']['z'].numel())
        except ValueError:rejected.append(label)
        else:raise AssertionError('incompatible checkpoint accepted: '+label)
        assert fingerprint(tester.state_dict())==guard
    bd=state['birth_death'];last=bd['last'];settings=bd['settings']
    paired=paired_average_audit(state,step,record['diagnostics']['birth_death'])
    assert bd['backend']=='feature_cells' and bd['backend_schema']==8
    assert settings['cells']==cfg['recipe']['birth_death_cells']==128
    assert settings['resolution_policy']==paired_methods['CELL_RESOLUTION_POLICY']
    assert settings['latent_kernel']=='bounded_local_dv12_lineage'
    assert settings['novel_birth_policy']=='paired_even_real_anchor_shared_3K_plus_2_v1'
    assert settings['count_family']=='original_K_plus_support_2K_plus_global_2_common_Q_over_3K_plus_2'
    assert 'snapshot' not in bd and 'latent_geometry' not in bd and 'moved_rows' not in bd
    graph=bd['lineage_neighbors'];assert graph.shape==(n,8) and graph.dtype==torch.long
    rows=torch.arange(n)[:,None].expand_as(graph);edges=graph>=0
    assert bool(((graph>=-1)&(graph<n)).all()) and not bool(((graph==rows)&edges).any())
    packed=(rows[edges]*n+graph[edges]).sort().values;reverse=(graph[edges]*n+rows[edges]).sort().values
    assert torch.equal(packed,reverse) and len(torch.unique(packed))==len(packed)
    assert bd['counters']['ordinary_moves']==bd['counters']['moves']
    assert state['row_evidence']['counters']['resets']==bd['counters']['moves']+bd['counters']['iso_moves']
    assert state['row_evidence']['counters']['updates']==step
    allowed_times=['trainer.birth_death.last.eval_seconds'] if last else []
    assert time_fields(state)==allowed_times,time_fields(state)
    json.dumps(last,allow_nan=True)
    phase=None
    if last:
        k=last['cells'];assert k==paired_methods['_fit_cell_count'](settings['cells'],(n+1)//2,last['metric_rank'])
        assert last['count_multiplicity']==3*k+2 and last['count_cutoff']==.05/(3*k+2)
        assert last['count_categories']==2*k and last['ordinary_budget']==math.floor(.05*n)
        copies=last['ordinary_mass_moves']+last['ordinary_support_moves']+last['ordinary_global_moves']
        born=last['ordinary_novel_birth_moves'];assert copies==last['ordinary_copy_moves']
        assert copies+born==last['ordinary_moves']<=math.floor(.05*n)
        assert last['moves']==copies+born+last['iso_moves']
        novel=last['novel_birth'];assert novel['moves']==born<=4 and novel['paired_ema_births']==born
        children=novel['child_rows'];seeds=novel['source_seed_rows']
        assert len(children)==len(seeds)==born and len(set(children+seeds))==2*born
        assert all(0<=i<n for i in children+seeds) and novel['supported_copy_parent_rows']==[]
        assert all(c%2==0 for c in novel['new_latent_destination_categories'])
        assert all(c%2==1 for c in novel['death_category_ids'])
        assert not children or bool((graph[children]==-1).all())
        before_cert,after_cert=novel['certificates_before'],novel['certificates_after']
        for kind in ('death','birth'):
            assert after_cert['spent_'+kind]==before_cert['spent_'+kind]+born
            assert after_cert['raw_'+kind]==before_cert['raw_'+kind]
            assert after_cert['residual_'+kind]==max(0,before_cert['residual_'+kind]-born)
            assert born<=before_cert['residual_'+kind]
            assert last['ordinary_global_moves']<=after_cert['residual_'+kind]
        for acceptance in novel['acceptance']:
            assert acceptance['current_p']>.05 and acceptance['paired_ema_p']>.05
            assert 0<=acceptance['current_linearizations']<=4 and 0<=acceptance['paired_ema_linearizations']<=4
        phase={key:last[key] for key in ('step','ordinary_mass_moves','ordinary_support_moves','ordinary_global_moves',
            'ordinary_novel_birth_moves','ordinary_copy_moves','ordinary_moves','iso_moves','moves')}
    rng={key:state[key] for key in ('cpu_rng','cuda_rng')}
    rng.update({'streams.'+key:value for key,value in state['streams'].items()});rng['birth_death.stream']=bd['stream']
    assert all(v.device.type=='cpu' and v.dtype==torch.uint8 for v in rng.values())
    participants=lambda pairs: int((torch.isfinite(torch.stack(pairs)).sum(0)>=2).sum()) if pairs else 0
    ev=state['row_evidence'];neff=ev['W'].square()/ev['S'].clamp_min(1e-30)
    result=dict(status='VALID',scope='sealed prospective saved endpoint; no model/optimizer update',step=step,
        checkpoint_sha256=before,receipt_sha256=sha(RUN/'config.json'),
        checks=dict(sealed_post_save_log=True,record_and_metrics_exact=True,matched_sources_and_init=True,
            genuine_quarter_generator_and_sigma_base_rates=True,prior_and_critic_base_rates_exact=True,
            population_state_load_exact=True,old_law_atomic_rejections=rejected,
            population_stamp_and_coverage_consistent=True,moved_row_reset_ledger_exact=True,
            graph_bounded_symmetric=True,novel_children_without_seed_links=True,
            common_count_family_and_phase_budget=True,paired_birth_support_and_true_certificate_accounting=True,
            semantic_timing_fields=allowed_times,last_metadata_json_serializable=True,rng_cpu_uint8=True,
            backend8_geometry_stamp_and_actual_cap_consistent=True,FIFO_step_snapshot_history_exact=True,
            paired_average_atomic_rejections=paired['rejected_atomic_controls']),
        population=dict(active=active,mask_rows=int(mask.sum()),required_rows=required,s=table['s'],b=table['b'],
            last_decisive=table['last_decisive'],undo_s=table['stationary_undo_s'],
            coverage_rejections=table['counts']['population_coverage_rejections'],expiries=table['counts']['population_expiries'],
            accepted_stationary=table['counts']['stationary'],rebases=table['counts']['rebases'],
            last_population=table['last_population'],last_decision=table['last'],log=table['log'],
            current_window_participants_b=participants(table['r_b']),current_window_participants_2b=participants(table['r_2b'])),
        paired_average=paired,
        serving=dict(served_model=paired['derived_served_model'],derived_from_frozen_RA9_geometry_predicate=True,
            average_rate=table['s']/(cfg['recipe']['serve_average']*table['b']),
            note='Saved tensors are FAST training iterates; serving derives from the frozen empirical geometry lease, independently of table stationarity.'),
        base_learning_rates=state['initial_lrs'],applied_learning_rates=applied_rates,
        birth_counters=bd['counters'],last_phase=phase,
        row_evidence=dict(mature_rows=int((neff>=384).sum()),flags=int(ev['flag'].sum()),
            neff_median=float(neff.median()),neff_max=float(neff.max())),
        graph=dict(edges=int(edges.sum())//2,max_degree=int(edges.sum(1).max())),
        metrics=record['metrics'],quality_verdict=None,
        quality_scope='Watcher records frozen metrics only; numerical queue owns strict quality adjudication.',
        overall_target='Toy AND original full Grid100 required; intermediate metrics have no acceptance verdict.',
        optimizer_updates=0,new_seeds=0,cuda_initialized=False)
    assert sha(path)==before
    return result


integrity=verify_sources()
source_path=OUT/'SOURCE-RECEIPT.json'
if source_path.exists():assert read(source_path)==integrity
else:
    write(source_path,integrity)
    write(OUT/'SOURCE-FROZEN.json',dict(status='VALID',files={str(p):sha(p) for p in
        (Path(__file__),source_path)},reviewed_hashes=integrity['reviewed_hashes']))
spec=importlib.util.spec_from_file_location('prospective_ra9_continuous',PACKAGE/'continuous.py')
continuous=importlib.util.module_from_spec(spec);spec.loader.exec_module(continuous)
feature_tree=ast.parse((PACKAGE/'feature_cells.py').read_text())
feature_class=next(n for n in feature_tree.body if isinstance(n,ast.ClassDef) and n.name=='FeatureCellBirthDeath')
names={'paired_average_eligible','_check_paired_average_state','check_paired_average_step'}
definitions=[deepcopy(n) for n in feature_class.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert {n.name for n in definitions}==names
early=deepcopy(next(n for n in feature_class.body if isinstance(n,ast.FunctionDef) and n.name=='check_state'))
early.name='check_early_backend_schema';early.body=early.body[:3];early.decorator_list=[]
definitions.append(early)
resolution_nodes=[deepcopy(n) for n in feature_tree.body if (isinstance(n,ast.FunctionDef) and n.name=='_fit_cell_count') or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CELL_RESOLUTION_POLICY' for t in n.targets))]
assert len(resolution_nodes)==2
paired_methods=dict(math=math,Q=.05,torch=torch,Fraction=Fraction)
exec(compile(ast.Module(body=resolution_nodes+definitions,type_ignores=[]),'<frozen-backend8-semantic-checks>','exec'),paired_methods)
base_class=next(n for n in ast.parse((PACKAGE/'birth_death.py').read_text()).body
    if isinstance(n,ast.ClassDef) and n.name=='ParticleBirthDeath')
base_tensor_names=ast.literal_eval(next(n.value for n in base_class.body if isinstance(n,ast.Assign)
    and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_TENSORS'))
cpu_rng=torch.get_rng_state().clone();processed=set()
while True:
    sealed=[e for e in events() if e.get('event')=='training_checkpoint']
    for event in sealed:
        step=event['step']
        if step in processed:continue
        target=OUT/f'checkpoint-{step:04d}.json'
        if target.exists():
            assert read(target).get('checkpoint_sha256')==sha(RUN/f'checkpoint-{step:04d}.pt')
            processed.add(step);continue
        verify_sources()
        try:row=audit(event,continuous)
        except Exception as error:row=dict(status='INVALID',step=step,error=str(error),traceback=traceback.format_exc())
        assert torch.equal(cpu_rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
        write(target,row)
        write(OUT/f'checkpoint-{step:04d}-FROZEN.json',dict(status=row['status'],
            files={str(p):sha(p) for p in (target,Path(__file__),source_path)},
            artifacts={str(RUN/f'checkpoint-{step:04d}.pt'):sha(RUN/f'checkpoint-{step:04d}.pt'),
                str(RUN/'config.json'):sha(RUN/'config.json')}))
        print(json.dumps(dict(event='sealed_checkpoint_audit',step=step,status=row['status'],
            population=row.get('population'),paired_average=row.get('paired_average'),serving=row.get('serving'),error=row.get('error'))),flush=True)
        processed.add(step)
    rows=[read(OUT/f'checkpoint-{step:04d}.json') for step in sorted(processed)]
    write(OUT/'summary.json',dict(status='INVALID' if any(r['status']=='INVALID' for r in rows) else 'VALID',
        completed_checkpoint_audits=len(rows),sealed_steps=sorted(processed),
        population_trace=[dict(step=r['step'],population=r.get('population'),paired_average=r.get('paired_average'),serving=r.get('serving'),
            base_learning_rates=r.get('base_learning_rates'),applied_learning_rates=r.get('applied_learning_rates'),
            metrics=r.get('metrics'),quality_verdict=r.get('quality_verdict')) for r in rows],
        source_integrity=integrity,cpu_only=True,cuda_initialized=False,training_updates=0))
    if not args.watch or 2000 in processed or any(e.get('event')=='error' for e in events()):break
    time.sleep(10)
