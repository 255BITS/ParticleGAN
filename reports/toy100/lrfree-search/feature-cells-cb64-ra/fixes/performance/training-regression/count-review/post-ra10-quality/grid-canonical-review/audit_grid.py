"""Fixed stdlib-only RA10 full-grid receipt and original Boolean gate audit."""
from __future__ import annotations
import __future__
import argparse
import ast
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
LANE = ROOT / 'validation-cb64-ra10'
RUN = LANE / 'screens/runs/grid100'
MONITOR = ROOT / 'integration/review/validation-cb64-ra10-monitor'
LANE_REVIEW = ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra9-quality/ra10-lane-review'
HOST = Path('/ml2/hypergan/ParticleGAN-k3p-continuous-search/benchmarks/toy100')
HARNESS = Path('/ml2/hypergan/lrfree-20260926/harness')
CANONICAL = MONITOR / 'canonical-receipts/screens/runs/grid100/acceptance-receipt.json'
REFERENCE = ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra8-quality/ra9-grid-final-review/canonical-grid-acceptance-receipt.json'
READY_SHA = '9f29f98761b136055ef5bfce3697de0cd45cb8d12c2d65fee01e5f5281edf6fd'
LANE_SHA = '4cd1be039f2d6fb9bd12fe3d93bae60ca331bb68d77d365c9eadb4019bd8b856'
PACKAGE_SHA = 'c8f8b343bded25ce8153bdd74ab9f722b369c0e087aee8a44d3c6e816baef2e4'
CONFIG_SHA = 'b3656ea7494413106484c556e53877900dd5ccebf2597b3f80b7d0e7d5ba4437'


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        if key in result: raise ValueError('duplicate JSON key: '+key)
        result[key] = value
    return result


def parse(text):
    def bad(value): raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad)


def read(path): return parse(Path(path).read_text())
def write(name, value):
    path = HERE / name
    assert path.parent == HERE and not path.exists(), str(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify(mapping):
    for name, expected in mapping.items(): assert sha(name) == expected, name


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def assignments(tree, names):
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    found = {t.id for n in nodes for t in n.targets if isinstance(t, ast.Name)}
    assert found == set(names), (found, names)
    return nodes


def pure_nodes():
    """Select original ASTs only; never import original model/scorer modules."""
    p = ast.parse((HOST/'problems.py').read_text())
    m = ast.parse((HOST/'metrics.py').read_text())
    a = ast.parse((HOST/'accuracy.py').read_text())
    t = ast.parse((HOST/'train.py').read_text())
    g = ast.parse((HOST/'gate.py').read_text())
    ag = ast.parse((HOST/'accuracy_gate.py').read_text())
    nodes = assignments(p, ('N_MODES','PROBLEM_NAMES'))
    nodes += assignments(m, ('EVAL_N','MIN_HQ_MODE_MASS','MIN_PRECISION','MAX_MASS_TV',
        'MAX_MODE_MASS','MIN_COV_EIG_RATIO','MAX_COV_EIG_RATIO',
        'MIN_RADIAL_MEDIAN_RATIO','MAX_RADIAL_MEDIAN_RATIO','REQUIRED_KEYS'))
    nodes += [function(m,'passes')]
    nodes += assignments(a, ('LIMITS','PROTOCOL')) + [function(a,'passes_accuracy')]
    nodes += assignments(t, ('EARLY_EVAL_STEPS',)) + [function(t,'evaluation_steps')]
    nodes += assignments(g, ('MIN_BUDGET_STEPS','MAX_EVAL_INTERVAL','MIN_STABLE_CHECKS','MANDATORY_EARLY_STEPS'))
    nodes += [function(g,'_finite_tree'), function(g,'_expected_steps')]
    nodes += assignments(ag, ('HOLDOUT_N','HOLDOUT_SEED_OFFSETS'))
    # Verbatim suffix after validation, with no file/array re-score.
    original = function(g, 'score_run')
    start = next(i for i,n in enumerate(original.body) if isinstance(n,ast.Assign)
        and any(isinstance(k,ast.Name) and k.id=='postinitial' for k in n.targets))
    suffix = original.body[start:]
    reconstructed = ast.FunctionDef(name='coverage_from_recorded',
        args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in
            ('rows','passing','problem','budget','audited_final_samples','run_dir')],
            kwonlyargs=[],kw_defaults=[],defaults=[]),body=suffix,decorator_list=[])
    nodes += [reconstructed]
    mean = ast.parse((ROOT/'pkg-CB64-RA10/particlegan/mean_transport.py').read_text())
    nodes += assignments(mean, ('Q','POLICY','MEAN_POLICY','SCALAR_FIELDS','MEAN_KEYS'))
    nodes += [function(mean,'initial_mean_diagnostics'),function(mean,'check_mean_diagnostics')]
    fc = ast.parse((ROOT/'pkg-CB64-RA10/particlegan/feature_cells.py').read_text())
    cls = next(n for n in fc.body if isinstance(n,ast.ClassDef) and n.name=='FeatureCellBirthDeath')
    nodes += [function(cls,'_check_mean_actions')]
    screen = ast.parse((HARNESS/'screen.py').read_text())
    nodes += assignments(screen, ('NATIVE_COVERAGE_THRESHOLDS','NATIVE_ACCURACY_THRESHOLDS'))
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) for node in nodes for n in ast.walk(node))
    return nodes


def original_gates():
    nodes = pure_nodes()
    module = ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    ns = dict(math=math,Mapping=Mapping,Path=Path)
    exec(compile(module,'<original pure gate ASTs>','exec',flags=__future__.annotations.compiler_flag),ns)
    return ns


def verify_frozen():
    prepared = read(HERE/'HELPERS-FROZEN.json')
    verify(prepared['source_and_input_sha256'])
    verify(prepared['helper_sha256'])
    ready = read(ROOT/'quality/ra10/READY.json')
    assert sha(ROOT/'quality/ra10/READY.json') == READY_SHA
    assert sha(LANE/'source-freeze.json') == LANE_SHA
    assert ready['package_sha256']==PACKAGE_SHA and ready['config_sha256']==CONFIG_SHA
    assert ready['backend_schema']==9 and ready['trainer_schema']==5
    return prepared, ready


def jsonl(path):
    return [parse(line) for line in path.read_text().splitlines() if line.strip()]


def root_events():
    path = LANE/'run.log'
    rows=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try: rows.append(parse(line))
            except (ValueError,json.JSONDecodeError): continue  # Live queue tail only.
    return rows


def ready_to_audit(queue):
    if not any(r.get('event')=='job_complete' and r.get('name')=='screen-grid100' for r in queue): return False
    p=RUN/'execution-receipt.json'
    if not p.exists() or not CANONICAL.exists() or not (MONITOR/'summary.json').exists(): return False
    try:
        e=read(p); summary=read(MONITOR/'summary.json')
    except json.JSONDecodeError: return False
    return type(e.get('process_exit_code')) is int and any(r.get('task')=='grid100'
        and r.get('acceptance_status')!='PENDING' for r in summary.get('records',()))


def metric_failures(event, thresholds):
    values=dict(event['metrics'],**{'acc_'+k:v for k,v in event['accuracy'].items()})
    result=[]
    for key,relation,bound in thresholds:
        value=values[key]
        good=type(value) in (int,float) and math.isfinite(value)
        good=good and (value>=bound if relation=='>=' else value<=bound)
        if not good: result.append(dict(metric=key,value=value,relation=relation,bound=bound))
    return result


def audit_completed(queue):
    prepared,ready=verify_frozen(); gates=original_gates()
    files=sorted(p for p in RUN.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    manifest={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    execution=read(RUN/'execution-receipt.json'); result=read(RUN/'result.json')
    accepted=read(CANONICAL); reference=read(REFERENCE); summary=read(MONITOR/'summary.json')
    assert execution['status']=='COMPLETE' and type(execution['process_exit_code']) is int and execution['process_exit_code']==0
    assert accepted['result_sha256']==sha(RUN/'result.json')==execution['result_sha256']
    assert accepted['execution_receipt_sha256']==sha(RUN/'execution-receipt.json')
    assert accepted['canonical_fixture_validity']=='VALID' and not accepted['validity_reasons']
    assert accepted['legacy_fixture_comparison']=='MATCH' and not accepted['error']
    assert accepted['source_integrity']['status']=='VALID' and 'mechanism_collection_error' not in accepted
    for field in ('source_integrity_before','source_integrity_after'):
        s=execution[field]; assert s['status']=='VALID'
        assert s['package_sha256']==PACKAGE_SHA and s['config_sha256']==CONFIG_SHA
    assert execution['source_freeze_sha256']==sha(LANE/'screens/source-freeze.json')
    assert execution['ready_sha256']==sha(LANE/'screens/READY.json')
    assert accepted['candidate_package_sha256']==PACKAGE_SHA and accepted['config_sha256']==CONFIG_SHA
    assert result['cand']=='CB64-RA10' and result['completed_steps']==7000 and result['observations']==34
    assert type(result['completed_steps']) is int and type(result['observations']) is int
    assert result['recipe']['birth_death_cells']==128
    expected=reference['expected']; assert accepted['expected']==execution['expected']==expected
    assert expected==dict(steps=7000,num_particles=20000,z_dim=2,batch_size=2048,seed=1234,
        observation_steps=[0,1,10,25,50,100]+list(range(250,7001,250)),
        terminal_steps=[6000,6250,6500,6750,7000],terminal_samples=20000,holdout_samples=100000,
        holdout_seeds=dict(target=2835,noise=2836,latent=2837))
    thresholds=gates['NATIVE_COVERAGE_THRESHOLDS']+gates['NATIVE_ACCURACY_THRESHOLDS']
    assert accepted['thresholds']==result['thresholds']==reference['thresholds']==[list(t) for t in thresholds]
    assert accepted['pass_rule']==result['pass_rule']==reference['pass_rule']
    assert accepted['primary_status']==accepted['canonical_gpu_acceptance']==accepted['acceptance_status']==result['status']
    assert result['status'] in ('PASS','FAIL')
    assert accepted['native']==result['native'] and accepted['final']==result['final']
    assert accepted['clean_final']==result['clean_final'] and accepted['ema_final']==result['ema_final']
    assert result['eval_output_noise'] is True and result['stream_deviations']==0
    options=dict(eval_output_noise=True,strict_streams=True,save_final_state=True,diagnostics=True,
        evaluation_generate='indexed',serial_backward_argument=True,initialization='batch_feature_zero',
        image_prior_perturb=False,ring_frozen_control=False)
    assert result['header']['options']==options
    assert result['header']['package_sha256']==PACKAGE_SHA
    assert result['header']['device']=='cuda:0' and result['header']['cuda_visible_devices']=='0'
    resources=execution['resources']
    assert resources['gpu_uuid']=='GPU-72c1b506-891d-b8bc-b353-e020585e1c47'
    assert resources['physical_gpu']==0 and resources['device']=='cuda:0'
    assert resources['cuda_memory_fraction']==.2 and resources['numeric_threads']==1
    assert execution['environment']['CUDA_VISIBLE_DEVICES']=='0' and all(execution['unset_environment'].values())
    assert summary['source_integrity']['status']=='VALID'
    assert next(r for r in summary['records'] if r['task']=='grid100')==accepted
    evidence=accepted['native_evidence']; fixture=read(HARNESS/'tasks/native100_fixture.json')
    assert evidence['prior_range_match'] is True
    assert all(v is True for params in evidence['initial_parameter_match'].values() for v in params.values())
    assert evidence['fixture']==result['native_fixture']==read(RUN/'native-fixture.json')
    native=RUN/'native-noisy'; native_summary=read(native/'summary.json'); config=read(native/'config.json')
    assert native_summary['config']==config and native_summary['status']=='complete'
    budget,steps=gates['_expected_steps'](config)
    assert budget==7000 and steps==expected['observation_steps']==native_summary['eval_steps']
    assert native_summary['completed_steps']==7000 and native_summary['accuracy_check_steps']==expected['terminal_steps']
    for key in ('steps','seed','num_particles','z_dim','batch_size'): assert config[key]==expected[key]
    assert config['eval_samples']==20000 and config['threads']==1 and config['device']=='cuda:0'
    declaration=native_summary['accuracy']
    assert declaration['protocol']==gates['PROTOCOL'] and declaration['check_steps']==expected['terminal_steps']
    assert declaration['sample_count']==20000 and declaration['holdout_samples']==gates['HOLDOUT_N']==100000
    assert declaration['holdout_seed_offsets']==gates['HOLDOUT_SEED_OFFSETS']==dict(target=1601,noise=1602,latent=1603)
    for model,observed in evidence['event_steps'].items(): assert observed==steps and model in ('live','ema')
    for rel,n in [(f'quality_checks/step_{s:06d}.npz',20000) for s in expected['terminal_steps']]+[
            ('final_samples.npz',20000),('holdout_samples.npz',100000)]:
        assert set(evidence['cloud_shapes'][rel])=={'live','ema','target'}
        assert all(shape==[n,2] for shape in evidence['cloud_shapes'][rel].values())
    assert native_summary['snapshot_steps']==steps
    assert all((native/'snapshots'/f'step_{s:06d}.npz').is_file() for s in steps)
    verdict=read(native/'verdict.json'); assert verdict['sources']==fixture['host_source_sha256']
    assert evidence['coverage']==verdict['coverage'] and evidence['accuracy']==verdict['accuracy']
    events=jsonl(native/'events.jsonl'); live=[]
    for model in ('live','ema'):
        rows=[e for e in events if e.get('event')=='eval' and e.get('model')==model]
        assert [e['step'] for e in rows]==steps and all(type(e['step']) is int for e in rows)
        elapsed=-1.; previous=None
        for event in rows:
            assert type(event['elapsed']) in (int,float) and math.isfinite(event['elapsed']) and event['elapsed']>=elapsed
            elapsed=event['elapsed']; m=event['metrics']; a=event['accuracy']
            assert gates['_finite_tree'](m) and m['n']==20000
            coverage=gates['passes']('grid100',m); fidelity=gates['passes_accuracy'](a)
            assert type(m['passed']) is bool and m['passed']==coverage
            assert type(a['frozen_pass']) is bool and a['frozen_pass']==coverage
            assert type(a['accuracy_pass']) is bool and a['accuracy_pass']==fidelity
            assert type(a['passed']) is bool and a['passed']==(coverage and fidelity)
        if model=='live': live=rows
    passing=[gates['passes']('grid100',e['metrics']) for e in live]
    official_coverage=verdict['coverage']; assert official_coverage['audited_final_samples'] is True
    rebuilt=gates['coverage_from_recorded'](live,passing,'grid100',7000,True,native)
    assert rebuilt==official_coverage
    acc=verdict['accuracy']; assert len(acc['terminal_checks'])==5
    terminal=[]
    for check,step in zip(acc['terminal_checks'],expected['terminal_steps']):
        event=next(e for e in live if e['step']==step)
        assert check['step']==step and check['metrics']==event['accuracy']
        assert check['ema_metrics']==next(e for e in events if e.get('event')=='eval' and e.get('model')=='ema' and e['step']==step)['accuracy']
        passed=gates['passes']('grid100',event['metrics']) and gates['passes_accuracy'](check['metrics'])
        assert type(check['passed']) is bool and check['passed']==passed
        terminal.append(dict(step=step,coverage_pass=passing[steps.index(step)],
            accuracy_pass=gates['passes_accuracy'](check['metrics']),combined_pass=passed,
            failed_original_thresholds=metric_failures(event,thresholds)))
    hold=acc['holdout_metrics']; assert hold==native_summary['holdout']['live'] and hold['n']==100000
    for model,key in (('live','holdout_metrics'),('ema','holdout_ema_metrics'),('target','oracle_metrics')):
        h=acc[key]; assert h==native_summary['holdout'][model] and h['n']==100000
        assert type(h['frozen_pass']) is bool and type(h['accuracy_pass']) is bool and type(h['passed']) is bool
        assert h['accuracy_pass']==gates['passes_accuracy'](h)
        assert h['passed']==(h['frozen_pass'] and h['accuracy_pass'])
    assert acc['oracle_metrics']['passed'] is True
    hold_pass=hold['frozen_pass'] and gates['passes_accuracy'](hold)
    final_pass=rebuilt['passed'] and all(t['combined_pass'] for t in terminal) and hold_pass
    assert acc['passed']==final_pass and acc['status']==('PASS' if final_pass else 'FAIL')==result['status']
    assert result['native']['terminal_accuracy']==[t['combined_pass'] for t in terminal]
    assert result['native']['holdout_pass']==hold_pass
    assert result['native']['coverage_status']==rebuilt['status'] and result['final_streak']==rebuilt['stable_checks']
    assert evidence['official_status']==dict(coverage=rebuilt['status'],accuracy=acc['status'])
    bd=accepted['mechanisms']['birth_death']; last=bd['last']; paired=last['paired_average']; count=bd['counters']
    assert bd['backend']=='feature_cells' and type(bd['snapshot_serial']) is int
    assert type(last['cells']) is int and type(last['metric_rank']) is int
    k=last['cells']; assert 1<=k<=128 and 0<=last['metric_rank']<=8
    assert last['count_categories']==2*k and last['count_multiplicity']==3*k+3
    assert type(last['count_cutoff']) is float and last['count_cutoff']==.05/(3*k+3)
    assert type(last['step']) is int and last['step']<=7000 and last['snapshot']==bd['snapshot_serial']
    gates['check_mean_diagnostics'](last['mean_transport'],last=last,paired=paired,n=20000,
        snapshot_serial=bd['snapshot_serial'],dry_run=False)
    gates['_check_mean_actions'](SimpleNamespace(N=20000),dict(last=last,snapshot_serial=bd['snapshot_serial']))
    assert all(type(v) is int and v>=0 for v in count.values())
    assert count['mean_evals']==bd['snapshot_serial'] and count['mean_witness_fires']<=count['mean_evals']
    assert count['mean_moves']>=last['ordinary_mean_moves']
    checker=read(MONITOR/'CHECKER-IDENTITY.json'); owned=read(LANE_REVIEW/'MONITOR-START.json')
    assert checker['global_integrity']['source_freeze_sha256']==LANE_SHA
    assert checker['collector_source_sha256']==sha(LANE/'screens/collect.py')
    assert checker['checker_source_sha256']==sha(ROOT/'integration/review/monitor_validation.py')
    assert checker['write_redirection_only'] is True and checker['original_canonical_checks_unchanged'] is True
    assert owned['output']==str(MONITOR) and owned['validation']==str(LANE) and owned['cpu_only'] is True
    snapshots={'canonical-grid-acceptance-receipt.json':CANONICAL,'grid-result.json':RUN/'result.json',
        'grid-execution-receipt.json':RUN/'execution-receipt.json','summary-at-grid.json':MONITOR/'summary.json',
        'checker-identity.json':MONITOR/'CHECKER-IDENTITY.json','queue-at-grid.log':LANE/'run.log'}
    for name,path in snapshots.items():
        target=HERE/name; assert not target.exists(); target.write_bytes(path.read_bytes())
    verify_frozen(); verify({p:item['sha256'] for p,item in manifest.items()})
    assert manifest=={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    assert sha(CANONICAL)==sha(HERE/'canonical-grid-acceptance-receipt.json')
    write('GRID-ARTIFACT-MANIFEST.json',dict(inputs=manifest))
    value=dict(status='PASS',utc=now(),scope='COMPLETED_ORIGINAL_GRID_ARTIFACT_AND_BOOLEAN_GATE_AUDIT',
        canonical_fixture_validity='VALID',quality_verdict=result['status'],package_sha256=PACKAGE_SHA,
        config_sha256=CONFIG_SHA,ready_sha256=READY_SHA,source_freeze_sha256=LANE_SHA,
        source_and_input_sha256=prepared['source_and_input_sha256'],artifact_sha256={p:item['sha256'] for p,item in manifest.items()},
        canonical_receipt_sha256=sha(CANONICAL),result_sha256=sha(RUN/'result.json'),
        execution_receipt_sha256=sha(RUN/'execution-receipt.json'),
        terminal=terminal,holdout=dict(frozen_coverage_pass=hold['frozen_pass'],
            reconstructed_fidelity_pass=gates['passes_accuracy'](hold),combined_pass=hold_pass,metrics=hold),
        coverage=rebuilt,final=result['final'],mean_transport=last['mean_transport'],
        ordinary_phase_moves={key:last[key] for key in ('ordinary_mass_moves','ordinary_support_moves',
            'ordinary_global_moves','ordinary_novel_birth_moves','ordinary_mean_moves','ordinary_moves','iso_moves','moves')},
        checks=dict(original_canonical_VALID=True,all_source_input_hashes_exact=True,
            original_options_init_prior_stream_runtime_seeds_exact=True,all34_observations_and5terminal_clouds=True,
            original100k_holdout_declared_and_original_scorer_audit_retained=True,
            original_pure_coverage_and_fidelity_predicates_exact=True,coverage_suffix_and_final_conjunction_reconstructed=True,
            raw_canonical_quality_identical=True,mean_typed_JSON_and_phase_budget_consistent=True,
            saved_artifact_bytes_unchanged=True),
        completed_screens=summary['completed'],total_screens=summary['total'],
        pending_tasks=[r['task'] for r in summary['records'] if r['acceptance_status']=='PENDING'],
        cpu_only=True,Torch_imported=False,PT_objects_loaded=0,new_training_updates=0,new_draws=0,
        model_forwards=0,new_scoring_calls=0,signals_to_other_processes=0,
        limits=['Audit PASS is independent from original quality PASS/FAIL.',
            'No numerical rescore. Original scorer rescored saved arrays; only original Boolean predicates are re-applied here.',
            'Holdout raw coverage metrics are not fully serialized; its original frozen_pass bit is retained and hash-bound.',
            'Mean geometry/control evidence is not population equivalence, stationarity or serving quality certification.',
            'Unrun original screens remain pending and unverified; no broad qualification is claimed.'])
    write('receipt.json',value)
    (HERE/'REPORT.md').write_text('# RA10 original full-grid artifact audit\n\n'
        f'Artifact audit: PASS / VALID. Original quality: {result["status"]}.\n\n'
        'The frozen sources, fixtures, runtime, schedule and original gates agree. '
        'All34 observations, five terminal checks and100k holdout are retained. '
        'The original JSON predicates and final conjunction were independently reconstructed. '
        'No scoring, training, sample generation or PT interpretation occurred.\n\n'
        + '\n'.join(f'- Step{t["step"]}: coverage {t["coverage_pass"]}, fidelity {t["accuracy_pass"]}, '
            f'combined {t["combined_pass"]}; failed original thresholds {t["failed_original_thresholds"]}' for t in terminal)
        +f'\n\nHoldout coverage/fidelity: {hold["frozen_pass"]}/{gates["passes_accuracy"](hold)}. '
        'The holdout coverage bit is bound to the original saved-cloud scorer receipt.\n')
    print(json.dumps(dict(event='grid_audit_complete',status='PASS',quality_verdict=result['status'],
        canonical_fixture_validity='VALID',artifact_files=len(manifest))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and 'torch' not in sys.modules
    assert not (HERE/'receipt.json').exists() and not (HERE/'WATCH-START.json').exists()
    verify_frozen()
    stat=Path(f'/proc/{os.getpid()}/stat').read_text()
    command=[x.decode() for x in Path(f'/proc/{os.getpid()}/cmdline').read_bytes().split(b'\0') if x]
    write('WATCH-START.json',dict(status='CPU_STDLIB_GRID_WATCH_STARTED',utc=now(),pid=os.getpid(),
        startticks=int(stat[stat.rfind(')')+2:].split()[19]),command=command,cpu_only=True,
        output=str(HERE),source_freeze_sha256=LANE_SHA,helper_freeze_sha256=sha(HERE/'HELPERS-FROZEN.json')))
    print(json.dumps(dict(event='prepared_grid_watcher_started',source_freeze_sha256=LANE_SHA)),flush=True)
    previous=None
    try:
        while True:
            queue=root_events()
            if ready_to_audit(queue): audit_completed(queue); return
            state='WAITING_FOR_GRID_COMPLETION_AND_CANONICAL_RECEIPT'
            if queue and queue[-1].get('event')=='queue_aborted': state='ROOT_QUEUE_ABORTED_NO_COMPLETE_GRID'
            if state!=previous:
                print(json.dumps(dict(event='grid_audit_pending',status=state,quality_verdict=None)),flush=True); previous=state
            if state.startswith('ROOT_QUEUE_ABORTED') or not args.watch: return
            time.sleep(15)
    except BaseException as error:
        if not (HERE/'FAILURE.json').exists():
            write('FAILURE.json',dict(status='INVALID',utc=now(),error=repr(error),traceback=traceback.format_exc(),
                original_quality_verdict_unchanged=True,helper_sha256=sha(Path(__file__))))
        raise


if __name__=='__main__': main()
