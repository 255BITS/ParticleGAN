"""Original RA10 Boolean gate authority, applied only to closed RA11 JSON."""
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

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
LANE = ROOT / 'validation-cb64-ra11'
RUN = LANE / 'screens/runs/grid100'
MONITOR = ROOT / 'integration/review/validation-cb64-ra11-monitor'
HOST = Path('/ml2/hypergan/ParticleGAN-k3p-continuous-search/benchmarks/toy100')
HARNESS = Path('/ml2/hypergan/lrfree-20260926/harness')
AUTHORITY = ROOT / 'performance/training-regression/count-review/post-ra10-quality/grid-canonical-review'
REFERENCE = ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra8-quality/ra9-grid-final-review/canonical-grid-acceptance-receipt.json'
READY_SHA = 'f7a357d0a40d353fd72bc1377240d7f89f47b34758086d284d628e35989fa5b9'
LANE_SHA = 'd973828e057a63131e6498fc3188f82b8fff3a3f58ab1ace12615566fe9837fe'
PACKAGE_SHA = '1b54cb00461df0ad89fcad59bba1e3012bf94a71ac887ecf708aafdfadc18a93'
CONFIG_SHA = 'b3656ea7494413106484c556e53877900dd5ccebf2597b3f80b7d0e7d5ba4437'

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result

def parse(text):
    def bad(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=bad)

def read(path): return parse(Path(path).read_text())
def jsonl(path): return [parse(line) for line in Path(path).read_text().splitlines() if line.strip()]
def verify(mapping):
    for path, digest in mapping.items():
        assert sha(path) == digest, path

def write(path, value):
    path = Path(path)
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def function(tree, name):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)

def assignments(tree, names):
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
    assert {target.id for node in nodes for target in node.targets if isinstance(target, ast.Name)} == set(names)
    return nodes

def pure_nodes():
    """Verbatim gate definitions selected by the accepted RA10 authority."""
    p = ast.parse((HOST / 'problems.py').read_text())
    m = ast.parse((HOST / 'metrics.py').read_text())
    a = ast.parse((HOST / 'accuracy.py').read_text())
    t = ast.parse((HOST / 'train.py').read_text())
    g = ast.parse((HOST / 'gate.py').read_text())
    ag = ast.parse((HOST / 'accuracy_gate.py').read_text())
    nodes = assignments(p, ('N_MODES', 'PROBLEM_NAMES'))
    nodes += assignments(m, ('EVAL_N', 'MIN_HQ_MODE_MASS', 'MIN_PRECISION', 'MAX_MASS_TV',
        'MAX_MODE_MASS', 'MIN_COV_EIG_RATIO', 'MAX_COV_EIG_RATIO',
        'MIN_RADIAL_MEDIAN_RATIO', 'MAX_RADIAL_MEDIAN_RATIO', 'REQUIRED_KEYS'))
    nodes += [function(m, 'passes')]
    nodes += assignments(a, ('LIMITS', 'PROTOCOL')) + [function(a, 'passes_accuracy')]
    nodes += assignments(t, ('EARLY_EVAL_STEPS',)) + [function(t, 'evaluation_steps')]
    nodes += assignments(g, ('MIN_BUDGET_STEPS', 'MAX_EVAL_INTERVAL', 'MIN_STABLE_CHECKS', 'MANDATORY_EARLY_STEPS'))
    nodes += [function(g, '_finite_tree'), function(g, '_expected_steps')]
    nodes += assignments(ag, ('HOLDOUT_N', 'HOLDOUT_SEED_OFFSETS'))
    original = function(g, 'score_run')
    start = next(index for index, node in enumerate(original.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'postinitial' for target in node.targets))
    nodes += [ast.FunctionDef(name='coverage_from_recorded',
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in
            ('rows', 'passing', 'problem', 'budget', 'audited_final_samples', 'run_dir')],
            kwonlyargs=[], kw_defaults=[], defaults=[]), body=original.body[start:], decorator_list=[])]
    screen = ast.parse((HARNESS / 'screen.py').read_text())
    nodes += assignments(screen, ('NATIVE_COVERAGE_THRESHOLDS', 'NATIVE_ACCURACY_THRESHOLDS'))
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for root in nodes for node in ast.walk(root))
    return nodes

def original_gates():
    namespace = dict(math=math, Mapping=Mapping, Path=Path)
    module = ast.fix_missing_locations(ast.Module(body=pure_nodes(), type_ignores=[]))
    exec(compile(module, '<original pure gate ASTs>', 'exec', flags=__future__.annotations.compiler_flag), namespace)
    return namespace

def metric_failures(event, thresholds):
    values = dict(event['metrics'], **{'acc_' + key: value for key, value in event['accuracy'].items()})
    result = []
    for key, relation, bound in thresholds:
        value = values[key]
        good = type(value) in (int, float) and math.isfinite(value)
        good = good and (value >= bound if relation == '>=' else value <= bound)
        if not good:
            result.append(dict(metric=key, value=value, relation=relation, bound=bound))
    return result

def verify_sources():
    seal = read(HERE / 'SOURCE-FROZEN.json')
    verify(seal['source_sha256'])
    assert sha(ROOT / 'quality/ra11/READY.json') == READY_SHA
    assert sha(LANE / 'source-freeze.json') == LANE_SHA
    return seal

def seal_sources():
    paths = {str(Path(__file__).resolve()): sha(__file__)}
    ready_path = ROOT / 'quality/ra11/READY.json'
    assert sha(ready_path) == READY_SHA
    ready = read(ready_path)
    assert (ready['backend_schema'], ready['trainer_schema']) == (10, 5)
    assert (ready['package_sha256'], ready['config_sha256']) == (PACKAGE_SHA, CONFIG_SHA)
    paths.update(ready['numerical_source_sha256'])
    frozen = read(LANE / 'source-freeze.json')
    assert sha(LANE / 'source-freeze.json') == LANE_SHA
    paths.update(frozen['external_sources'])
    paths.update({str(LANE / name): digest for name, digest in frozen['local_sources'].items()})
    for path in [ready_path, LANE / 'source-freeze.json', AUTHORITY / 'audit_grid.py',
            AUTHORITY / 'receipt.json', AUTHORITY / 'FROZEN.json', REFERENCE,
            ROOT / 'quality/ra11/lane-review/receipt.json', ROOT / 'quality/ra11/lane-review/FROZEN.json',
            ROOT / 'quality/results/RA11-grid-launch.json', Path('/home/mikkel/anaconda3/bin/python3.9')]:
        paths[str(path)] = sha(path)
    fixture = read(HARNESS / 'tasks/native100_fixture.json')
    for name, digest in fixture['host_source_sha256'].items():
        paths[str(Path(fixture['frozen_repo']) / name)] = digest
    assert read(AUTHORITY / 'receipt.json')['status'] == 'PASS'
    verify(paths)
    nodes = pure_nodes()
    compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
        '<pure gate AST inventory only>', 'exec', flags=__future__.annotations.compiler_flag)
    inventory = [dict(name=node.name if isinstance(node, ast.FunctionDef) else ','.join(t.id for t in node.targets),
        ast_sha256=hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()) for node in nodes]
    write(HERE / 'SOURCE-FROZEN.json', dict(status='FROZEN_SOURCE_ONLY_GRID_JSON_GATE_AUDIT',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=paths, pure_gate_ASTs=inventory,
        authority=str(AUTHORITY / 'receipt.json'), package_sha256=PACKAGE_SHA, config_sha256=CONFIG_SHA,
        Python='3.9.13', Torch_PT_imports=0, numerical_gate_functions_executed=0, final_artifacts_read=False))
    print(json.dumps(dict(status='FROZEN', source_guards=len(paths), gate_ASTs=len(inventory))))

def seal_inputs():
    verify_sources()
    assert not (HERE / 'closed-inputs').exists()
    slot = read(LANE / 'owned-slot-through-2.json')
    assert slot['returncode'] == 0 and slot['owned_numerical_parallelism'] == 1
    queue = jsonl(LANE / 'run.log')
    completed = [row for row in queue if row.get('event') == 'job_complete' and row.get('name') == 'screen-grid100']
    assert len(completed) == 1
    assert read(RUN / 'execution-receipt.json')['status'] == 'COMPLETE'
    assert (MONITOR / 'canonical-receipts/screens/runs/grid100/acceptance-receipt.json').is_file()
    closed = HERE / 'closed-inputs'
    closed.mkdir()
    artifacts = {str(path): sha(path) for path in RUN.rglob('*') if path.is_file() and '__pycache__' not in path.parts}
    snapshots = {'canonical-grid-acceptance-receipt.json': MONITOR / 'canonical-receipts/screens/runs/grid100/acceptance-receipt.json',
        'summary-at-grid.json': MONITOR / 'summary.json', 'checker-identity.json': MONITOR / 'CHECKER-IDENTITY.json',
        'queue-at-grid.log': LANE / 'run.log', 'slot-exit.json': LANE / 'owned-slot-through-2.json'}
    origins = {}
    for name, path in snapshots.items():
        data = path.read_bytes()
        (closed / name).write_bytes(data)
        origins[name] = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest())
        artifacts[str(closed / name)] = origins[name]['sha256']
    verify(artifacts)
    write(closed / 'INPUTS-FROZEN.json', dict(status='COMPLETED_ORIGINAL_GRID_INPUTS_FROZEN',
        utc=datetime.now(timezone.utc).isoformat(), source_freeze_sha256=sha(HERE / 'SOURCE-FROZEN.json'),
        artifact_sha256=artifacts, snapshot_origins=origins, complete_job_event=completed[0],
        PT_objects_loaded=0, array_values_decoded=0))
    print(json.dumps(dict(status='INPUTS_FROZEN', artifacts=len(artifacts))))

def audit():
    prepared = verify_sources()
    closed = HERE / 'closed-inputs'
    seal = read(closed / 'INPUTS-FROZEN.json')
    assert seal['status'] == 'COMPLETED_ORIGINAL_GRID_INPUTS_FROZEN'
    assert seal['source_freeze_sha256'] == sha(HERE / 'SOURCE-FROZEN.json')
    verify(seal['artifact_sha256'])
    gates = original_gates()
    execution = read(RUN / 'execution-receipt.json'); result = read(RUN / 'result.json')
    accepted = read(closed / 'canonical-grid-acceptance-receipt.json')
    reference = read(REFERENCE); summary = read(closed / 'summary-at-grid.json')
    # The original completed-grid predicate body is inserted below verbatim.
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
    assert result['cand']=='CB64-RA11' and result['completed_steps']==7000 and result['observations']==34
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
    checker=read(closed/'checker-identity.json')
    assert checker['global_integrity']['source_freeze_sha256']==LANE_SHA
    assert checker['collector_source_sha256']==sha(LANE/'screens/collect.py')
    assert checker['checker_source_sha256']==sha(ROOT/'integration/review/monitor_validation.py')
    assert checker['write_redirection_only'] is True and checker['original_canonical_checks_unchanged'] is True
    assert seal['complete_job_event']['event']=='job_complete' and seal['complete_job_event']['name']=='screen-grid100'
    verify_sources(); verify(seal['artifact_sha256'])
    value=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),scope='COMPLETED_ORIGINAL_GRID_JSON_AND_BOOLEAN_GATE_AUDIT',
        canonical_fixture_validity='VALID',quality_verdict=result['status'],package_sha256=PACKAGE_SHA,
        config_sha256=CONFIG_SHA,ready_sha256=READY_SHA,source_freeze_sha256=LANE_SHA,
        source_sha256=prepared['source_sha256'],artifact_sha256=seal['artifact_sha256'],
        canonical_receipt_sha256=sha(closed/'canonical-grid-acceptance-receipt.json'),
        result_sha256=sha(RUN/'result.json'),execution_receipt_sha256=sha(RUN/'execution-receipt.json'),
        terminal=terminal,holdout=dict(frozen_coverage_pass=hold['frozen_pass'],
            reconstructed_fidelity_pass=gates['passes_accuracy'](hold),combined_pass=hold_pass,metrics=hold),
        coverage=rebuilt,final=result['final'],
        checks=dict(original_canonical_VALID=True,all_source_input_hashes_exact=True,
            original_options_init_prior_stream_runtime_seeds_exact=True,all34_observations_and5terminal_clouds=True,
            original100k_holdout_declared_and_original_scorer_audit_retained=True,
            original_pure_coverage_and_fidelity_predicates_exact=True,coverage_suffix_and_final_conjunction_reconstructed=True,
            raw_canonical_quality_identical=True,saved_artifact_bytes_unchanged=True),
        cpu_only=True,Torch_imported=False,PT_objects_loaded=0,new_training_updates=0,new_draws=0,
        model_forwards=0,new_scoring_calls=0,
        limits=['Audit PASS is independent from original quality PASS/FAIL.',
            'No numerical rescore. Only the original Boolean predicates are reapplied to recorded metrics.',
            'Holdout raw coverage metrics are not fully serialized; its original frozen_pass bit is retained and hash-bound.',
            'PT and cloud bytes are hashed only. Typed final-state validity is a separate audit.',
            'No early checkpoint selection or broad screen qualification.'])
    write(HERE/'receipt.json',value)
    (HERE/'REPORT.md').write_text('# RA11 original full-grid JSON gate audit\n\n'
        f'Audit PASS / canonical VALID. Original quality {result["status"]}.\n\n'
        'All34 observations, five terminal20k clouds, full original stability and100k holdout are retained. '
        'Only original Boolean predicates were reapplied; no scoring/model/PT/array interpretation.\n\n'
        + '\n'.join(f'- Step{t["step"]}: coverage {t["coverage_pass"]}, fidelity {t["accuracy_pass"]}, '
            f'combined {t["combined_pass"]}; failed original thresholds {t["failed_original_thresholds"]}' for t in terminal)
        + f'\n\nHoldout coverage/fidelity: {hold["frozen_pass"]}/{gates["passes_accuracy"](hold)}. '
        'The holdout coverage bit remains bound to the original scorer receipt.\n')
    print(json.dumps(dict(status='PASS',quality_verdict=result['status'],canonical_fixture_validity='VALID',
        terminal=[t['combined_pass'] for t in terminal],holdout_pass=hold_pass)))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('seal-sources', 'seal-inputs', 'audit'))
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules
    assert sys.version_info[:2] == (3, 9)
    {'seal-sources': seal_sources, 'seal-inputs': seal_inputs, 'audit': audit}[args.mode]()
    assert 'torch' not in sys.modules

if __name__ == '__main__': main()
