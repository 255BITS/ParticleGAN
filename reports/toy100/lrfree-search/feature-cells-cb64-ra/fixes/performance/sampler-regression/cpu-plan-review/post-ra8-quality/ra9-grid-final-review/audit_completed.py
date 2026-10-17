"""Read-only final canonical grid acceptance and artifact/source audit."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE=Path(__file__).resolve().parent
LANE=ROOT/'validation-cb64-ra9'
RUN=LANE/'screens/runs/grid100'
MONITOR=ROOT/'integration/review/validation-cb64-ra9-monitor'
CANONICAL=MONITOR/'canonical-receipts/screens/runs/grid100/acceptance-receipt.json'
REFERENCE=ROOT/'integration/review/validation-cb64-ra8-monitor/canonical-receipts/screens/runs/grid100/acceptance-receipt.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def verify(mapping):
    for name,h in mapping.items(): assert sha(name)==h,name


def final_failures(event,thresholds):
    values={**event['metrics'],**{'acc_'+k:v for k,v in event['accuracy'].items()}}
    return [dict(metric=key,value=values[key],relation=relation,bound=bound)
        for key,relation,bound in thresholds
        if not (values[key]>=bound if relation=='>=' else values[key]<=bound)]


def main():
    assert not (HERE/'receipt.json').exists()
    verify(read(HERE/'HELPERS-FROZEN.json')['source_sha256'])
    assert sha(REFERENCE)=='fd9a4eb173c60c60b2a8cf77df5c1f42d210592c8dfda860ba38f07a3e823876'
    reference=read(REFERENCE)
    ready_path=ROOT/'quality/ra9/READY.json'
    assert sha(ready_path)=='ba558fff064f5e5613656a1c1550431091efc9989f4dd34f1d85a58d17d55990'
    ready=read(ready_path); verify(ready['numerical_source_sha256'])
    freeze_path=LANE/'source-freeze.json'
    assert sha(freeze_path)=='09e6e11fd7c862d7d98fca9b5209e93be971d034ab0c6553cbd9a18bdeb6bec2'
    frozen=read(freeze_path)
    source_map={**frozen['external_sources'],**{str(LANE/k):v for k,v in frozen['local_sources'].items()}}
    verify(source_map)
    # No completed verdict is accepted until the original wrapper and root
    # job have both exited. No invocation of the quality runner or scorer.
    queue=[json.loads(line) for line in (LANE/'run.log').read_text().splitlines() if line.strip().startswith('{')]
    assert any(row.get('event')=='job_complete' and row.get('name')=='screen-grid100' for row in queue)
    execution=read(RUN/'execution-receipt.json')
    assert execution['status']=='COMPLETE' and execution['process_exit_code']==0
    result=read(RUN/'result.json'); accepted=read(CANONICAL)
    assert accepted['result_sha256']==sha(RUN/'result.json')==execution['result_sha256']
    assert accepted['execution_receipt_sha256']==sha(RUN/'execution-receipt.json')
    assert accepted['canonical_fixture_validity']=='VALID' and not accepted['validity_reasons']
    assert accepted['legacy_fixture_comparison']=='MATCH' and not accepted['error']
    assert accepted['source_integrity']['status']=='VALID'
    for field in ('source_integrity_before','source_integrity_after'):
        value=execution[field]; assert value['status']=='VALID'
        assert value['package_sha256']==ready['package_sha256'] and value['config_sha256']==ready['config_sha256']
    assert accepted['candidate_package_sha256']==ready['package_sha256']
    assert accepted['config_sha256']==ready['config_sha256']
    assert result['cand']=='CB64-RA9' and result['recipe']['birth_death_cells']==128
    assert accepted['expected']==execution['expected']==reference['expected']
    assert accepted['thresholds']==result['thresholds']==reference['thresholds']
    assert accepted['pass_rule']==result['pass_rule']==reference['pass_rule']
    assert accepted['primary_status']==accepted['canonical_gpu_acceptance']==accepted['acceptance_status']==result['status']
    assert result['status'] in ('PASS','FAIL') and result['completed_steps']==7000
    assert accepted['completed_steps']==7000 and accepted['observations']==34
    assert accepted['native']==result['native']
    assert accepted['final']==result['final']
    assert accepted['clean_final']==result['clean_final'] and accepted['ema_final']==result['ema_final']
    assert result['eval_output_noise'] is True and result['stream_deviations']==0
    evidence=accepted['native_evidence']
    assert evidence['prior_range_match'] is True
    assert all(all(values.values()) for values in evidence['initial_parameter_match'].values())
    expected=accepted['expected']; terminal=expected['terminal_steps']
    for steps in evidence['event_steps'].values(): assert steps==expected['observation_steps']
    assert evidence['official_status']==dict(coverage=result['native']['coverage_status'],accuracy=result['native']['accuracy_status'])
    for step in terminal:
        name=f'quality_checks/step_{step:06d}.npz'
        assert all(shape==[expected['terminal_samples'],2] for shape in evidence['cloud_shapes'][name].values())
    assert len(result['native']['terminal_accuracy'])==5
    noisy=[json.loads(line) for line in (RUN/'native-noisy/events.jsonl').read_text().splitlines() if line]
    live=[event for event in noisy if event.get('event')=='eval' and event.get('model')=='live']
    assert [event['step'] for event in live]==expected['observation_steps']
    terminal_events=[event for event in live if event['step'] in terminal]
    terminal_rows=[dict(step=event['step'],coverage_pass=event['metrics']['passed'],
        accuracy_pass=event['accuracy']['passed'],failed_original_thresholds=final_failures(event,accepted['thresholds']))
        for event in terminal_events]
    assert [row['accuracy_pass'] for row in terminal_rows]==result['native']['terminal_accuracy']
    # Preserve all saved bytes. Summary/queue are evolving across later jobs,
    # so copy their current snapshots and exclude their live source from seals.
    files=sorted(p for p in RUN.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    manifest={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    snapshots={
        'canonical-grid-acceptance-receipt.json':CANONICAL,
        'grid-result.json':RUN/'result.json',
        'grid-execution-receipt.json':RUN/'execution-receipt.json',
        'summary-at-grid.json':MONITOR/'summary.json',
        'checker-identity.json':MONITOR/'CHECKER-IDENTITY.json'}
    for name,path in snapshots.items():
        copy=HERE/name; assert not copy.exists(); copy.write_bytes(path.read_bytes())
    summary=read(HERE/'summary-at-grid.json')
    summary_grid=next(row for row in summary['records'] if row['task']=='grid100')
    assert summary_grid==accepted and summary['source_integrity']['status']=='VALID'
    (HERE/'GRID-ARTIFACT-MANIFEST.json').write_text(json.dumps(dict(inputs=manifest),indent=2)+'\n')
    assert manifest=={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    verify(source_map); verify(ready['numerical_source_sha256'])
    assert sha(CANONICAL)==sha(HERE/'canonical-grid-acceptance-receipt.json')
    value=dict(status='PASS',scope='COMPLETED_CANONICAL_FIXTURE_VALIDITY_AND_VERDICT_PRESERVATION',
        utc=datetime.now(timezone.utc).isoformat(),quality_verdict=result['status'],canonical_fixture_validity='VALID',
        package_sha256=ready['package_sha256'],config_sha256=ready['config_sha256'],
        canonical_receipt_sha256=sha(CANONICAL),result_sha256=sha(RUN/'result.json'),
        execution_sha256=sha(RUN/'execution-receipt.json'),
        source_and_input_sha256={**source_map,**ready['numerical_source_sha256'],str(REFERENCE):sha(REFERENCE)},
        artifact_sha256={name:item['sha256'] for name,item in manifest.items()},
        checks=dict(original_collector_complete_VALID=True,raw_and_canonical_quality_identical=True,
            original_initializer_prior_range_data_stream_and_options_valid=True,
            all34_observation_steps_and5_terminal20k_clouds_valid=True,
            original100k_holdout_and_seeds_valid=True,all_original_thresholds_and_pass_rule_exact=True,
            configured128_used_by_actual_native_recipe=True,all_original_sources_and_saved_artifacts_unchanged=True),
        terminal=terminal_rows,holdout=result['native']['holdout'],final=result['final'],
        completed_screens=summary['completed'],total_screens=summary['total'],
        pending_tasks=[row['task'] for row in summary['records'] if row['acceptance_status']=='PENDING'],
        remaining_pending_have_no_quality_verdict=True,cpu_only=True,new_emissions=0,new_draws=0,
        new_training=0,new_scoring_calls=0,production_source_changes=0,
        limits=['This verifies stored original scorer/collector evidence; it does not rerun numerical jobs or reinterpret quality.',
            'PASS is the artifact audit status. The independent quality_verdict is exactly the original strict gate.',
            'Unrun original screens remain pending with unverified fixtures; this is not broad regression qualification.',
            'The CPU watcher can continue collecting later authorized jobs; no other queue or watcher is signaled.'])
    (HERE/'receipt.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',quality_verdict=result['status'],canonical_fixture_validity='VALID',
        canonical_receipt_sha256=value['canonical_receipt_sha256'],artifact_files=len(files))),flush=True)


if __name__=='__main__': main()
