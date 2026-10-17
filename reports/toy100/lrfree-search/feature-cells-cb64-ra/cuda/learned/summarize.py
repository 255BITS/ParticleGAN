"""Numeric learned-lane report and manifest from saved CUDA evidence only."""
import common
from common import ROOT, DEVICE, CHECKPOINTS, PROBLEMS, VARIANTS, sha, verify_inputs, write_json, utc_now, require
from functools import lru_cache
import json
from pathlib import Path


@lru_cache(None)
def artifact_sha(path):
    return sha(path)


def read(path):
    return json.loads(Path(path).read_text())


def number(value,digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def activity(record):
    bd=record.get('diagnostics',{}).get('birth_death',{})
    counters=bd.get('counters',{})
    isolation=counters.get('iso_moves',0)
    return dict(ordinary_moves=counters.get('ordinary_moves',counters.get('moves',0)-isolation),
                isolation_moves=isolation,moves=counters.get('moves',0),cell_evals=counters.get('cell_evals'),
                counters=counters,last=bd.get('last',{}),backend=bd.get('backend','knn_beta'),
                row_evidence=record.get('diagnostics',{}).get('row_evidence'))


def toy_gate(metrics):
    checks=dict(precision=metrics['precision']>=.9,coverage=metrics['coverage']==25,mass_tv=metrics['mass_tv']<=.1)
    return dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
                thresholds=dict(precision_min=.9,coverage=25,mass_tv_max=.1),scope='independent learned toy only')


def main():
    inputs=verify_inputs()
    training={};time_comparisons={};replays={};report=[]
    training_evidence_complete=True
    for problem in PROBLEMS:
        training[problem]={}
        for variant in VARIANTS:
            outdir=ROOT/'training'/problem/variant
            path=outdir/'result.json'
            if path.exists():
                result=read(path);receipt=result.get('receipt',{});issues=[]
                initial={k:receipt.get(k) for k in inputs['expected_initial_hashes'][problem]}
                if initial!=inputs['expected_initial_hashes'][problem]:issues.append('initial hash receipt differs from GPU E22')
                if receipt.get('package')!=inputs['variants'][variant]:issues.append('source/config receipt differs')
                if result.get('steps')!=2000 or result.get('final',{}).get('step')!=2000:issues.append('fixed update budget incomplete')
                if result.get('device')!=DEVICE:issues.append('device differs')
                if receipt.get('source_freeze_sha256')!=sha(ROOT/'SOURCE-FREEZE.json'):issues.append('lane source freeze differs')
                for step in CHECKPOINTS:
                    name=f'checkpoint-{step:04d}.pt';checkpoint=outdir/name
                    if not checkpoint.exists():issues.append(f'checkpoint absent: {name}')
                    elif result.get('checkpoint_sha256',{}).get(name)!=artifact_sha(str(checkpoint)):
                        issues.append(f'checkpoint byte receipt differs: {name}')
                curves=[json.loads(line) for line in (outdir/'metrics.jsonl').read_text().splitlines() if line.strip()]
                if [r['step'] for r in curves]!=list(CHECKPOINTS):issues.append('metric checkpoint schedule differs')
                status='INVALID' if issues else result.get('status','COMPLETE')
                entry=dict(status=status,evidence_issues=issues,result=str(path),result_sha256=artifact_sha(str(path)),
                           training_seconds=result['training_seconds'],updates_per_second=result['updates_per_second'],
                           whole_run_seconds=result['whole_run_seconds'],process_seconds=result.get('process_seconds'),
                           peak_gpu_allocated_bytes=result['peak_gpu_allocated_bytes'],peak_gpu_reserved_bytes=result['peak_gpu_reserved_bytes'],
                           peak_cpu_rss_bytes=result['peak_cpu_rss_bytes'],initial_hashes=initial,final=result['final'],
                           birth_death_activity=activity(result['final']),curves=curves,runtime=receipt['runtime'],
                           config_sha256=artifact_sha(str(outdir/'config.json')))
                if problem=='toy':entry['toy_gate']=toy_gate(result['final']['metrics']) if status=='COMPLETE' else None
                else:
                    entry['evaluator']=read(outdir/'evaluator.json')
                    entry['image_acceptance_gate']=None
            elif (outdir/'error.json').exists():
                error=read(outdir/'error.json')
                entry=dict(status='ERROR',error=error,error_receipt=str(outdir/'error.json'),
                           error_receipt_sha256=artifact_sha(str(outdir/'error.json')))
            else:entry=dict(status='PENDING')
            training[problem][variant]=entry
            if entry['status']!='COMPLETE':training_evidence_complete=False
        if all(training[problem][v]['status']=='COMPLETE' for v in VARIANTS):
            budget=min(training[problem][v]['training_seconds'] for v in VARIANTS)
            chosen={}
            for variant in VARIANTS:
                eligible=[r for r in training[problem][variant]['curves'] if r['training_seconds']<=budget]
                checkpoint=max(eligible,key=lambda r:r['step'])
                chosen[variant]=dict(step=checkpoint['step'],training_seconds=checkpoint['training_seconds'],
                                     unused_training_seconds=budget-checkpoint['training_seconds'],
                                     metrics=checkpoint['metrics'],birth_death_activity=activity(checkpoint))
            time_comparisons[problem]=dict(common_training_seconds=budget,checkpoints=chosen)
        else:time_comparisons[problem]=dict(status='UNAVAILABLE',reason='both complete valid CUDA runs required')
    for variant in VARIANTS:
        path=ROOT/f'replay-{variant}.json'
        entries=read(path) if path.exists() else {}
        replays[variant]={}
        for problem in PROBLEMS:
            entry=entries.get(problem,dict(status='PENDING'))
            issues=[]
            if entry['status']=='PASS':
                checkpoint=Path(entry['checkpoint'])
                if entry['checkpoint_sha256']!=artifact_sha(str(checkpoint)):issues.append('source checkpoint bytes differ')
                if not entry.get('losses_bit_identical') or not entry.get('semantic_state_bit_identical'):
                    issues.append('required endpoint loss/state equality absent')
                if not entry.get('restoration_semantic_bit_identical'):issues.append('exact semantic restoration absent')
                if entry.get('excluded_observational_fields')!=['birth_death.last.eval_seconds']:issues.append('unapproved equality exclusions')
                if entry.get('steps_replayed')!=10 or entry.get('start_step')!=1000:issues.append('continuation budget/cursor differs')
                if entry.get('device')!=DEVICE:issues.append('replay device differs')
                if len(entry.get('branches',[]))!=2 or len(entry.get('per_update_comparison',[]))!=10:
                    issues.append('branch/per-update evidence incomplete')
                for branch in entry.get('branches',[]):
                    endpoint=Path(branch['endpoint'])
                    if not endpoint.exists() or artifact_sha(str(endpoint))!=branch['endpoint_sha256']:
                        issues.append('endpoint byte receipt differs')
            replays[variant][problem]=dict(**entry,evidence_issues=issues)
            if issues:replays[variant][problem]['status']='INVALID'
    candidate_activity={problem:training[problem]['CB64-RA'].get('birth_death_activity') for problem in PROBLEMS}
    candidate_ordinary_moves=sum((value or {}).get('ordinary_moves',0) for value in candidate_activity.values())
    replay_complete=all(replays[v][p]['status']=='PASS' for v in VARIANTS for p in PROBLEMS)
    results=dict(generated_at=utc_now(),scope='learned-model CUDA and saved-state replay lane only',device=DEVICE,
                 source_freeze_sha256=sha(ROOT/'SOURCE-FREEZE.json'),inputs_sha256=sha(ROOT/'INPUTS.json'),
                 training_evidence_complete=training_evidence_complete,replay_correctness_passed=replay_complete,
                 training=training,matched_training_time=time_comparisons,replays=replays,
                 candidate_ordinary_activity=dict(observed_in_learned_lane=candidate_ordinary_moves>0,
                    ordinary_moves_total=candidate_ordinary_moves,by_problem=candidate_activity,
                    cross_lane_requirement='root combines this activity with declared original-harness tasks'),
                 canonical_harness_verdict='supplied by original-harness lane and root report',
                 cpu_study='archived read-only diagnostic evidence; no CUDA verdict inherited',
                 image_quality_gate=None)
    write_json(ROOT/'results.json',results)
    report.append('# CB64-RA learned CUDA retest\n')
    report.append(f'Valid completed learned runs: {sum(training[p][v]["status"]=="COMPLETE" for p in PROBLEMS for v in VARIANTS)}/4. '
                  f'Exact semantic/loss replay checks: {sum(replays[v][p]["status"]=="PASS" for v in VARIANTS for p in PROBLEMS)}/4. '
                  'These results cover the learned lane. Original frozen portability/native acceptance is reported by root.\n')
    report.append('## Nonlinear toy at 2000 updates\n')
    report.append('| Variant | Evidence | Precision | Modes /25 | Mass TV | Toy gate | CUDA train s | Updates/s | Ordinary / isolation moves |\n'
                  '|---|---|---:|---:|---:|---|---:|---:|---:|')
    for v in VARIANTS:
        row=training['toy'][v];m=row.get('final',{}).get('metrics',{});a=row.get('birth_death_activity',{})
        report.append(f'| {v} | {row["status"]} | {number(m.get("precision"))} | {m.get("coverage","—")} | '
            f'{number(m.get("mass_tv"))} | {(row.get("toy_gate") or {}).get("status","—")} | '
            f'{number(row.get("training_seconds"),2)} | {number(row.get("updates_per_second"),2)} | '
            f'{a.get("ordinary_moves","—")} / {a.get("isolation_moves","—")} |')
    report.append('\nIndependent toy gate: precision>=.9, 25 supported modes, massTV<=.1. Clean particle-centre metrics and complete mass vectors are in results.json.\n')
    report.append('## MNIST at 2000 updates\n')
    report.append('| Variant | Evidence | Raw FD | Active FD | Active precision | Active recall | Class TV | Confident classes /10 | CUDA train s | Ordinary / isolation moves |\n'
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for v in VARIANTS:
        row=training['mnist'][v];m=row.get('final',{}).get('metrics',{});raw=m.get('raw_embedding',{});active=m.get('active_embedding',{});a=row.get('birth_death_activity',{})
        report.append(f'| {v} | {row["status"]} | {number(raw.get("embedding_frechet"))} | '
            f'{number(active.get("embedding_frechet"))} | {number(active.get("embedding_precision"))} | '
            f'{number(active.get("embedding_recall"))} | {number(m.get("class_mass_tv"))} | '
            f'{m.get("confident_class_coverage","—")} | {number(row.get("training_seconds"),2)} | '
            f'{a.get("ordinary_moves","—")} / {a.get("isolation_moves","—")} |')
    report.append('\nThe classifier/embeddings are trained only on real MNIST; raw64 and real-training-active standardized features use the declared first5000-image rule. '
                  'FD is learned embedding Frechet distance. No numerical image quality gate was declared. Confidence, clipping, class masses, normalization hashes and heldout controls are retained in results.json.\n')
    if all(training['mnist'][v]['status']=='COMPLETE' for v in VARIANTS):
        e=training['mnist']['E22']['final']['metrics'];c=training['mnist']['CB64-RA']['final']['metrics']
        report.append('CB64-RA minus E22 at equal updates: active FD '
            f'{c["active_embedding"]["embedding_frechet"]-e["active_embedding"]["embedding_frechet"]:+.4f}, '
            f'active precision {c["active_embedding"]["embedding_precision"]-e["active_embedding"]["embedding_precision"]:+.4f}, '
            f'active recall {c["active_embedding"]["embedding_recall"]-e["active_embedding"]["embedding_recall"]:+.4f}, '
            f'classTV {c["class_mass_tv"]-e["class_mass_tv"]:+.4f}. Lower FD/TV and higher precision/recall are favorable; assess quality and diversity together.\n')
    report.append('## Common contemporary CUDA training time\n')
    report.append('| Problem | Variant | Common budget s | Selected update | Used train s | Unused s | Precision / active FD | Coverage / active recall | Mass TV / class TV |\n'
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for p in PROBLEMS:
        comparison=time_comparisons[p]
        if 'checkpoints' not in comparison:
            report.append(f'| {p} | — | — | — | — | — | — | — | — |');continue
        for v in VARIANTS:
            r=comparison['checkpoints'][v];m=r['metrics'];active=m.get('active_embedding',{})
            quality=m['precision'] if p=='toy' else active['embedding_frechet']
            diversity=m['coverage'] if p=='toy' else active['embedding_recall']
            tv=m['mass_tv'] if p=='toy' else m['class_mass_tv']
            report.append(f'| {p} | {v} | {number(comparison["common_training_seconds"],2)} | {r["step"]} | '
                f'{number(r["training_seconds"],2)} | {number(r["unused_training_seconds"],2)} | '
                f'{number(quality)} | {number(diversity)} | {number(tv)} |')
    report.append('\nTimers synchronize around complete trainer updates/controllers and exclude batch staging, evaluation and checkpoint I/O. '
                  'Checkpoint selection uses the minimum final training duration; unused budget comes from checkpoint granularity. GPU0 may share other work. Archived timings are noncontemporary.\n')
    report.append('## State continuation\n')
    report.append('| Problem | Variant | Status | All loss bits | Every semantic section | Exact restoration |\n'
                  '|---|---|---|---|---|---|')
    for p in PROBLEMS:
        for v in VARIANTS:
            r=replays[v][p]
            report.append(f'| {p} | {v} | {r["status"]} | {r.get("losses_bit_identical","—")} | '
                f'{all(r["semantic_sections_bit_identical"].values()) if "semantic_sections_bit_identical" in r else "—"} | '
                f'{r.get("restoration_semantic_bit_identical","—")} |')
    report.append('\nEach check comprises two saved CUDA checkpoint1000 continuations through1010. Every returned loss tensor and every semantic state section are compared after every update; '
                  'global CPU/CUDA RNG, trainer streams, all controllers and optimizer states are included. Only birth_death.last.eval_seconds is excluded. '
                  'Full endpoints and loss tensors are retained; CPU RNG buffers remain on CPU.\n')
    report.append('## Activity and resources\n')
    report.append(f'Candidate ordinary moves observed in learned runs: {candidate_ordinary_moves}. '
                  'Full ordinary reaction/isolation/parent selection diagnostics are retained at every checkpoint. '
                  'The required ordinary activity across declared training or frozen tasks is combined by root. '
                  'Row-evidence asymptotic effective n99 remains below required384; observed fractions/counters are retained.\n')
    report.append('| Problem | Variant | Allocated peak MiB | Reserved peak MiB | CPU RSS MiB | Whole run s | Process s |\n'
                  '|---|---|---:|---:|---:|---:|---:|')
    for p in PROBLEMS:
        for v in VARIANTS:
            r=training[p][v]
            mib=lambda key:None if r.get(key) is None else r[key]/(1024**2)
            report.append(f'| {p} | {v} | {number(mib("peak_gpu_allocated_bytes"),1)} | '
                f'{number(mib("peak_gpu_reserved_bytes"),1)} | {number(mib("peak_cpu_rss_bytes"),1)} | '
                f'{number(r.get("whole_run_seconds"),2)} | {number(r.get("process_seconds"),2)} |')
    report.append('\nCUDA memory peaks include setup/evaluator/checkpoint state across the process. PhysicalGPU0 only, fraction.2, CPU threads2, deterministic algorithms, serialized backward and TF32 disabled.\n')
    errors=[]
    for p in PROBLEMS:
        for v in VARIANTS:
            for lane,row in (('training',training[p][v]),('replay',replays[v][p])):
                if row['status'] in ('ERROR','INVALID','FAIL','PENDING'):
                    error=row.get('error',{})
                    reason=(error.get('error') if isinstance(error,dict) else error) or '; '.join(row.get('evidence_issues',[])) or row.get('status')
                    errors.append(f'- {lane} {p}/{v}: {row["status"]}; {reason}')
    if errors:
        report.append('## Errors or incomplete correctness evidence\n');report.extend(errors);report.append('')
    report.append('Frozen inputs and local source hashes verified against this lane freeze at reporting time. '
                  'Source/data/config/checkpoint/endpoint receipts are in results.json and artifact-manifest.json. '
                  'The earlier CPU study remains archived with its device scope; its quality failures supply no canonical CUDA verdict.\n')
    (ROOT/'REPORT.md').write_text('\n'.join(report))
    manifest={}
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and path.name not in ('.gpu0.lock','artifact-manifest.json') and not path.name.endswith('.tmp'):
            manifest[str(path.relative_to(ROOT))]=dict(sha256=artifact_sha(str(path)),bytes=path.stat().st_size)
    write_json(ROOT/'artifact-manifest.json',dict(generated_at=utc_now(),scope='learned owned directory',files=manifest,
                                                read_only_file_sha256=inputs['read_only_file_sha256']))
    print(json.dumps(dict(event='learned_report_complete',training_evidence_complete=training_evidence_complete,
                          replay_correctness_passed=replay_complete,candidate_ordinary_moves=candidate_ordinary_moves,
                          report=str(ROOT/'REPORT.md'),results=str(ROOT/'results.json'),manifest=str(ROOT/'artifact-manifest.json'))),flush=True)


if __name__=='__main__':main()
