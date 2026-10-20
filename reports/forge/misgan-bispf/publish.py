"""Export compact, source-bound summaries and the one current research board.

No training, regrading, new sampling or global inventory/memory regeneration.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    args=parser.parse_args()
    destination=Path(__file__).resolve().parent
    media=destination/'media';media.mkdir(exist_ok=True)
    rows=[];errors=[]
    for directory in sorted(args.queue.glob('misgan-*/*')):
        if not directory.is_dir() or not (directory/'result.json').is_file():
            continue
        result=json.loads((directory/'result.json').read_text())
        wrapper=json.loads((directory/'request.json').read_text())
        request=wrapper['request']
        identity=dict(attempt_id=directory.name,candidate_id=request['candidate']['id'],
                      candidate_revision=request['candidate_revision'],
                      source_digest=request['source']['digest'],source_commit=request['source'].get('origin_commit'),
                      request_sha256=digest(directory/'request.json'),result_sha256=digest(directory/'result.json'),
                      artifact_directory=str(directory.resolve()),paid_seconds=result.get('raw',{}).get('elapsed_seconds',0.))
        if not (directory/'api/receipt.json').is_file():
            errors.append(dict(**identity,status='ERROR',optimizer_updates=0))
            continue
        api=json.loads((directory/'api/receipt.json').read_text())
        adapter=json.loads((directory/'adapter-receipt.json').read_text())
        observations=[row for row in api['observations'] if row['step']>0 and row['step'] in api['protocol']['metric_evaluation_steps']]
        terminal=observations[-api['protocol']['terminal_observations']:]
        gif_name=request['candidate']['id']+'.gif'
        shutil.copyfile(directory/'api/goal.gif',media/gif_name)
        row=dict(**identity,study=request.get('study',{}).get('id'),
                 api_receipt_sha256=digest(directory/'api/receipt.json'),
                 adapter_receipt_sha256=digest(directory/'adapter-receipt.json'),
                 recipe=api['recipe'],protocol=api['protocol'],runtime=api['runtime'],
                 completed_updates=api['completed_updates'],verdict=api['verdict'],
                 forge_gate=result['task_results'][0]['gate_status'],
                 failed_bounds=api['failed_bounds'],final=observations[-1]['metrics'],
                 terminal=[dict(step=o['step'],passed=o['passed'],metrics=o['metrics'],failed_bounds=o['failed_bounds']) for o in terminal],
                 passed_terminal_checks=sum(o['passed'] for o in terminal),
                 reproducibility=api['reproducibility'],guards=adapter['evidence']['guards'],
                 gif=dict(path='media/'+gif_name,sha256=digest(media/gif_name),frames=api['gif_frames']),
                 checkpoint=api['artifacts']['final-state.pt'])
        rows.append(row)
    # Keep an honest stable comparison: full-gate status first, then the frozen
    # branch-mass metric, with the candidate name providing a deterministic tie.
    rows.sort(key=lambda row:(row['forge_gate']!='PASS',row['final']['ambiguous_mass_tv'],row['candidate_id']))
    initial={json.dumps(row['reproducibility']['initial_models'],sort_keys=True) for row in rows}
    batches={row['reproducibility']['batch_sequence_sha256'] for row in rows}
    output=dict(schema='misgan_bispf_research_results_v1',qualification_input=False,
                historical_100_mode_pass_credit=False,task_id='model_misgan_four_mode',
                within_mode_covariance_calibration_claim=False,rows=rows,infrastructure_errors=errors,
                matched_initial_models=len(initial)==1,matched_observed_batch_sequences=len(batches)==1,
                paid_seconds=sum(row['paid_seconds'] for row in rows+errors))
    (destination/'results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    lines=['# Current MisGAN research leaderboard','',
           'One fixed four-mode task and protocol; all failures retained. No ordinary or historical 100-mode qualification.','',
           '| Recipe | Forge gate | Last 5 checks | Generator HQ | Mass TV | Mask TV | Imputer accuracy | Ambiguous HQ | Branch TV | Branch coverage | GIF |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        m=row['final']
        numbers=[f'{m[key]:.4f}' for key in ('hq','mass_tv','mask_tv','imputer_accuracy','ambiguous_hq','ambiguous_mass_tv','ambiguous_query_coverage')]
        lines.append(f"| `{row['candidate_id']}` | {row['forge_gate']} | {row['passed_terminal_checks']}/5 | "+' | '.join(numbers)+f" | [actual training]({row['gif']['path']}) |")
    lines+=['',f"Charged elapsed time: {output['paid_seconds']:.3f} seconds, including {len(errors)} preserved infrastructure errors. GPU contention prevents throughput comparison.",
            f"Matched initial model hashes: {output['matched_initial_models']}; matched observed-batch digests: {output['matched_observed_batch_sequences']}.",
            '', 'Exact task, source, recipe, budget, RNG and sample-law identities are in [results.json](results.json). The gates do not certify within-mode covariance calibration.']
    (destination/'LEADERBOARD.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
