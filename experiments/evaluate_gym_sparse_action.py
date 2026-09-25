#!/usr/bin/env python
"""Fresh paired control evaluation with five labeled expert-action episodes."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.gym_data import sha256, simulator_provenance
from lib.gym_control import build_expert_records
from lib.gym_control_evaluation import evaluate_controller, paired_comparison, selection_key
from experiments.evaluate_gym_control import write_json
from experiments.evaluate_gym_state_control import auxiliary_metrics, rollout_auxiliary, check_disjoint, verify_checkpoint as verify_full_label_checkpoint

REPORTS = ROOT/'reports/gym/lunar_lander_sparse_action'
EPISODES = ROOT/'results/gym/lunar_lander/data/episodes.json'
REFERENCE = ROOT/'results/gym/lunar_lander_state_control/probes/best.pt'


def freeze_protocol(path, episodes_path, reference):
    previous_paths = sorted((ROOT/'reports/gym').rglob('protocol.json'))
    previous_paths = [p for p in previous_paths if p.resolve() != path.resolve()]
    previous = [json.loads(p.read_text()) for p in previous_paths]
    validation, test = list(range(791000,791020)), list(range(891000,891050))
    used = check_disjoint(json.loads(episodes_path.read_text()), previous, validation, test)
    source_names = ['experiments/evaluate_gym_sparse_action.py', 'experiments/evaluate_gym_state_control.py',
        'experiments/train_gym_sparse_action.py', 'lib/gym_sparse_action.py', 'experiments/evaluate_gym_control.py',
        'experiments/train_gym_state_control.py', 'experiments/train_gym_transition.py', 'experiments/config.py',
        'lib/gym_state_control.py', 'lib/gym_control.py', 'lib/gym_control_evaluation.py',
        'lib/gym_transition.py', 'lib/gym_data.py']
    sources = [ROOT/name for name in source_names] + sorted((ROOT/'particlegan').glob('*.py'))
    configs = sorted((ROOT/'configs/gym/lunar_lander_sparse_action').glob('*.yaml'))
    from lib.gym_sparse_action import build_sparse_records, fit_sparse_scaler
    records,selection = build_sparse_records(episodes_path)
    scaler=fit_sparse_scaler(records)
    value = dict(version=1, supervision=dict(selection=selection,
        array_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in records.items()},
        normalization_statistics={k:v.cpu().tolist() for k,v in scaler.state_dict().items()},
        labeled_records=int(records['label_mask'].sum()), all_state_pairs=len(records['states']),
        rule='Five whole training episodes chosen by SHA256(lunar-sparse-actions-v1:<episode_id>); no subset repeats',
        normalization='State statistics from all available training state pairs; action statistics from labeled records only'), validation_seeds=validation, test_seeds=test,
        candidate_updates=[250,1000,2500], training_updates=2500, training_batch_size=256,
        training_config=dict(seed=24003,num_particles=1024,z_dim=32,width=128,encoder_width=128,context_dim=11,
            lambda_state=1.,lambda_next=1.,continuous_weight=1.,contact_weight=1.),
        source_episodes=str(episodes_path.resolve()), source_episodes_sha256=sha256(episodes_path),
        previously_used_reset_seeds=used, prior_protocols={str(p):sha256(p) for p in previous_paths},
        reference_checkpoint=str(reference.resolve()), reference_checkpoint_sha256=sha256(reference),
        simulator=simulator_provenance(), sources={str(p):sha256(p) for p in sources+configs},
        selection=['validation landing rate descending', 'validation mean return descending', 'earlier update exact tie'],
        success='terminated and not lander.awake; pinned simulator final sleep check takes precedence',
        landing_interval='Wilson 95% over paired evaluation episodes; no training seed repeats',
        inference='E(st, terrain) -> z -> G2 -> at; real persistent simulator.step(at)',
        inference_timing='Controller call with routing/offset diagnostics; see each controller timing_scope',
        auxiliary='Offline diagnostics on expert records and actual learner rollout successors; no extra simulator calls',
        interpretation='G3 has no action input: expert-behavior successor predictor, not a general counterfactual model',
        max_episode_steps=1000)
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError('Frozen protocol differs; use a new report directory')
    write_json(path,value)
    return value


def verify_protocol(path):
    value = json.loads(Path(path).read_text())
    if value['simulator'] != simulator_provenance():
        raise RuntimeError('Simulator changed')
    artifacts = {**value['sources'], **value['prior_protocols'],
        value['source_episodes']:value['source_episodes_sha256'],
        value['reference_checkpoint']:value['reference_checkpoint_sha256']}
    for filename,digest in artifacts.items():
        if sha256(filename) != digest:
            raise RuntimeError(f'Frozen evaluation artifact changed: {filename}')
    return value


def verify_checkpoint(bundle, checkpoint):
    p, summary = bundle['provenance'], bundle.get('training_summary')
    if not summary or sha256(checkpoint) not in summary['checkpoints'].values():
        raise RuntimeError('Checkpoint is not a hashed completed training output')
    if summary['provenance'] != p:
        raise RuntimeError('Checkpoint provenance differs from training summary')
    for filename,digest in p['sources'].items():
        if sha256(ROOT/filename) != digest:
            raise RuntimeError(f'Training source changed: {filename}')
    for key in ('episodes',):
        if sha256(ROOT/p[key]['path']) != p[key]['sha256']:
            raise RuntimeError(f'Training artifact changed: {key}')
    for filename,digest in [('source.zip',p['source_archive_sha256']), ('sparse_records.npz',p['expert_data']['npz_sha256'])]:
        if sha256(Path(checkpoint).parent/filename) != digest:
            raise RuntimeError(f'Training archive changed: {filename}')


def verify_training_protocol(bundle, protocol):
    p,cfg=bundle['provenance'],bundle['config']
    supervision=protocol['supervision']
    expected={**protocol['training_config'], 'steps':protocol['training_updates'], 'batch_size':protocol['training_batch_size']}
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise RuntimeError('Training configuration differs from frozen protocol')
    if p['selection']!=supervision['selection'] or p['labeled_data']['count']!=supervision['labeled_records']:
        raise RuntimeError('Action-label subset differs from frozen protocol')
    if p['expert_data']['count']!=supervision['all_state_pairs']:
        raise RuntimeError('State-pair count differs from frozen protocol')
    actual_hashes={k:v['sha256'] for k,v in p['expert_data']['arrays'].items()}
    if actual_hashes!=supervision['array_hashes']:
        raise RuntimeError('Training data arrays differ from frozen protocol')
    normalization={k:v.cpu().tolist() for k,v in bundle['scaler'].state_dict().items()}
    if normalization!=supervision['normalization_statistics']:
        raise RuntimeError('Normalization differs from frozen sparse supervision')


def make_controller(arm, checkpoint, device):
    if arm == 'expert':
        from gymnasium.envs.box2d.lunar_lander import heuristic
        return lambda env,s,a,c: (heuristic(env.unwrapped,s),{}), None
    from lib.gym_state_control import load_state_control_checkpoint, control_action_details
    bundle = load_state_control_checkpoint(checkpoint,device)
    if arm == 'full_label':
        verify_full_label_checkpoint(bundle,checkpoint)
    else:
        verify_checkpoint(bundle,checkpoint)
    if arm != 'full_label' and bundle['config']['arm'] != arm:
        raise ValueError('Requested arm differs from checkpoint')
    # Previous action is deliberately ignored: only measured current state and terrain enter E.
    return lambda env,s,a,c: control_action_details(bundle,s,c), bundle


def score(arm, checkpoint, split, name, out, protocol, device):
    destination=out/'evaluations'/f'{name}_{split}.json'
    digest=sha256(checkpoint) if checkpoint else None
    if destination.exists():
        old=json.loads(destination.read_text())
        if old['checkpoint_sha256'] != digest or old['protocol_sha256'] != sha256(out/'protocol.json'):
            raise RuntimeError('Cached score provenance differs')
        if sha256(old['traces']) != old['traces_sha256']:
            raise RuntimeError('Cached traces changed')
        return old
    action,bundle=make_controller(arm,checkpoint,device)
    if arm in ('probes','auxiliary'):
        verify_training_protocol(bundle,protocol)
    row=evaluate_controller(action,protocol[split+'_seeds'],out/'traces'/f'{name}_{split}.npz',name+'/'+split)
    row.update(name=name,arm=arm,split=split,checkpoint=str(Path(checkpoint).resolve()) if checkpoint else None,
        checkpoint_sha256=digest,protocol_sha256=sha256(out/'protocol.json'),device=device,
        step=int(bundle.get('step',0)) if bundle else 0)
    if bundle:
        row['training']=bundle.get('training_summary')
        row['checkpoint_provenance']=bundle['provenance']
    if arm in ('probes','auxiliary','full_label'):
        records=build_expert_records(protocol['source_episodes'],split)
        row['expert_diagnostics']=auxiliary_metrics(bundle,records)
        row['expert_diagnostics'].update(source_split=split,source_sha256=protocol['source_episodes_sha256'],
            episode_ids=np.unique(records['episode_ids']).tolist(), scope='Historical held-out expert behavior transitions; secondary diagnostics only',
            action_target='Recorded expert current command; never provided to the state encoder')
        row['on_policy_diagnostics']=rollout_auxiliary(bundle,row)
        row['timing_scope']='E + G2 + MoG routing/offset diagnostics; auxiliary heads scored offline'
    else:
        row['timing_scope']='Hand-written heuristic only'
    write_json(destination,row)
    logging.info('SCORE %s %s landing=%d/%d mean_return=%.3f',name,split,row['summary']['landing_count'],row['summary']['episodes'],row['summary']['mean_return'])
    return row


def refresh_leaderboard(out):
    selections=[json.loads(p.read_text()) for p in sorted((out/'selections').glob('*.json'))]
    rows=[]
    for selection in selections:
        for role in ('selected','final'):
            if selection.get(role+'_test'):
                row=json.loads(Path(selection[role+'_test']).read_text())
                row['report_role']=role
                rows.append(row)
    references={r['arm']:r for r in rows if r['report_role']=='selected'}
    for row in rows:
        row['paired']={name:paired_comparison(row['episodes'],ref['episodes']) for name,ref in references.items() if name != row['arm']}
    rows.sort(key=lambda r: (r['summary']['landing_rate'],r['summary']['mean_return']),reverse=True)
    write_json(out/'leaderboard.json',dict(protocol_sha256=sha256(out/'protocol.json'),rows=rows))
    lines=['# Lunar Lander with scarce action labels','',
        'Selection uses validation landing rate, then mean return. Test worlds are paired and fresh for this round. Intervals are 95% Wilson intervals over evaluation episodes.','',
        '| Controller | Checkpoint | Landings | 95% interval | Mean return | Median return | Crash / bounds / time limit |',
        '| --- | --- | ---: | --- | ---: | ---: | --- |']
    for row in rows:
        s=row['summary']; lo,hi=s['landing_rate_wilson95']; c=s['outcomes']
        lines.append(f"| {row['arm']} | {row['report_role']} ({row['step']}) | {s['landing_count']}/{s['episodes']} | {lo:.1%}–{hi:.1%} | {s['mean_return']:.2f} | {s['median_return']:.2f} | {c['crash']} / {c['out_of_bounds']} / {c['time_limit']} |")
    lines += ['', 'Full action traces, paired comparisons, inference/training costs, engine usage, expert reconstruction and learner-successor metrics are in `leaderboard.json`. G3 has no alternative-action input; its consistency with G2 is measured on actual learner rollouts. The full-label reference uses its original scaler; normalized errors are directly comparable between the two sparse arms, while physical errors support comparison to the reference.','']
    (out/'README.md').write_text('\n'.join(lines))
    eligible=[s for s in selections if s['arm']!='expert']
    if eligible:
        winner=max(eligible,key=lambda s:selection_key(s['validation']))
        name=lambda arm: 'sparse_'+arm if arm in ('probes','auxiliary') else ('state_probes' if arm == 'full_label' else arm)
        controllers={name(s['arm']):dict(checkpoint=s.get('selected_checkpoint'),validation=s['validation']['summary'],step=s['validation'].get('step',0)) for s in selections}
        write_json(out/'controllers.json',dict(default_controller=name(winner['arm']),controllers=controllers,
            selection='Fresh paired validation landing rate, then mean return; expert excluded',protocol_sha256=sha256(out/'protocol.json')))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=REPORTS)
    parser.add_argument('--episodes',type=Path,default=EPISODES)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--arm',choices=('probes','auxiliary'))
    parser.add_argument('--checkpoint',type=Path,action='append',default=[])
    parser.add_argument('--final',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--log',type=Path,default=ROOT/'results/gym/lunar_lander_sparse_action/live.log')
    args=parser.parse_args()
    if args.device not in ('cpu','cuda:1'): parser.error('Evaluation device must be cpu or cuda:1; GPU0 belongs to the user')
    if not (args.freeze or args.baseline or args.arm): parser.error('Choose --freeze, --baseline or --arm')
    if args.arm and (not args.checkpoint or not args.final): parser.error('--arm requires candidates and --final')
    args.log.parent.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(args.log)])
    torch.set_num_threads(1)
    if args.freeze:
        freeze_protocol(args.out/'protocol.json',args.episodes,args.reference)
        logging.info('FROZEN %s',args.out/'protocol.json')
    protocol=verify_protocol(args.out/'protocol.json')
    if args.baseline:
        if sha256(args.reference) != protocol['reference_checkpoint_sha256']: raise RuntimeError('Reference differs from protocol')
        for arm,checkpoint in [('expert',None),('full_label',args.reference)]:
            validation=score(arm,checkpoint,'validation',arm,args.out,protocol,args.device)
            selection=dict(arm=arm,validation=validation,selected_checkpoint=str(checkpoint.resolve()) if checkpoint else None)
            write_json(args.out/'selections'/f'{arm}.json',selection)
            score(arm,checkpoint,'test',arm,args.out,protocol,args.device)
            selection['selected_test']=str((args.out/'evaluations'/f'{arm}_test.json').resolve())
            write_json(args.out/'selections'/f'{arm}.json',selection)
            refresh_leaderboard(args.out)
    if args.arm:
        from lib.gym_state_control import load_state_control_checkpoint
        final_bundle=load_state_control_checkpoint(args.final,args.device)
        verify_checkpoint(final_bundle,args.final)
        verify_training_protocol(final_bundle,protocol)
        if final_bundle['step'] != protocol['training_updates'] or final_bundle['config']['arm'] != args.arm:
            raise RuntimeError('Final checkpoint is not the declared final training update/arm')
        candidate_updates=[]
        for checkpoint in args.checkpoint:
            candidate=load_state_control_checkpoint(checkpoint,args.device)
            verify_checkpoint(candidate,checkpoint)
            verify_training_protocol(candidate,protocol)
            if candidate['provenance'] != final_bundle['provenance'] or candidate['config']['arm'] != args.arm:
                raise RuntimeError('Candidates must come from the same completed training run')
            candidate_updates.append(candidate['step'])
        if sorted(candidate_updates) != protocol['candidate_updates']:
            raise RuntimeError('Candidate updates differ from frozen protocol')
        candidates=[]
        for checkpoint in args.checkpoint:
            name=f'{args.arm}_{checkpoint.stem}_{sha256(checkpoint)[:10]}'
            candidates.append(score(args.arm,checkpoint,'validation',name,args.out,protocol,args.device))
        if sorted(r['step'] for r in candidates) != protocol['candidate_updates']: raise RuntimeError('Candidate updates differ from protocol')
        selected=max(candidates,key=selection_key)
        best=args.final.parent/'best.pt'
        if best.exists() and sha256(best) != selected['checkpoint_sha256']: raise RuntimeError('Different best checkpoint already exists')
        if not best.exists(): shutil.copyfile(selected['checkpoint'],best)
        selection=dict(arm=args.arm,validation=selected,candidates=[dict(name=r['name'],checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],step=r['step'],summary=r['summary']) for r in candidates],selected_checkpoint=str(best.resolve()))
        selection_path=args.out/'selections'/f'{args.arm}.json'
        write_json(selection_path,selection) # Selection persisted before any test result.
        for role,checkpoint in [('selected',best),('final',args.final)]:
            name=f'{args.arm}_{role}'
            if role=='final' and sha256(checkpoint)==sha256(best):
                row=json.loads(Path(selection['selected_test']).read_text())
                row.update(name=name,checkpoint=str(checkpoint.resolve()),reused_identical_selected_test=True)
                write_json(args.out/'evaluations'/f'{name}_test.json',row)
            else: score(args.arm,checkpoint,'test',name,args.out,protocol,args.device)
            selection[role+'_test']=str((args.out/'evaluations'/f'{name}_test.json').resolve())
        write_json(selection_path,selection)
        refresh_leaderboard(args.out)


if __name__ == '__main__':
    main()
