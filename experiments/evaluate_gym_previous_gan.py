#!/usr/bin/env python
"""Fresh paired evaluation for scratch previous-action GAN + action L2."""
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
from experiments.train_gym_previous_gan import DEFAULTS, source_paths
from experiments.evaluate_gym_control import (write_json, expert_action_diagnostics,
    make_controller as make_legacy)
from experiments.evaluate_gym_gan_control import make_controller as make_gan
from experiments.evaluate_gym_state_control import check_disjoint
from lib.gym_control import build_expert_records
from lib.gym_control_evaluation import evaluate_controller, paired_comparison, selection_key
from lib.gym_data import sha256, simulator_provenance
from lib.gym_previous_gan import load_checkpoint, hashes, control_action_details, decode

REPORTS = ROOT / 'reports/gym/lunar_lander_previous_gan'
REFERENCES = dict(joint=ROOT/'results/gym/lunar_lander_gan_control/joint/best.pt',
    marginals=ROOT/'results/gym/lunar_lander_gan_control/marginals/best.pt',
    legacy_joint=ROOT/'results/gym/lunar_lander_control/joint/best.pt',
    imitation_reference=ROOT/'results/gym/lunar_lander_control/imitation/best.pt')
GAN_ARMS = ('previous_marginals', 'joint', 'marginals', 'legacy_joint')


def freeze():
    path = REPORTS / 'protocol.json'
    if path.exists():
        return verify_protocol()
    previous = [p for p in sorted((ROOT/'reports/gym').rglob('protocol.json')) if p != path]
    validation, test = list(range(1191000, 1191020)), list(range(1291000, 1291050))
    episodes = ROOT / DEFAULTS['episodes']
    used = check_disjoint(json.loads(episodes.read_text()), [json.loads(p.read_text()) for p in previous], validation, test)
    sources = source_paths() + [Path(__file__), ROOT/'lib/gym_control_evaluation.py', ROOT/'lib/gym_data.py',
        ROOT/'experiments/evaluate_gym_control.py', ROOT/'experiments/evaluate_gym_gan_control.py',
        ROOT/'experiments/evaluate_gym_state_control.py', ROOT/'lib/gym_gan_control.py']
    records = build_expert_records(episodes)
    value = dict(version=1, validation_seeds=validation, test_seeds=test,
        previously_used_reset_seeds=used, prior_protocols={str(p): sha256(p) for p in previous},
        sources={str(p): sha256(p) for p in sources}, source_episodes=str(episodes),
        source_episodes_sha256=sha256(episodes), simulator=simulator_provenance(), training_config=DEFAULTS,
        candidate_updates=[250, 1000, 2500], training_records=len(records['states']),
        expert_episode_ids=np.unique(records['episode_ids']).tolist(),
        array_hashes={k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in records.items()},
        references={k: dict(path=str(p), sha256=sha256(p)) for k,p in REFERENCES.items()},
        eligible_gan_arms=list(GAN_ARMS),
        reference_policy='Imitation is a separate non-GAN reference, excluded from GAN ranking and defaults',
        selection=['validation landing rate', 'validation mean return', 'earlier update exact tie'],
        interpretation='New model changes encoder, label count, initialization, and training objective relative to references; not a one-factor ablation',
        inference='E(st, at-1, terrain) -> z -> G2 -> at; one encoder; actual simulator advances state',
        previous_action='Expert previous command in training; learner previous command during rollout; reset [-1, 0]',
        training='Scratch G1/G2/G3/E/prior/D; joint + shared state + action critics and action MSE throughout; no synthetic cycle',
        success='terminated and lander asleep; simulator final sleep precedence',
        max_episode_steps=1000, training_seed_repeats=False)
    write_json(path, value)
    return value


def verify_protocol():
    value = json.loads((REPORTS/'protocol.json').read_text())
    artifacts = {**value['sources'], **value['prior_protocols'],
        value['source_episodes']: value['source_episodes_sha256'],
        **{v['path']: v['sha256'] for v in value['references'].values()}}
    for name, digest in artifacts.items():
        if sha256(name) != digest:
            raise RuntimeError(f'Frozen artifact changed: {name}')
    if simulator_provenance() != value['simulator']:
        raise RuntimeError('Simulator changed')
    return value


def verify_candidate(bundle, checkpoint, protocol):
    summary, provenance, cfg = bundle.get('training_summary'), bundle['provenance'], bundle['config']
    if not summary or sha256(checkpoint) not in summary['checkpoints'].values():
        raise ValueError('Candidate is not a completed hashed training checkpoint')
    if summary['provenance'] != provenance or summary['gan_steps'] != cfg['steps'] or bundle['gan_steps'] != bundle['step']:
        raise ValueError('Missing every-update GAN provenance')
    if summary['config'] != cfg or provenance.get('gan_training') is not True:
        raise ValueError('Completed summary must establish the GAN training configuration')
    expected_sources = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths()}
    if provenance['sources'] != expected_sources:
        raise ValueError('Candidate lacks the exact verified training implementation')
    expected = protocol['training_config']
    if any(cfg[k] != v for k,v in expected.items() if k not in ('out_dir', 'live_log', 'device')):
        raise ValueError('Candidate differs from frozen configuration')
    if provenance['expert_data']['episode_ids'] != protocol['expert_episode_ids']:
        raise ValueError('Different training episodes')
    if {k: v['sha256'] for k,v in provenance['expert_data']['arrays'].items()} != protocol['array_hashes']:
        raise ValueError('Different training records or previous-action alignment')
    if provenance['episodes']['sha256'] != protocol['source_episodes_sha256']:
        raise ValueError('Different episode source')
    for name, digest in provenance['sources'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Training source changed: {name}')
    for name, digest in [('source.zip', provenance['source_archive_sha256']),
                         ('expert_records.npz', provenance['expert_data']['npz_sha256'])]:
        if sha256(Path(checkpoint).parent/name) != digest:
            raise ValueError(f'Training artifact changed: {name}')
    if set(bundle['D'].critics) != {'joint', 'action', 'state'}:
        raise ValueError('Joint and marginal discriminator weights required')
    current = hashes(bundle)
    for name in ('G', 'E', 'prior', 'D.joint', 'D.action', 'D.state'):
        if current[name] == provenance['initial_parameters'][name]:
            raise ValueError(f'Untrained component: {name}')
    if any(not torch.isfinite(p).all() for k in ('G', 'E', 'prior', 'D') for p in bundle[k].parameters()):
        raise ValueError('Nonfinite model')


@torch.no_grad()
def diagnostics(bundle):
    records = build_expert_records(ROOT/DEFAULTS['episodes'], 'test')
    decoded = []
    for start in range(0, len(records['states']), 512):
        inputs = [torch.from_numpy(records[k][start:start+512]) for k in ('states', 'previous_actions', 'terrain')]
        decoded.append(decode(bundle, *inputs)[0])
    pred = torch.cat(decoded)
    scaler = bundle['scaler']
    states, actions, following = [torch.from_numpy(records[k]) for k in ('states', 'actions', 'next_states')]
    sn, nn = scaler.state(states), scaler.state(following)
    return dict(records=len(states), source='held-out expert behavior; expert previous commands',
        action=expert_action_diagnostics(bundle, ROOT/DEFAULTS['episodes'], 'test'),
        state_mse=float((pred[:, :6]-sn[:, :6]).square().mean()),
        next_mse=float((pred[:, 10:16]-nn[:, :6]).square().mean()),
        next_contact_brier=float((pred[:, 16:].sigmoid()-nn[:, 6:]).square().mean()),
        persistence_mse=float((sn[:, :6]-nn[:, :6]).square().mean()),
        persistence_contact_brier=float((sn[:, 6:]-nn[:, 6:]).square().mean()))


def score(arm, checkpoint, split, name, protocol):
    destination = REPORTS/'evaluations'/f'{name}_{split}.json'
    digest = sha256(checkpoint)
    if destination.exists():
        old = json.loads(destination.read_text())
        if old['checkpoint_sha256'] != digest or old['protocol_sha256'] != sha256(REPORTS/'protocol.json') or sha256(old['traces']) != old['traces_sha256']:
            raise RuntimeError('Cached score differs')
        return old
    if arm == 'previous_marginals':
        bundle = load_checkpoint(checkpoint)
        verify_candidate(bundle, checkpoint, protocol)
        action = lambda env,s,a,c: control_action_details(bundle,s,a,c)
    elif arm == 'imitation_reference':
        action, bundle = make_legacy('imitation', checkpoint)
    else:
        action, bundle = make_gan(arm, checkpoint, 'cpu')
    result = evaluate_controller(action, protocol[split+'_seeds'], REPORTS/'traces'/f'{name}_{split}.npz', name)
    result.update(arm=arm, step=bundle['step'], split=split, checkpoint=str(checkpoint),
        checkpoint_sha256=digest, protocol_sha256=sha256(REPORTS/'protocol.json'), gan_eligible=arm in GAN_ARMS)
    write_json(destination, result)
    logging.info('SUMMARY %s %s landed=%d/%d mean=%.3f', name, split, result['summary']['landing_count'],
        result['summary']['episodes'], result['summary']['mean_return'])
    return result


def evaluate(run_dir):
    protocol = verify_protocol()
    vals = [score('previous_marginals', run_dir/f'checkpoint_{step}.pt', 'validation', f'previous_{step}', protocol)
            for step in protocol['candidate_updates']]
    winner = max(vals, key=selection_key)
    shutil.copyfile(winner['checkpoint'], run_dir/'best.pt')
    selected = score('previous_marginals', run_dir/'best.pt', 'test', 'previous_selected', protocol)
    final = selected if sha256(run_dir/'final.pt') == selected['checkpoint_sha256'] else score(
        'previous_marginals', run_dir/'final.pt', 'test', 'previous_final', protocol)
    rows = [dict(arm='previous_marginals', validation=winner, test=selected)]
    reference = None
    for arm, checkpoint in REFERENCES.items():
        row = dict(arm=arm, validation=score(arm, checkpoint, 'validation', arm, protocol),
                   test=score(arm, checkpoint, 'test', arm, protocol))
        if arm == 'imitation_reference':
            reference = row
        else:
            rows.append(row)
    for row in rows + [reference]:
        for left, right in zip(selected['episodes'], row['test']['episodes']):
            if left['seed'] != right['seed'] or left['initial_state'] != right['initial_state'] or left['terrain'] != right['terrain']:
                raise RuntimeError('Paired world starts differ')
    pairs = {row['arm']: paired_comparison(selected['episodes'], row['test']['episodes']) for row in rows[1:]+[reference]}
    chosen = max(rows, key=lambda row: selection_key(row['validation']))
    board = dict(protocol_sha256=sha256(REPORTS/'protocol.json'), gan_rows=rows,
        non_gan_reference=reference, validation_selected_gan=chosen['arm'], comparisons=pairs,
        selected_update=winner['step'], final_test=final)
    write_json(REPORTS/'leaderboard.json', board)
    write_json(REPORTS/'selection.json', dict(validation=winner, test=selected))
    write_json(REPORTS/'diagnostics.json', diagnostics(load_checkpoint(run_dir/'best.pt')))
    train_archive = REPORTS/'training'
    train_archive.mkdir(exist_ok=True)
    for name in ('summary.json', 'provenance.json', 'source.zip', 'config.yaml', 'normalization.json',
                 'recipe.json', 'environment.json', 'metrics.jsonl'):
        shutil.copyfile(run_dir/name, train_archive/name)
    lines = ['# Previous-action GAN from scratch', '',
        'GAN-only ranking on paired fresh worlds; checkpoints/default selected by validation.', '',
        '| GAN | Validation landings | Test landings | Mean test return |',
        '| --- | ---: | ---: | ---: |']
    for row in sorted(rows, key=lambda row: selection_key(row['validation']), reverse=True):
        v,t = row['validation']['summary'], row['test']['summary']
        lines.append(f'| {row["arm"]} | {v["landing_count"]}/20 | {t["landing_count"]}/50 | {t["mean_return"]:.2f} |')
    r = reference['test']['summary']
    lines += ['', f'Non-GAN imitation reference, excluded from ranking: {r["landing_count"]}/50, mean return {r["mean_return"]:.2f}.',
        '', f'Validation-selected GAN: `{chosen["arm"]}`. New model selected update: {winner["step"]}.',
        '', 'The references differ in labels, initialization, encoder, and losses; these are benchmark comparisons, not a one-factor ablation.',
        '', 'See [the readout](READOUT.md) for interpretation and training costs.']
    (REPORTS/'README.md').write_text('\n'.join(lines)+'\n')
    return board


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    live = Path(DEFAULTS['live_log'])
    live.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(live)])
    if args.freeze:
        freeze()
        logging.info('PROTOCOL FROZEN %s', REPORTS/'protocol.json')
    if args.run_dir:
        evaluate(args.run_dir)


if __name__ == '__main__':
    main()
