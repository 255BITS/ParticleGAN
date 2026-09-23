#!/usr/bin/env python
"""Compare previous-action GAN and imitation on fixed saved inputs; no rollouts."""
import inspect
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.evaluate_gym_previous_gan import REPORTS, REFERENCES, verify_protocol
from experiments.evaluate_gym_control import write_json, expert_action_diagnostics, verify_control_checkpoint
from experiments.diagnose_gym_control_actions import action_metrics
from experiments.diagnose_gym_sparse_actions import lateral_metrics
from lib.gym_control import load_control_checkpoint, predict_control
from lib.gym_previous_gan import load_checkpoint
from lib.gym_data import make_env, sha256
import json


@torch.no_grad()
def main():
    from gymnasium.envs.box2d.lunar_lander import heuristic
    torch.set_num_threads(1)
    protocol = verify_protocol()
    checkpoint = ROOT/'results/gym/lunar_lander_previous_gan/previous_marginals/best.pt'
    new = load_checkpoint(checkpoint)
    imitation = load_control_checkpoint(REFERENCES['imitation_reference'])
    verify_control_checkpoint(imitation, REFERENCES['imitation_reference'])
    bundles = dict(previous_marginals={**new, 'E_control': new['E']}, imitation=imitation)
    output = dict(kind='Posthoc fixed-input diagnostics; no training or selection',
        protocol_sha256=sha256(REPORTS/'protocol.json'), sources={str(p):sha256(p) for p in
            [Path(__file__), Path(inspect.getfile(action_metrics)), Path(inspect.getfile(lateral_metrics)), Path(inspect.getfile(heuristic))]},
        checkpoints=dict(previous_marginals=sha256(checkpoint), imitation=sha256(REFERENCES['imitation_reference'])),
        simulator_steps=0, simulator_resets=0, training_records_added=0,
        interpretation='Both models receive identical saved current states, previous commands, and terrain within each trace dataset. Heuristic recommendations are descriptive, not proven recovery actions.',
        expert={}, fixed_traces={})
    for name, bundle in bundles.items():
        output['expert'][name] = expert_action_diagnostics(bundle, ROOT/'results/gym/lunar_lander/data/episodes.json', 'test')
    env = make_env()  # Only continuous flag used by heuristic; no reset or step.
    try:
        for source, filename in [('previous_marginals','previous_selected_test.json'), ('imitation','imitation_reference_test.json')]:
            path = REPORTS/'evaluations'/filename
            row = json.loads(path.read_text())
            if row['protocol_sha256'] != output['protocol_sha256'] or sha256(row['traces']) != row['traces_sha256']:
                raise RuntimeError('Trace provenance mismatch')
            with np.load(row['traces'], allow_pickle=False) as archive:
                arrays = {k:archive[k] for k in archive.files}
            if sorted(np.unique(arrays['seeds']).tolist()) != protocol['test_seeds']:
                raise RuntimeError('Trace reset worlds differ')
            terrains = {e['seed']:e['terrain'] for e in row['episodes']}
            terrain = np.asarray([terrains[int(s)] for s in arrays['seeds']], np.float32)
            expert = np.asarray([heuristic(env.unwrapped, s) for s in arrays['states']], np.float32)
            results = {}
            for name, bundle in bundles.items():
                predictions = []
                for start in range(0, len(expert), 512):
                    args = [x[start:start+512] for x in (arrays['states'], arrays['previous_actions'], terrain)]
                    predictions.append(predict_control(bundle, *args)[0].cpu().numpy())
                prediction = np.concatenate(predictions)
                scale = bundle['scaler'].action_scale.cpu().numpy()
                masks = dict(overall=np.ones(len(expert), bool), first20=arrays['steps'] < 20, later=arrays['steps'] >= 20)
                results[name] = {group:{**action_metrics(prediction, expert, scale, mask),
                                        **lateral_metrics(prediction, expert, mask)} for group,mask in masks.items()}
                results[name]['episode_mean_physical_mse'] = float(np.mean([
                    np.mean((prediction[arrays['seeds']==seed]-expert[arrays['seeds']==seed])**2)
                    for seed in protocol['test_seeds']]))
                if source == name:
                    replay_mse = float(np.mean((prediction-arrays['actions'])**2))
                    if replay_mse > 1e-10:
                        raise RuntimeError('Saved control actions do not replay')
                    results[name]['own_action_replay_mse'] = replay_mse
            output['fixed_traces'][source] = dict(evaluation_sha256=sha256(path),
                traces_sha256=row['traces_sha256'], records=len(expert), models=results)
    finally:
        env.close()
    write_json(REPORTS/'action_diagnostics.json', output)
    lines = ['# Action disagreement on identical saved inputs', '',
        'Posthoc only: no simulator resets/steps or training. Each row compares with the heuristic on the recorded current state.', '',
        '| Input source | Evaluated model | Physical action MSE | Main agreement | Side agreement |',
        '| --- | --- | ---: | ---: | ---: |']
    for name, row in output['expert'].items():
        lines.append(f'| Expert states + expert previous actions | {name} | {row["physical_action_mse"]:.6f} | {row["main_regime_agreement"]:.1%} | {row["lateral_regime_agreement"]:.1%} |')
    for source, row in output['fixed_traces'].items():
        for name, groups in row['models'].items():
            g=groups['overall']
            lines.append(f'| {source} traces | {name} | {g["physical_action_mse"]:.6f} | {g["main_regime_agreement"]:.1%} | {g["lateral_regime_agreement"]:.1%} |')
    lines += ['', 'Within a trace source, both models receive exactly the same state, previous command, and terrain. Physical MSE is comparable across models; standardized MSE in JSON uses each model’s own scaler.', '',
        'Expert and learner datasets differ in both current states and previous commands. This does not isolate previous-action feedback, establish a causal failure mechanism, or prove that the heuristic/imitation would recover from learner states.', '']
    (REPORTS/'action_diagnostics.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
