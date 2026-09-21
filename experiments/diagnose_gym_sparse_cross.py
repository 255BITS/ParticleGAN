#!/usr/bin/env python
"""Compare both selected sparse-action models on both frozen learner trace datasets."""
import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.evaluate_gym_sparse_action import (REPORTS, auxiliary_metrics,
    verify_checkpoint, verify_protocol, write_json)
from lib.gym_data import sha256
from lib.gym_state_control import load_state_control_checkpoint


def diagnose(reports, output):
    if output.exists() or output.with_suffix('.md').exists():
        raise FileExistsError('Cross diagnostics already exist; use a fresh output path')
    protocol=verify_protocol(reports/'protocol.json')
    models, datasets, provenance={}, {}, {}
    for arm in ('probes','auxiliary'):
        selection_path=reports/'selections'/f'{arm}.json'
        selection=json.loads(selection_path.read_text())
        checkpoint=Path(selection['selected_checkpoint'])
        if sha256(checkpoint)!=selection['validation']['checkpoint_sha256']:
            raise RuntimeError('Selected model differs from validation selection')
        models[arm]=load_state_control_checkpoint(checkpoint,'cpu')
        verify_checkpoint(models[arm],checkpoint)
        row_path=Path(selection['selected_test'])
        row=json.loads(row_path.read_text())
        if row['protocol_sha256']!=sha256(reports/'protocol.json') or row['checkpoint_sha256']!=sha256(checkpoint):
            raise RuntimeError('Selected test trace provenance differs')
        if sha256(row['traces'])!=row['traces_sha256']:
            raise RuntimeError('Recorded learner traces changed')
        terrain={e['seed']:e['terrain'] for e in row['episodes']}
        if sorted(terrain)!=protocol['test_seeds']:
            raise RuntimeError('Trace worlds differ from frozen test worlds')
        with np.load(row['traces']) as trace:
            datasets[arm]={k:trace[k] for k in ('states','actions','next_states')}
            datasets[arm]['terrain']=np.asarray([terrain[int(s)] for s in trace['seeds']],np.float32)
        provenance[arm]=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
            selection=str(selection_path),selection_sha256=sha256(selection_path),
            evaluation=str(row_path),evaluation_sha256=sha256(row_path),
            traces=row['traces'],traces_sha256=row['traces_sha256'])
    cells=[]
    for trace_arm,records in datasets.items():
        for model_arm,model in models.items():
            metrics=auxiliary_metrics(model,records)
            cells.append(dict(model=model_arm,trace_controller=trace_arm,**metrics))
            logging.info('CROSS model=%s traces=%s n=%d state_mse=%.6f next_mse=%.6f action_mse=%.6f',
                model_arm,trace_arm,metrics['records'],metrics['state']['standardized_continuous_mse'],
                metrics['next_state']['standardized_continuous_mse'],metrics['standardized_action_mse'])
    value=dict(protocol_sha256=sha256(reports/'protocol.json'),provenance=provenance,
        sources={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'experiments/evaluate_gym_sparse_action.py',ROOT/'lib/gym_state_control.py']},
        device='cpu',simulator_calls=0,training_updates=0,cells=cells,
        scope='Posthoc comparison on previously evaluated test trajectories. Equal inputs within each trace dataset; transition-weighted metrics.',
        caveats=['G3 has no alternative-action input. On another controller trajectory its predicted action may differ from the recorded action that produced the successor.',
            'Action MSE compares each model command against the recorded learner command, not expert commands.',
            'This describes generalization on fixed states; it is not a causal test of control quality, an unrestricted counterfactual model test, or a new checkpoint selection.'])
    write_json(output,value)
    lines=['# Cross-policy state and successor diagnostics','',
        'Both validation-selected models receive the same measured states and terrain within each row pair. All successors come from existing real simulator traces. No new simulator calls or training.','',
        '| Trace controller | Model | Records | G1 state MSE | G3 successor MSE | Persistence MSE | Action MSE vs recorded command |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for c in cells:
        lines.append(f"| {c['trace_controller']} | {c['model']} | {c['records']} | {c['state']['standardized_continuous_mse']:.6f} | {c['next_state']['standardized_continuous_mse']:.6f} | {c['persistence_next_state']['standardized_continuous_mse']:.6f} | {c['standardized_action_mse']:.6g} |")
    lines += ['', 'State and successor MSE average the six continuous coordinates after the frozen training normalization; action MSE averages two standardized commands. JSON also contains physical errors and contact BCE, Brier score, and accuracy.','',
        'G1 has an identical target across models on each trace set, making its comparison direct. G3 cannot condition on the other policy’s recorded action: action disagreement limits interpretation of cross-policy successor errors. Action errors here measure agreement with recorded learner commands, not the heuristic. These posthoc results do not select checkpoints or establish causality.','']
    output.with_suffix('.md').write_text('\n'.join(lines))
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,default=REPORTS)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--log',type=Path,default=ROOT/'results/gym/lunar_lander_sparse_action/live.log')
    args=parser.parse_args()
    args.log.parent.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(args.log)])
    torch.set_num_threads(1)
    diagnose(args.reports,args.out or args.reports/'cross_diagnostics.json')


if __name__=='__main__':
    main()
