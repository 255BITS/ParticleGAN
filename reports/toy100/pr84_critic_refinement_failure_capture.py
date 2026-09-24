"""Passive exact cold replay that saves the first failed inner critic fit."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from reports.toy100 import pr84_critic_relaxation as fit
from reports.toy100.pr84_critic_refinement_cold import pr84_critic_refinement_cold
from reports.toy100.pr84_critic_refinement_capture import snapshot
from reports.toy100.pr84_prediction_state_filter import clone, state_hash
from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    declaration = json.loads((args.original / 'declaration.json').read_text())
    for name, digest in declaration['source'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'failed cold source changed: {name}')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'original-declaration.json').write_bytes((args.original / 'declaration.json').read_bytes())
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    config = json.loads((args.original / 'config.json').read_text())
    recipe, noise, _ = declared_recipe(config)
    spec = next(job['spec'] for job in plan() if job['spec']['name'] == 'mode_hold')
    torch.set_num_threads(1)
    original_fit = fit.relax
    saved = {}
    with pr84_critic_refinement_cold(task='mode_hold') as (recorder, _):
        phases = recorder.phases
        def observed_phases(step, opt_d, opt_g, local):
            saved['pre_step'] = snapshot(local)
            saved['host_update'] = step + 1
            for phase in phases(step, opt_d, opt_g, local):
                yield phase
            if recorder.outer_steps % 20 == 0:
                print(json.dumps(dict(event='EXACT_REPLAY_PROGRESS', update=step+1,
                    fit_gradient_evaluations=recorder.fit_gradient_evaluations)), flush=True)
        recorder.phases = observed_phases

        def observed_fit(critic, bank, gan, regularizer, step, metric):
            accepted = snapshot(recorder._local)
            try:
                return original_fit(critic, bank, gan, regularizer, step, metric)
            except FloatingPointError as error:
                if str(error) != 'nonfinite local critic fit':
                    raise
                frames = []
                tb = error.__traceback__
                while tb is not None:
                    frames.append((tb.tb_frame.f_code.co_name, tb.tb_frame.f_locals))
                    tb = tb.tb_next
                relaxation = next(values for name, values in frames if name == 'relax')
                closure = next(values for name, values in frames if name == 'closure')
                wolfe = next((values for name, values in frames if name == '_strong_wolfe'), {})
                payload = dict(pre_step=saved['pre_step'], post_accepted_d=accepted,
                    bank=clone(bank), metric=clone(metric), best=clone(relaxation['best']),
                    finite_records=clone(relaxation['records']),
                    invalid_critic=clone(critic.state_dict()),
                    invalid_gradient=[None if p.grad is None else p.grad.detach().clone() for p in critic.parameters()],
                    inner_optimizer=clone(relaxation['optimizer'].state_dict()),
                    completed_dynamics=recorder.receipt(), host_update=saved['host_update'])
                target = args.output / 'failed-fit.pt'
                torch.save(payload, target)
                finite_loss = float(closure['loss'].detach())
                finite_parameters = sum(int(torch.isfinite(p).all()) for p in critic.parameters())
                row = dict(status='EXACT_NONFINITE_FIT_CAPTURE', host_update=saved['host_update'],
                    completed_updates=recorder.outer_steps, error=str(error),
                    finite_closures=len(relaxation['records']),
                    failed_closure=len(relaxation['records'])+1,
                    trial_loss_repr=repr(finite_loss),
                    trial_logistic_repr=repr(float(closure['logistic'].detach())),
                    trial_penalty_repr=repr(float(closure['penalty'].detach())),
                    parameter_tensors=len(list(critic.parameters())),
                    finite_parameter_tensors=finite_parameters,
                    best_loss=relaxation['best_loss'], initial_loss=relaxation['records'][0]['total_loss'],
                    wolfe_scalar_state={key:repr(wolfe[key]) for key in ('t','t_prev','f','f_new','f_prev','ls_iter') if key in wolfe},
                    state_file_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    saved_pre_step_sha256=state_hash(saved['pre_step']),
                    accepted_d_sha256=state_hash(accepted),
                    original_source=declaration['source'],
                    observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    quality_gate_complete=False, training_rule_changed=False)
                (args.output / 'failure.json').write_text(json.dumps(row, indent=2)+'\n')
                print(json.dumps(dict(event='FAILURE_CAPTURED', **row)), flush=True)
                raise
        with patch.object(fit, 'relax', observed_fit):
            try:
                run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
            except FloatingPointError as error:
                if str(error) == 'nonfinite local critic fit' and (args.output / 'failure.json').exists():
                    return
                raise
    raise RuntimeError('declared original fitting error did not reproduce')


if __name__ == '__main__':
    main()
