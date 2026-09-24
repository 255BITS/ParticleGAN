"""Exact saved-state local comparison, without a new prefix or seed.

This edits only the outer host loop bounds and restores one archived pre-step
state before its first gradient block. Ordinary replay must match the archived
accepted state bit for bit before opponent prediction is interpreted. Ring
grades are read-only diagnostics and never enter the candidate update.
"""

from contextlib import ExitStack
from copy import deepcopy
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks import learned_lr_evaluation as bridge
from benchmarks.locked_shared import mode_hold
from benchmarks.smart_descent import evaluate
from benchmarks.toy100.continuous_probe import _noise_policy, prepared_config
from benchmarks.toy100.warm_equilibrium_probe import _feed_hash
from benchmarks.transfer_suite import vector_tasks
from benchmarks.transfer_suite.compare_defaults import candidate, optimizer_defaults
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
from reports.toy100 import pr84_opponent_prediction as prediction_module
from reports.toy100.coverage_fixed_eval import fixed_draw, score_support


def state_hash(value):
    digest = hashlib.sha256()
    _feed_hash(digest, value)
    return digest.hexdigest()


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone(item) for item in value)
    return deepcopy(value)


def snapshot(local):
    policy = local['noise_policy']
    return clone(dict(
        generator=local['generator'].state_dict(), critic=local['critic'].state_dict(),
        prior=local['prior'].state_dict(), optimizer_d=local['opt_d'].state_dict(),
        optimizer_g=local['opt_g'].state_dict(), ema_g=local['ema_g'], ema_z=local['ema_z'],
        rng=dict(torch=torch.get_rng_state(), data=local['stream'].get_state(),
                 input=policy.input_stream.get_state(), output=None if policy.output_stream is None
                 else policy.output_stream.get_state()),
        noise=dict(step_calls=policy._step_calls, input_sigma=policy.input_sigma,
                   output_sigma=policy.output_sigma)))


def restore(local, saved):
    for name in ('generator', 'critic', 'prior'):
        local[name].load_state_dict(saved[name])
    local['opt_d'].load_state_dict(deepcopy(saved['optimizer_d']))
    local['opt_g'].load_state_dict(deepcopy(saved['optimizer_g']))
    with torch.no_grad():
        for target, source in zip(local['ema_g'], saved['ema_g']):
            target.copy_(source)
        local['ema_z'].copy_(saved['ema_z'])
    policy = local['noise_policy']
    policy._step_calls = saved['noise']['step_calls']
    policy.input_sigma = saved['noise']['input_sigma']
    policy.output_sigma = saved['noise']['output_sigma']
    torch.set_rng_state(saved['rng']['torch'])
    local['stream'].set_state(saved['rng']['data'])
    policy.input_stream.set_state(saved['rng']['input'])
    if policy.output_stream is not None:
        policy.output_stream.set_state(saved['rng']['output'])
    if state_hash(snapshot(local)) != state_hash(saved):
        raise RuntimeError('saved-state restoration was not exact')


def limited_host(source, start, end):
    tree = ast.parse(source)
    loops = [node for node in tree.body[0].body if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == 'step']
    if len(loops) != 1 or ast.unparse(loops[0].iter) != 'range(recipe.steps)':
        raise RuntimeError('outer host loop changed')
    original = deepcopy(tree)
    loops[0].iter = ast.parse(f'range({start - 1}, {end})', mode='eval').body
    ast.fix_missing_locations(tree)
    inverse = deepcopy(tree)
    inverse_loop = next(node for node in inverse.body[0].body if isinstance(node, ast.For)
                        and isinstance(node.target, ast.Name) and node.target.id == 'step')
    inverse_loop.iter = deepcopy(next(node for node in original.body[0].body
                                    if isinstance(node, ast.For)
                                    and isinstance(node.target, ast.Name)
                                    and node.target.id == 'step').iter)
    if ast.dump(original, include_attributes=False) != ast.dump(inverse, include_attributes=False):
        raise RuntimeError('local continuation changed more than loop bounds')
    return tree, ast.unparse(tree) + '\n'


def run_local(config, saved, *, start, end, opponent, source_dir):
    if opponent not in ('current', 'old', 'predicted'):
        raise ValueError(opponent)
    recipe, noise, _ = declared_recipe(prepared_config(config, 'constant'))
    policy = _noise_policy(noise, 1200)
    supports, checkpoints, rates, phases = [], [], [], []
    first = True
    first_gradient = None
    local_final = None
    with ExitStack() as stack:
        if opponent == 'old':
            stack.enter_context(patch.object(prediction_module, 'predicted_opponent',
                lambda base, materialized: [value.detach().clone() for value in base]))
        recorder, source = stack.enter_context(prediction_module.pr84_opponent_prediction(
            task='mode_hold', prediction=opponent != 'current'))
        ordinary_phases = recorder.phases

        def resume_phases(step, opt_d, opt_g, local):
            nonlocal first, first_gradient, local_final
            if first:
                if step + 1 != start:
                    raise RuntimeError('wrong resume update')
                restore(local, saved)
                first = False
            for phase in ordinary_phases(step, opt_d, opt_g, local):
                yield phase
                if phase == 2:
                    with torch.no_grad():
                        clean = getattr(local['generator'], 'model', local['generator'])
                        support = clean(local['prior'].z).detach().clone()
                    supports.append(dict(step=step + 1, support=support))
                    if step + 1 == start:
                        first_gradient = [g.detach().clone() for g in recorder.gg0]
                    phases.append(dict(step=step + 1, accepted_state_sha256=state_hash(snapshot(local))))
                    local_final = local
            for opt, role in ((opt_d, 'd'), (opt_g, 'g_prior')):
                rates.append(dict(step=step + 1, role=role,
                                  rates=[group['lr'] for group in opt.param_groups]))

        recorder.phases = resume_phases
        tree, limited_source = limited_host(source, start, end)
        target = source_dir / f'host-{start}-{end}.py'
        if target.exists() and target.read_text() != limited_source:
            raise RuntimeError('resumed source changed across opponents')
        target.write_text(limited_source)
        namespace = {}
        exec(compile(tree, '<saved-state-local-host>', 'exec'), mode_hold.__dict__, namespace)
        stack.enter_context(patch.object(mode_hold, 'train_mode_hold', namespace['train_mode_hold']))
        stack.enter_context(patch.object(mode_hold, 'checkpoint',
            lambda step, measure: checkpoints.append(dict(step=step, **measure()))))
        applied = []
        stack.enter_context(optimizer_defaults(recipe, applied))
        control = evaluate.FixedControl(vector_tasks.fixed_policy('cosine'), 1200)
        stack.enter_context(bridge.control_host_schedules(control))
        settings = mode_hold.ModeHoldRecipe(steps=end, particle_l2=0., vicreg_weight=recipe.prior_reg)
        common = candidate(recipe)
        mode_hold.train_mode_hold(settings, gan_factory=common.make_loss,
                                  cap_factory=common.make_penalty, diagnostics=False,
                                  noise_policy=policy)
    if first or len(supports) != end - start + 1:
        raise RuntimeError('local continuation did not complete')
    moment_steps = {role: sorted({int(state['step']) for state in opt.state.values()})
                    for role, opt in zip(('d', 'g'), recorder.optimizers)}
    if moment_steps != {'d': [end], 'g': [end]}:
        raise RuntimeError(f'Adam moments did not advance exactly once: {moment_steps}')
    for row in rates:
        expected = [.00425] if row['role'] == 'd' else [.00425, .0085]
        if row['rates'] != expected:
            raise RuntimeError(f'changed nominal rate: {row}')
    clean_rows = []
    for row in supports:
        indices, fixed_noise = fixed_draw(row['step'], row['support'])
        grade = score_support(row['support'], indices, fixed_noise, mode_hold.ring_means())
        checkpoint = next(x for x in checkpoints if x['step'] == row['step'])
        if any(grade[key] != checkpoint[key] for key in ('modes', 'hq')):
            raise RuntimeError('offline grade differs from actual host evaluation')
        clean_rows.append(dict(step=row['step'], support=row['support'].tolist(), grade=grade))
    receipt = recorder.receipt()
    return dict(opponent=opponent, start=start, end=end, points=clean_rows,
                accepted_states=phases, final_state_sha256=state_hash(snapshot(local_final)),
                moment_steps=moment_steps, rates=rates, noise=policy.receipt(),
                rng_final_sha256=state_hash(snapshot(local_final)['rng']),
                source_sha256=hashlib.sha256(limited_source.encode()).hexdigest(),
                dynamics=receipt), first_gradient


def gradient_comparison(a, b):
    an = sum(float(x.double().square().sum()) for x in a)
    bn = sum(float(x.double().square().sum()) for x in b)
    dot = sum(float((x.double() * y.double()).sum()) for x, y in zip(a, b))
    change = sum(float((x.double() - y.double()).square().sum()) for x, y in zip(a, b))
    return dict(cosine=dot / (an * bn) ** .5 if an and bn else None,
                relative_change=(change / an) ** .5 if an else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source_dir = args.output / 'source'
    source_dir.mkdir()
    diagnosis = json.loads((args.capture / 'diagnosis.json').read_text())
    states_file = args.capture / 'selected-states.pt'
    if diagnosis['status'] != 'EXACT_REFERENCE_PARITY' or hashlib.sha256(
            states_file.read_bytes()).hexdigest() != diagnosis['selected_states_sha256']:
        raise RuntimeError('capture provenance failed')
    states = torch.load(states_file, weights_only=True)
    original_rows = {row['step']: row for row in diagnosis['rows']}
    files = ['reports/toy100/pr84_prediction_state_filter.py',
             'reports/toy100/pr84_opponent_prediction.py',
             'reports/toy100/pr84_smoothed_candidate.py',
             'reports/toy100/alternating_curvature_scratch.py',
             'reports/toy100/extra_adam_scratch.py',
             'benchmarks/locked_shared/mode_hold.py',
             'configs/toy100/constraints_simple_regularization.json']
    hashes = {}
    for name in files:
        data = (ROOT / name).read_bytes()
        dest = source_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    declaration = dict(scope='saved_state_local_diagnostic_only', shared_gate_eligible=False,
        scratch_optimizer_policy=prediction_module.METHOD, seed=0, noise_horizon=1200,
        single_updates=list(states), short_continuations=[[1324, 1335], [1380, 1395], [1530, 1545]],
        opponents=['current', 'old', 'predicted'],
        old_scope='diagnostic old-opponent comparator, not a training candidate',
        sources=hashes, capture_sha256=hashlib.sha256((args.capture/'diagnosis.json').read_bytes()).hexdigest(),
        states_sha256=diagnosis['selected_states_sha256'])
    (args.output / 'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    config = json.loads((ROOT / files[-1]).read_text())
    rows = []
    for step, captured in states.items():
        variants, gradients = {}, {}
        for opponent in declaration['opponents']:
            value, grad = run_local(config, captured['pre_step'], start=step, end=step,
                                    opponent=opponent, source_dir=source_dir)
            if opponent == 'current':
                expected = state_hash(captured['post_bounded_g'])
                if value['accepted_states'][0]['accepted_state_sha256'] != expected:
                    raise RuntimeError(f'original accepted-state replay differs at {step}')
                if value['dynamics']['records'][0] != dict(original_rows[step]['stages']['record'], outer_step=1):
                    raise RuntimeError(f'original update receipt replay differs at {step}')
            elif value['rng_final_sha256'] != variants['current']['rng_final_sha256']:
                raise RuntimeError(f'counterfactual RNG consumption differs at {step}')
            variants[opponent], gradients[opponent] = value, grad
        row = dict(step=step, original_exact=True, variants=variants,
            fields={name: gradient_comparison(gradients['current'], gradients[name])
                    for name in ('old', 'predicted')})
        rows.append(row)
        (args.output / f'step-{step}.json').write_text(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(dict(event='SINGLE_UPDATE', step=step, original_exact=True,
            grades={name:value['points'][0]['grade'] for name,value in variants.items()}, fields=row['fields'])),flush=True)
    continuations = []
    for start, end in declaration['short_continuations']:
        variants = {}
        for opponent in ('current', 'predicted'):
            value, _ = run_local(config, states[start]['pre_step'], start=start, end=end,
                                  opponent=opponent, source_dir=source_dir)
            if opponent == 'current':
                for point in value['points']:
                    expected = original_rows[point['step']]['stages']['bounded_joint']
                    if point['support'] != expected:
                        raise RuntimeError(f'original short continuation differs at {point["step"]}')
            elif value['rng_final_sha256'] != variants['current']['rng_final_sha256']:
                raise RuntimeError('short continuation consumed different RNG')
            variants[opponent] = value
        row = dict(start=start,end=end,variants=variants)
        continuations.append(row)
        (args.output/f'continuation-{start}-{end}.json').write_text(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(dict(event='SHORT_CONTINUATION',start=start,end=end,
            grades={name:[p['grade'] for p in value['points']] for name,value in variants.items()})),flush=True)
    (args.output/'summary.json').write_text(json.dumps(dict(declaration=declaration,
        original_all_exact=True, singles=rows,continuations=continuations),allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
