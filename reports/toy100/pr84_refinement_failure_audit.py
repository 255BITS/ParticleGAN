"""One fixed-bank replay of a captured critic-fit numerical exception.

This makes no outer training update and selects no training candidate. The
captured pre-fit critic, bank, metric and original frozen solver are reused.
Run against the source-bound repository that produced the capture.
"""

import argparse
import gzip
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_bytes(folder, name):
    plain = folder / name
    if plain.exists():
        return plain.read_bytes()
    packed = folder / (name + '.gz')
    if packed.exists():
        with gzip.open(packed, 'rb') as handle:
            return handle.read()
    raise FileNotFoundError(plain)


def tensor_stats(values):
    tensors = list(values)
    finite = [bool(torch.isfinite(x).all()) for x in tensors]
    finite_parts = [x.detach()[torch.isfinite(x)] for x in tensors]
    maximum = max((float(x.abs().max()) for x in finite_parts if x.numel()), default=None)
    return dict(tensors=len(tensors), finite_tensors=sum(finite), finite_all=all(finite),
                nan_elements=sum(int(torch.isnan(x).sum()) for x in tensors),
                infinite_elements=sum(int(torch.isinf(x).sum()) for x in tensors),
                finite_max_abs=maximum)


def scalar(value):
    if isinstance(value, torch.Tensor):
        value = float(value.detach()) if value.numel() == 1 else repr(value)
    elif isinstance(value, (float, int)):
        value = float(value)
    else:
        return repr(value)
    return value if isinstance(value, float) and math.isfinite(value) else repr(value)


def same_tensors(first, second):
    return len(first) == len(second) and all(
        a is not None and b is not None and a.shape == b.shape and a.dtype == b.dtype
        and a.detach().contiguous().cpu().numpy().tobytes()
        == b.detach().contiguous().cpu().numpy().tobytes()
        for a, b in zip(first, second))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    declaration = json.loads((args.original / 'declaration.json').read_text())
    sources = declaration['source']
    for name, digest in sources.items():
        if sha(root / name) != digest:
            raise RuntimeError(f'frozen source mismatch: {name}')
    sys.path.insert(0, str(root))
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
    from reports.toy100 import pr84_critic_relaxation as fit
    from reports.toy100 import pr84_critic_refinement_cold as cold
    from reports.toy100.pr84_prediction_state_filter import state_hash
    from reports.toy100.coverage_fixed_eval import fixed_draw, score_support
    from benchmarks.locked_shared import mode_hold

    torch.set_num_threads(1)
    capture = json.loads(artifact_bytes(args.capture, 'failure.json'))
    payload_bytes = artifact_bytes(args.capture, 'failed-fit.pt')
    payload_sha = hashlib.sha256(payload_bytes).hexdigest()
    if payload_sha != capture['state_file_sha256']:
        raise RuntimeError('captured tensor file changed')
    payload = torch.load(BytesIO(payload_bytes), weights_only=True)
    if state_hash(payload['pre_step']) != capture['saved_pre_step_sha256']:
        raise RuntimeError('captured pre-step state changed')
    if state_hash(payload['post_accepted_d']) != capture['accepted_d_sha256']:
        raise RuntimeError('captured accepted-D state changed')
    if payload['host_update'] != capture['host_update']:
        raise RuntimeError('captured update number changed')
    config = json.loads((args.original / 'config.json').read_text())
    recipe, _, _ = declared_recipe(config)
    gan, regularizer = recipe.make_loss(), recipe.make_gradient_penalty()
    with torch.random.fork_rng(devices=[]):
        pre_g, pre_d, pre_prior = fit.modules(payload['pre_step'])
        accepted_g, accepted_d, accepted_prior = fit.modules(payload['post_accepted_d'])
        with torch.no_grad():
            clean = pre_g(pre_prior.z)
            index, noise = fixed_draw(payload['host_update'] - 1, clean)
            posthoc_grade = score_support(clean, index, noise, mode_hold.ring_means())
        if not same_tensors(list(pre_g.parameters()), list(accepted_g.parameters())) or not same_tensors(
                list(pre_prior.parameters()), list(accepted_prior.parameters())):
            raise RuntimeError('accepted D changed G or prior before its fit')
        saved_finiteness = dict(
            pre_step_g=tensor_stats(pre_g.parameters()),
            pre_step_prior=tensor_stats(pre_prior.parameters()),
            pre_step_d=tensor_stats(pre_d.parameters()),
            post_accepted_d=tensor_stats(accepted_d.parameters()),
            clean_support=tensor_stats([clean]),
            optimizer_d_moments=tensor_stats([
                x for state in payload['post_accepted_d']['optimizer_d']['state'].values()
                for x in state.values() if isinstance(x, torch.Tensor)]),
            optimizer_g_moments=tensor_stats([
                x for state in payload['post_accepted_d']['optimizer_g']['state'].values()
                for x in state.values() if isinstance(x, torch.Tensor)]))
        _, critic, _ = fit.modules(payload['post_accepted_d'])
        rng = torch.get_rng_state().clone()
        accepted_state = {key: value.detach().clone() for key, value in critic.state_dict().items()}
        parameter_before = [p.detach().clone() for p in critic.parameters()]
        accepted = tensor_stats(parameter_before)
        bank = payload['bank']
        bank_stats = tensor_stats([bank['real'], bank['fake']]
                                  + ([] if bank['input_noise'] is None else bank['input_noise']))
        metric_stats = tensor_stats(payload['metric'])
        best = payload['best']
        if best is None:
            raise RuntimeError('capture has no finite best point')
        critic.load_state_dict(best)
        with patch.object(fit, 'd_loss', cold.cached_d_loss), torch.enable_grad():
            best_loss, best_logistic, best_penalty = cold.cached_d_loss(
                critic, bank, gan, regularizer, payload['host_update'])
            best_gradient = fit.gradients(best_loss, critic)
        best_row = dict(total_loss=float(best_loss.detach()), logistic=float(best_logistic.detach()),
                        penalty=float(best_penalty.detach()), gradient=tensor_stats(best_gradient),
                        parameters=tensor_stats(critic.parameters()),
                        agrees_with_recorded_best=abs(float(best_loss.detach()) - capture['best_loss']) <= 1e-7)
        # Reset the exact accepted-D point before the one replay. `modules`
        # strips the host wrapper when it initially materializes the critic.
        critic.load_state_dict(accepted_state)
        if not same_tensors(list(critic.parameters()), parameter_before):
            raise RuntimeError('accepted-D reset failed')
        replay = dict(raised=False)
        with patch.object(fit, 'd_loss', cold.cached_d_loss), torch.enable_grad():
            try:
                fit.relax(critic, bank, gan, regularizer, payload['host_update'], payload['metric'])
            except FloatingPointError as error:
                from torch.optim.lbfgs import _cubic_interpolate
                frames = []
                tb = error.__traceback__
                while tb is not None:
                    frames.append((tb.tb_frame.f_code.co_name, tb.tb_frame.f_locals))
                    tb = tb.tb_next
                closure = next(values for name, values in frames if name == 'closure')
                wolfe = next((values for name, values in frames if name == '_strong_wolfe'), {})
                cubic = None
                if all(key in wolfe for key in ('bracket', 'bracket_f', 'bracket_gtd')):
                    x1, x2 = wolfe['bracket']
                    f1, f2 = wolfe['bracket_f']
                    g1, g2 = wolfe['bracket_gtd']
                    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
                    d2_square = d1 ** 2 - g1 * g2
                    cubic = dict(bracket=[scalar(x1), scalar(x2)],
                                 losses=[scalar(f1), scalar(f2)],
                                 derivatives=[scalar(g1), scalar(g2)],
                                 d1=scalar(d1), d1_square=scalar(d1 ** 2),
                                 d2_square=scalar(d2_square),
                                 recomputed_t=scalar(_cubic_interpolate(x1, f1, g1, x2, f2, g2)))
                replay = dict(raised=True, message=str(error), loss=scalar(closure['loss']),
                              logistic=scalar(closure['logistic']), penalty=scalar(closure['penalty']),
                              gradient=tensor_stats([p.grad for p in critic.parameters()]),
                              parameters=tensor_stats(critic.parameters()),
                              trial_parameters_match_capture=same_tensors(
                                  list(critic.state_dict().values()),
                                  list(payload['invalid_critic'].values())),
                              trial_gradients_match_capture=same_tensors(
                                  [p.grad for p in critic.parameters()], payload['invalid_gradient']),
                              wolfe={key: scalar(wolfe[key]) for key in
                                     ('t', 't_prev', 'f', 'f_new', 'f_prev',
                                      'gtd', 'gtd_new', 'ls_iter') if key in wolfe},
                              wolfe_direction=tensor_stats([wolfe['d']]) if 'd' in wolfe else None,
                              wolfe_previous_trial_gradient=tensor_stats([wolfe['g_new']])
                              if 'g_new' in wolfe else None,
                              cubic_interpolation=cubic)
        if not replay['raised'] or replay['message'] != 'nonfinite local critic fit':
            raise RuntimeError('captured numerical exception did not reproduce')
        if not torch.equal(rng, torch.get_rng_state()):
            raise RuntimeError('one-fit replay consumed a global RNG draw')
    result = dict(scope='single captured critic fit; no outer training',
                  capture_sha256=payload_sha, capture_source_sha256=capture['observer_sha256'],
                  audit_source_sha256=sha(Path(__file__)), torch_version=torch.__version__,
                  host_update=payload['host_update'], source=sources,
                  fixed_eval_source_sha256=sha(root / 'reports/toy100/coverage_fixed_eval.py'),
                  posthoc_pre_step_fixed_eval=dict(step=payload['host_update'] - 1,
                                                    grade=posthoc_grade,
                                                    scope='one frozen-state observation, not the cold quality gate'),
                  saved_finiteness=saved_finiteness,
                  finite_records=len(payload['finite_records']),
                  recorded_initial_loss=capture['initial_loss'], recorded_best_loss=capture['best_loss'],
                  accepted_parameters=accepted, bank=bank_stats, metric=metric_stats,
                  best_finite=best_row, recorded_invalid_parameters=tensor_stats(payload['invalid_critic'].values()),
                  recorded_invalid_gradients=tensor_stats(
                      [x for x in payload['invalid_gradient'] if x is not None]),
                  replay=replay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='REPRODUCED', host_update=result['host_update'],
                          best_finite=best_row['agrees_with_recorded_best'], replay=replay), allow_nan=False))


if __name__ == '__main__':
    main()
