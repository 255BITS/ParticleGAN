"""Read frozen CUDA states on CPU; no optimizer updates or RNG translation."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode = True
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
FIXES = ROOT.parent.parent
ATTEMPTS = FIXES.parent
PREV = ATTEMPTS/'scaling-portability-20260929'/'validation'
OLD = ATTEMPTS/'feature-cells-cuda-retest-20260929'/'learned'/'training'
NEW = FIXES/'validation'/'learned'/'training'
PKG = FIXES/'pkg-CB64-RA2'
sys.path.insert(0, str(PREV))
from models_metrics import networks, Evaluator, encode, oracle_centres
sys.path.insert(0, str(PKG))
from particlegan.feature_cells import BoundedLatentGeometry, bounded_jitter


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(x):
    return hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()


def stats(x):
    x = x.detach().double().flatten()
    return dict(min=float(x.min()), median=float(x.median()), mean=float(x.mean()),
                q95=float(torch.quantile(x, .95)), max=float(x.max()),
                rms=float(x.square().mean().sqrt()), finite=bool(torch.isfinite(x).all()))


def rms(x):
    return float(x.detach().double().square().mean().sqrt())


def cosine(a, b):
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    return float((a @ b)/(a.norm()*b.norm()).clamp_min(1e-30))


def json_safe(value, path='', nonfinite=None):
    """Keep archived undefined tester diagnostics visible in strict JSON."""
    if nonfinite is None:
        nonfinite = []
    if isinstance(value, dict):
        return {k:json_safe(v, path+'.'+str(k), nonfinite) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v, path+'.'+str(k), nonfinite) for k,v in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        label = 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
        nonfinite.append(dict(path=path, saved_value=label))
        return label
    return value


def cat_params(module):
    return torch.cat([p.detach().flatten() for p in module.parameters()])


def model(problem, state, name):
    with torch.random.fork_rng(devices=[]):
        G, D = networks(problem)
    out = D if name == 'D' else G
    out.load_state_dict(state['models'][name]); out.eval()
    return out


@torch.no_grad()
def toy_score(x):
    d, ids = torch.cdist(x, oracle_centres()).min(1)
    accepted = d <= .09
    mass = torch.bincount(ids[accepted], minlength=25).double()/len(x)
    return dict(precision=float(accepted.float().mean()), coverage=int((mass >= .01).sum()),
                mass_tv=float((mass-.04).abs().sum()/2+(~accepted).double().mean()/2),
                centre_distance=float(d.mean()), supported_mass=mass.tolist(),
                all_output_mode_counts=torch.bincount(ids, minlength=25).tolist(),
                output_coordinates=[stats(c) for c in x.T])


@torch.no_grad()
def image_score(x, evaluator):
    p, f = encode(evaluator, x)
    confidence, ids = p.max(1)
    return dict(predicted_class_mass=(torch.bincount(ids, minlength=10).double()/len(x)).tolist(),
                confident_fraction=float((confidence >= .9).float().mean()),
                mean_classifier_confidence=float(confidence.mean()),
                pixel_abs_ge_099_fraction=float((x.abs() >= .99).float().mean()),
                pixel_rms=rms(x), embedding_mean=f.mean(0).tolist())


def optimizer_summary(state):
    result = []
    for oi, opt in enumerate(state['optimizers']):
        groups = []
        for group in opt['param_groups']:
            row = {k: v for k, v in group.items() if k != 'params'}
            row['parameters'] = []
            for key in group['params']:
                v = opt['state'][key]
                row['parameters'].append(dict(id=key, shape=list(v['exp_avg'].shape),
                    step=float(v['step']), last_gradient=stats(v['exp_avg']),
                    second_moment=stats(v['exp_avg_sq']), max_second_moment=stats(v['max_exp_avg_sq'])))
                assert bool(torch.isfinite(v['exp_avg']).all())
                assert bool(torch.isfinite(v['exp_avg_sq']).all())
                assert bool(torch.isfinite(v['max_exp_avg_sq']).all())
                assert float(v['step']) == state['completed_steps']
            groups.append(row)
        extra = opt['regularizer']
        if oi == 0:
            latent = extra['latent']
            regularizer = dict(direct_present=extra['direct'] is not None,
                latent_state=latent['state'], latent_history=stats(latent['history']),
                latent_observation_fraction=latent['state']['observed']/latent['state']['total'])
            pid = opt['param_groups'][1]['params'][0]
            moment = opt['state'][pid]['exp_avg']
            regularizer['last_prior_gradient_active_rows'] = int((moment.norm(dim=1) > 0).sum())
            regularizer['last_prior_gradient_matches_history_max_error_on_active'] = float(
                (moment[(moment.norm(dim=1)>0)]-latent['history'][(moment.norm(dim=1)>0)]).abs().max())
        else:
            record = extra['record']
            regularizer = dict(guard=extra['guard'], record={k: v for k, v in record.items() if k != 'sur_hist'},
                               surprise_history=stats(torch.tensor(record['sur_hist'])))
        result.append(dict(groups=groups, regularizer=regularizer))
    return result


def summary_testers(state):
    return [[None if t is None else {k: v for k, v in t.items()
              if k in ('s', 'b', 'tau', 'windows', 'last_decisive', 'last_decisive_scale',
                       'hold_descent', 'release_rule', 'b_anchor', 'looks', 'counters')}
             for t in row] for row in state['lr_settle']]


@torch.no_grad()
def exact_delta(z, points, width, noise):
    nearest = z.new_full((len(z),), float('inf'))
    for block in points.split(max(1, 2048//points.shape[1])):
        distance = (z[:, None]-block[None]).square().sum(2)
        distance.masked_fill_(distance == 0, float('inf'))
        nearest = torch.minimum(nearest, distance.min(1).values)
    radius = nearest.sqrt()*.5
    radius = torch.where(torch.isfinite(radius), radius, torch.zeros_like(radius))
    delta = width*noise
    return delta*(radius/delta.norm(dim=1).clamp_min(1e-20)).clamp_max(1)[:, None]


def gradient_probe(problem, state, variant, real, noise, output_noise):
    G, D = model(problem, state, 'G'), model(problem, state, 'D')
    D.requires_grad_(False)
    table = state['models']['prior']['z'].detach().clone().requires_grad_(True)
    prior = SimpleNamespace(z=table)
    ids = torch.arange(128)
    latent = table[ids]
    width = state['controller']['latent_bandwidth']
    if variant == 'CB64-RA2':
        kernel = BoundedLatentGeometry(rank=8, neighbors=64, chunk=256)
        delta = kernel.displacement(latent, prior, width, noise[:128])
    elif variant == 'CB64-RA':
        delta = bounded_jitter(latent, noise[:128])
    else:
        delta = exact_delta(latent, table.detach(), width, noise[:128])
    # The neighborhood fit is observational: only the sampled table rows
    # should receive an identity derivative from latent+delta to the table.
    perturbation_grad = torch.autograd.grad((latent+delta).sum(), table, retain_graph=True)[0]
    expected_grad = torch.zeros_like(table); expected_grad[ids] = 1
    identity_error = float((perturbation_grad-expected_grad).abs().max())
    assert identity_error == 0
    gradients = {}; metrics = {}
    sigma = float(state['output_noise']['log_sigma'].exp().clamp_min(state['recipe']['output_noise_std']))
    for label, offset in (('no_latent_jitter', torch.zeros_like(delta)), ('saved_variant_kernel', delta)):
        fake = G(latent+offset)
        if problem == 'toy':
            fake = fake+sigma*output_noise[:128]
        d_real, d_fake = D(real), D(fake)
        loss = F.softplus(-(d_fake-d_real)).mean()
        grads = torch.autograd.grad(loss, [table, *G.parameters()], retain_graph=True)
        assert all(bool(torch.isfinite(g).all()) for g in grads)
        gradients[label] = grads
        metrics[label] = dict(loss_gan=float(loss.detach()), real_logit_mean=float(d_real.mean().detach()),
            fake_logit_mean=float(d_fake.mean().detach()), critic_real_minus_fake=float((d_real-d_fake).mean().detach()),
            prior_gradient_rms=rms(grads[0]), prior_active_rows=int((grads[0].norm(dim=1)>0).sum()),
            generator_gradient_rms=rms(torch.cat([g.flatten() for g in grads[1:]])))
        if problem == 'toy':
            metrics[label]['conditional_128_row_output_metrics'] = toy_score(fake.detach())
    clean, noisy = gradients['no_latent_jitter'], gradients['saved_variant_kernel']
    active_cosine = F.cosine_similarity(clean[0][ids].double(), noisy[0][ids].double(), dim=1)
    # AMSGrad moments are inspected without stepping or modifying them.
    prior_opt = state['optimizers'][0]
    group = prior_opt['param_groups'][1]
    moment = prior_opt['state'][group['params'][0]]
    denominator = (moment['max_exp_avg_sq']/(1-group['betas'][1]**float(moment['step']))).sqrt()+group['eps']
    predicted = -group['lr']*noisy[0][ids]/denominator[ids]
    result = dict(sampled_rows=ids.tolist(), real_batch_sha256=tensor_sha(real),
        saved_cuda_replay=False, optimizer_updated=False, output_noise='saved fold2' if problem == 'toy' else 'omitted',
        latent_displacement_rms=rms(delta), gradient_identity_max_error=identity_error,
        methods=metrics, prior_gradient_cosine=cosine(clean[0], noisy[0]),
        per_row_prior_gradient_cosine=stats(active_cosine),
        prior_gradient_opposed_rows=int((active_cosine < 0).sum()),
        generator_gradient_cosine=cosine(torch.cat([g.flatten() for g in clean[1:]]),
                                        torch.cat([g.flatten() for g in noisy[1:]])),
        saved_moment_preconditioned_raw_prior_step_rms=rms(predicted))
    if problem == 'toy':
        # Oracle labels enter evaluation of the direction only, never the GAN
        # loss, kernel, fit, optimizer or controller.
        raw = G(latent)
        target = oracle_centres()[torch.cdist(raw.detach(), oracle_centres()).argmin(1)]
        distance_grad = torch.autograd.grad((raw-target).square().sum(), table)[0][ids]
        directional = (distance_grad*predicted).sum(1)
        result['evaluation_only_prior_step_toward_nearest_mode_fraction'] = float((directional < 0).float().mean())
        result['evaluation_only_nearest_mode_directional_derivative'] = stats(directional)
    return result


def inspect(path, problem, variant, initial, real, noise, output_noise, evaluator):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    state = saved['trainer']
    step = state['completed_steps']
    served_ema = state['recipe']['serve_average'] > 0 and state['lr_settle'][0][1]['last_decisive'] == -1
    G, EG = model(problem, state, 'G'), model(problem, state, 'ema_G')
    z, ez = state['models']['prior']['z'], state['models']['ema_prior']['z']
    case = dict(problem=problem, variant=variant, step=step, checkpoint_sha256=sha(path),
        saved_device=state['device'], actual_serving_pair='ema_G/ema_prior' if served_ema else 'G/prior',
        served_ema=served_ema, original_cuda_record=saved['record'],
        model_tensor_summaries={k: stats(torch.cat([v.flatten() for v in values.values()]))
                                for k, values in state['models'].items()},
        prior=dict(coordinate_spread=stats(z.std(0, unbiased=False)),
            row_norm=stats(z.norm(dim=1)), exact_unique_rows=len(torch.unique(z, dim=0)),
            init_difference_rms=rms(z-initial['prior']['z']),
            fast_ema_difference_rms=rms(z-ez)),
        generator_fast_ema_difference_rms=rms(cat_params(G)-cat_params(EG)),
        output_sigma=saved['record']['diagnostics']['output_sigma'],
        output_noise_log_sigma=float(state['output_noise']['log_sigma']),
        rng_bytes={k: v.numel() for k,v in state['streams'].items()},
        saved_cpu_rng_bytes=state['cpu_rng'].numel(), saved_cuda_rng_bytes=state['cuda_rng'].numel(),
        test_state=summary_testers(state), optimizer_state=optimizer_summary(state),
        controller={k: v for k,v in state['controller'].items() if k in (
            'updates','reopens','closed','mobility','alignment','last_cosine','game_trust','game_ratio','payoff_error')},
        row_evidence=dict(fraction=state['row_evidence']['fraction'], valid=state['row_evidence']['valid'],
                          counters=state['row_evidence']['counters']), pairs={})
    for name, g, points in (('fast', G, z), ('ema', EG, ez), ('fast_G_ema_prior', G, ez), ('ema_G_fast_prior', EG, z)):
        with torch.no_grad():
            output = torch.cat([g(b) for b in points.split(128)])
            if problem == 'toy':
                case['pairs'][name] = dict(clean=toy_score(output),
                    matched_output_noise=toy_score(output+case['output_sigma']*output_noise))
            else:
                case['pairs'][name] = dict(clean=image_score(output, evaluator))
    case['gradient_probe'] = gradient_probe(problem, state, variant, real, noise, output_noise)
    return case


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    noise_path = FIXES/'geometry'/'gpu-inputs.pt'
    loaded = torch.load(noise_path, map_location='cpu', weights_only=False)
    noise = next(x for x in loaded['cases'] if x['name'] == 'fold128')['noise'][:1024]
    output_noise = next(x for x in loaded['cases'] if x['name'] == 'fold2')['noise'][:1024]
    toy = torch.load(PREV/'data'/'toy-stream.pt', map_location='cpu', weights_only=True)['points']
    indices = torch.load(PREV/'data'/'image-stream.pt', map_location='cpu', weights_only=True)['indices']
    from torchvision.datasets import MNIST
    images = MNIST(PREV/'data', train=True, download=False).data
    evaluator = Evaluator().eval()
    evaluator.load_state_dict(torch.load(PREV/'evaluator.pt', map_location='cpu', weights_only=True)['model'])
    cases = []
    for problem in ('toy', 'mnist'):
        for variant in ('E22', 'CB64-RA', 'CB64-RA2'):
            folder = (NEW if variant == 'CB64-RA2' else OLD)/problem/variant
            for step in (1000, 2000):
                cases.append((folder/f'checkpoint-{step:04d}.pt', folder/'checkpoint-0000.pt', problem, variant, step))
    sources = [Path(__file__), ROOT/'PROTOCOL.md', PREV/'models_metrics.py', PREV/'evaluator.pt',
        PREV/'data'/'toy-stream.pt', PREV/'data'/'image-stream.pt', noise_path, FIXES/'geometry'/'GPU-INPUTS.json']
    sources += list((PREV/'data'/'MNIST'/'raw').glob('*ubyte'))
    for filename in ('training.py','feature_cells.py','particle_prior.py','continuous.py','k3p.py','ka2.py','recipes.py'):
        sources.append(PKG/'particlegan'/filename)
    sources += [p for c in cases for p in c[:2]]
    hashes = {str(p): sha(p) for p in dict.fromkeys(sources)}
    result = dict(scope='CPU conditional saved-state diagnosis; no training update or CUDA continuation',
        new_seeds=0, python_executable=sys.executable, torch_version=torch.__version__,
        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], source_sha256=hashes,
        latent_noise_sha256=tensor_sha(noise), output_noise_sha256=tensor_sha(output_noise), cases=[])
    for path, init_path, problem, variant, step in cases:
        initial = torch.load(init_path, map_location='cpu', weights_only=False)['trainer']['models']
        pos = (2*(step-1)+1)*128
        real = toy[pos:pos+128] if problem == 'toy' else images[indices[pos:pos+128]].float().unsqueeze(1)/127.5-1
        begin = time.perf_counter()
        row = inspect(path, problem, variant, initial, real, noise, output_noise, evaluator)
        row['diagnostic_seconds'] = time.perf_counter()-begin
        result['cases'].append(row)
        print(json.dumps(dict(event='case',problem=problem,variant=variant,step=step,
            seconds=row['diagnostic_seconds'],serving=row['actual_serving_pair'],
            prior_spread=row['prior']['coordinate_spread']['mean'],
            fast_ema_prior_rms=row['prior']['fast_ema_difference_rms'],
            clean={name:{k:v for k,v in pair['clean'].items() if k in (
                'precision','coverage','mass_tv','confident_fraction','mean_classifier_confidence')}
                   for name,pair in row['pairs'].items()},
            gradient_identity_error=row['gradient_probe']['gradient_identity_max_error'],
            prior_gradient_cosine=row['gradient_probe']['prior_gradient_cosine'],
            generator_gradient_cosine=row['gradient_probe']['generator_gradient_cosine'])), flush=True)
    result.update(cuda_initialized=torch.cuda.is_initialized(), sources_unchanged=hashes == {p:sha(p) for p in hashes})
    assert result['sources_unchanged'] and not result['cuda_initialized']
    nonfinite = []
    result = json_safe(result, nonfinite=nonfinite)
    result['archived_nonfinite_diagnostic_fields'] = nonfinite
    (ROOT/'state-diagnosis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(event='complete',cases=len(result['cases']),cuda_initialized=result['cuda_initialized'],
        sources_unchanged=result['sources_unchanged'])), flush=True)


if __name__ == '__main__':
    main()
