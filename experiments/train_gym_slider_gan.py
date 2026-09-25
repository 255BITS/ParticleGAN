#!/usr/bin/env python
"""Train the previous-action GAN with Anima-style paired-error supervision."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.config import read_config
from experiments.train_gym_transition import parameter_count, sha256, write_json
from lib.gym_control import build_expert_records
from lib.gym_state_control import training_recipe
from lib.gym_transition import GymTransitionScaler
from lib.gym_previous_gan import fake_paths, real_record, adversarial_loss
from lib.gym_slider_gan import MODULE_KEYS, build_models, hashes, paired_loss, error_loss
from particlegan import scale_learning_rates

DEFAULTS = dict(arm='sliders', steps=2500, batch_size=256,
    checkpoints=[250, 1000, 2500], log_interval=250, seed=24003, device='cuda:1',
    z_dim=32, num_particles=1024, width=128, encoder_width=128, context_dim=11,
    d_width=256, marginal_width=128, marginal_weight=1., adversarial_weight=1.,
    action_weight=1., lambda_state=1., lambda_next=1., continuous_weight=1., contact_weight=1.,
    episodes='results/gym/lunar_lander/data/episodes.json',
    slider_scope='all', paired_error_weight=1., error_tokens=8, error_width=48, error_heads=4,
    out_dir='results/gym/lunar_lander_slider_gan/sliders_all',
    live_log='results/gym/lunar_lander_slider_gan/live.log')


def validate(cfg):
    if set(cfg) != set(DEFAULTS) or cfg['arm'] != 'sliders':
        raise ValueError('Unexpected slider GAN configuration')
    if cfg['slider_scope'] not in ('all', 'action'):
        raise ValueError('slider_scope must be all or action')
    if any(type(cfg[k]) is not int or cfg[k] < 1 for k in ('error_tokens', 'error_width', 'error_heads')) or cfg['error_width'] % cfg['error_heads']:
        raise ValueError('Invalid error critic dimensions')
    for key in ('steps', 'batch_size', 'log_interval', 'z_dim', 'num_particles', 'width',
                'encoder_width', 'd_width', 'marginal_width'):
        if type(cfg[key]) is not int or cfg[key] < (2 if key in ('batch_size', 'num_particles') else 1):
            raise ValueError(f'Invalid {key}')
    if type(cfg['seed']) is not int or cfg['context_dim'] != 11:
        raise ValueError('Integer seed and terrain11 required')
    for key in ('marginal_weight', 'adversarial_weight', 'action_weight', 'lambda_state',
                'lambda_next', 'continuous_weight', 'contact_weight', 'paired_error_weight'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must remain positive')
    if not isinstance(cfg['checkpoints'], list) or any(type(s) is not int or s <= 0 for s in cfg['checkpoints']):
        raise ValueError('Invalid checkpoints')
    if str(cfg['device']).startswith('cuda') and cfg['device'] != 'cuda:1':
        raise ValueError('Use GPU 1; GPU 0 belongs to the user')


def source_paths():
    names = ['experiments/train_gym_slider_gan.py', 'lib/gym_slider_gan.py', 'lib/gym_previous_gan.py',
             'lib/gym_control.py', 'lib/gym_state_control.py', 'lib/gym_transition.py',
             'experiments/train_gym_transition.py', 'experiments/config.py']
    return ([ROOT / n for n in names] + sorted((ROOT / 'particlegan').glob('*.py')) +
            sorted((ROOT / 'configs/gym/lunar_lander_slider_gan').glob('*.yaml')) +
            sorted(p for p in (ROOT / 'lib/vendor/concept_slider_core').glob('*') if p.is_file()))


def train(cfg):
    validate(cfg)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device, out, live = torch.device(cfg['device']), Path(cfg['out_dir']), Path(cfg['live_log'])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {out}')
    live.parent.mkdir(parents=True, exist_ok=True)
    records = build_expert_records(cfg['episodes'])
    triples = np.concatenate([records[k] for k in ('states', 'actions', 'next_states')], 1)
    scaler = GymTransitionScaler.fit(triples)
    bundle = build_models(cfg, scaler, scaler(torch.from_numpy(triples)), device)
    np.savez_compressed(out / 'expert_records.npz', **records)
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths()}
    with zipfile.ZipFile(out / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, digest in sources.items():
            value = (ROOT / name).read_bytes()
            if hashlib.sha256(value).hexdigest() != digest:
                raise RuntimeError('Source changed during capture')
            archive.writestr(name, value)
    provenance = dict(sources=sources, source_archive_sha256=sha256(out / 'source.zip'),
        episodes=dict(path=cfg['episodes'], sha256=sha256(cfg['episodes'])),
        expert_data=dict(count=len(triples), episode_ids=np.unique(records['episode_ids']).tolist(),
            npz_sha256=sha256(out / 'expert_records.npz'),
            arrays={k: dict(shape=list(v.shape), sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in records.items()}),
        initial_parameters=hashes(bundle), initialization='All G/E/prior/D/R from scratch; no checkpoint or pretrained scaler',
        control_input='Current state, previous action, terrain; teacher previous actions in training, own actions in rollout',
        normalization='Shared state statistics from training current/successor states; action statistics from all training expert actions',
        gan_training=True, adversarial_semantics='Prior and control-encoded full triples; average paths per role; no synthetic cycle',
        supervision=f'All expert training triples; sliders scope={cfg["slider_scope"]}; all removes MSE and BCE from optimization',
        error_normalization='Per-coordinate sample std of training target minus training mean, then median row RMS gain; no held-out inputs',
        slider_source='Anima paired-error game; shared core pinned at beaffeb3640c4554a7315998c04a5909f384b972')
    values = {k: torch.as_tensor(v, device=device) for k,v in records.items() if k not in ('episode_ids', 'steps')}
    recipe = training_recipe(cfg)
    opt_g, opt_d = recipe.make_optimizers(bundle['G'], bundle['D'], bundle['prior'],
        encoder=bundle['E'], ema_critic=copy.deepcopy(bundle['D']), fused=device.type == 'cuda')
    opt_r = recipe.make_critic_optimizer(bundle['R'], ema_critic=copy.deepcopy(bundle['R']), fused=device.type == 'cuda')
    optimizers = (opt_g, opt_d, opt_r)
    rates = [[g['lr'] for g in opt.param_groups] for opt in optimizers]
    gan, spread = recipe.make_loss(), recipe.make_prior_regularizer()
    r_penalty = recipe.make_critic_penalty(opt_r)
    ema = {**bundle, **{k: copy.deepcopy(bundle[k]).eval().requires_grad_(False) for k in ('G', 'E', 'prior')}}
    rng = {k: torch.Generator(device=device).manual_seed(cfg['seed'] + offset)
           for k,offset in dict(data=11, d_data=31, latent=51, contact=61, d_latent=71, d_contact=81, error_noise=151, d_error_noise=161).items()}
    reg_rng = {role: torch.Generator(device=device).manual_seed(cfg['seed'] + 100 + i)
               for i,role in enumerate(bundle['D'].roles())}
    penalties = {role: recipe.make_critic_penalty(opt_d) for role, generator in reg_rng.items()}
    digests = {k: hashlib.sha256() for k in ('data', 'd_data')}
    def batch(name):
        ids = torch.randint(len(triples), (cfg['batch_size'],), device=device, generator=rng[name])
        digests[name].update(ids.cpu().numpy().tobytes())
        return {k: v[ids] for k,v in values.items()}
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    counts = {k: parameter_count(bundle[k]) for k in MODULE_KEYS}
    write_json(out / 'provenance.json', provenance)
    write_json(out / 'recipe.json', recipe.to_dict())
    write_json(out / 'error_normalization.json', dict(scope=cfg['slider_scope'],
        target_mean=bundle['R'].target_mean.cpu().tolist(), scale=bundle['R'].target_std.cpu().tolist(),
        edit_rms=float(bundle['R'].edit_rms), noise_start=bundle['R'].sigma(1),
        noise_hold=1., noise_floor=.03, noise_horizon=cfg['steps'],
        dimensions=len(bundle['R'].target_std), normalization=bundle['R'].normalization))
    write_json(out / 'normalization.json', {k: v.cpu().tolist() for k,v in scaler.state_dict().items()})
    write_json(out / 'environment.json', dict(python=sys.version, torch=str(torch.__version__),
        cuda=torch.version.cuda, device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None))
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg))
    checkpoints = sorted({s for s in cfg['checkpoints'] if s <= cfg['steps']} | {cfg['steps']})
    with (out / 'log.txt').open('w', buffering=1) as logfile, live.open('a', buffering=1) as livefile, \
         (out / 'metrics.jsonl').open('w', buffering=1) as metrics:
        def log(message):
            message = f'[{out.name}] {message}'
            print(message, flush=True)
            logfile.write(message + '\n')
            livefile.write(message + '\n')
        log(f'START steps={cfg["steps"]} records={len(triples)} episodes={len(np.unique(records["episode_ids"]))} device={device}')
        log('Scratch E(st, at-1, terrain) -> z; G1 -> st; G2 -> at; G3 -> st+1; joint + marginal GAN + paired-error critic throughout')
        sync()
        started = segment = time.perf_counter()
        optimization_seconds = 0.
        for step in range(1, cfg['steps'] + 1):
            scale, _ = scale_learning_rates(step - 1, recipe, optimizers, rates, bundle['prior'])
            opt_g.zero_grad(set_to_none=True)
            bundle['D'].requires_grad_(True)
            bundle['R'].requires_grad_(True)
            db = batch('d_data')
            with torch.no_grad():
                df, d_decoded = fake_paths(bundle, db, rng['d_latent'], rng['d_contact'])
            dl, dt = adversarial_loss(bundle['D'], real_record(bundle, db), df, db['terrain'], gan,
                reg=penalties)
            if not torch.isfinite(dl):
                raise FloatingPointError(f'Nonfinite D loss at {step}')
            rl, rt = error_loss(bundle['R'], d_decoded, real_record(bundle, db), step, rng['d_error_noise'], penalty=r_penalty)
            if not torch.isfinite(rl):
                raise FloatingPointError(f'Nonfinite paired-error D loss at {step}')
            opt_r.zero_grad(set_to_none=True)
            rl.backward()
            opt_r.step()
            opt_r.zero_grad(set_to_none=True)
            bundle['R'].requires_grad_(False)
            opt_d.zero_grad(set_to_none=True)
            dl.backward()
            opt_d.step()
            opt_d.zero_grad(set_to_none=True)
            bundle['D'].requires_grad_(False)
            gb = batch('data')
            gf, decoded = fake_paths(bundle, gb, rng['latent'], rng['contact'], True)
            real = real_record(bundle, gb)
            gl, gt = adversarial_loss(bundle['D'], real, gf, gb['terrain'], gan, marginal_weight=cfg['marginal_weight'])
            task, terms = paired_loss(bundle, decoded, real, step, rng['error_noise'])
            prior_loss = spread(bundle['prior'].z)
            loss = task + prior_loss + cfg['adversarial_weight'] * gl
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite G loss at {step}')
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            opt_g.step()
            with torch.no_grad():
                for key in ('G', 'E', 'prior'):
                    for target, source in zip(ema[key].parameters(), bundle[key].parameters()):
                        target.lerp_(source, 1 - recipe.ema_decay)
            if step == 1 or step % cfg['log_interval'] == 0 or step in checkpoints:
                sync()
                row = dict(step=step, loss=float(loss.detach()), prior_loss=float(prior_loss.detach()),
                    d_loss=float(dl.detach()), g_adversarial_loss=float(gl.detach()),
                    error_d_loss=float(rl.detach()), error_sigma=bundle['R'].sigma(step),
                    **{k: float(v.detach()) for k,v in rt.items()},
                    **{f'd_{k}': float(v.detach()) for k,v in dt.items()},
                    **{f'g_{k}': float(v.detach()) for k,v in gt.items()},
                    **{k: float(v.detach()) for k,v in terms.items()}, lr_scale=scale,
                    elapsed_seconds=time.perf_counter() - started)
                metrics.write(json.dumps(row, allow_nan=False) + '\n')
                log(f'step={step}/{cfg["steps"]} action={row["action_loss"]:.5f} state={row["state_loss"]:.5f} '
                    f'next={row["next_loss"]:.5f} G_GAN={row["g_adversarial_loss"]:.5f} D={row["d_loss"]:.5f} '
                    f'error_G={row["error_g_adversarial"]:.5f} error_D={row["error_d_adversarial"]:.5f} '
                    f'cap={row["error_cap"]:.5f} sigma={row["error_sigma"]:.3f} elapsed_s={row["elapsed_seconds"]:.1f}')
            if step in checkpoints:
                sync()
                optimization_seconds += time.perf_counter() - segment
                if any(not torch.isfinite(p).all() for k in MODULE_KEYS for p in bundle[k].parameters()):
                    raise FloatingPointError('Nonfinite checkpoint parameters')
                saved = dict(format='gym_slider_gan_v1', gan_steps=step, step=step,
                    config=cfg, recipe=recipe.to_dict(), scaler=scaler.state_dict(), provenance=provenance,
                    data_draw_sha256={k: v.hexdigest() for k,v in digests.items()},
                    **{k: ema[k].state_dict() for k in MODULE_KEYS})
                torch.save(saved, out / f'checkpoint_{step}.pt')
                log(f'CHECKPOINT {step}; selection deferred to validation')
                sync()
                segment = time.perf_counter()
        shutil.copyfile(out / f'checkpoint_{cfg["steps"]}.pt', out / 'final.pt')
        summary = dict(config=cfg, provenance=provenance, gan_steps=cfg['steps'], parameters=counts,
            total_trainable_parameters=sum(counts.values()), inference_parameters=counts['E'] + counts['prior'] + parameter_count(bundle['G'].branches[1]),
            unique_training_records=len(triples), unique_labeled_episodes=len(np.unique(records['episode_ids'])),
            real_draws=2 * cfg['steps'] * cfg['batch_size'], simulator_calls=0,
            train_seconds=optimization_seconds, total_seconds=time.perf_counter() - started,
            final_parameters=hashes(bundle), data_draw_sha256={k: v.hexdigest() for k,v in digests.items()},
            checkpoints={p.name: sha256(p) for p in sorted(out.glob('*.pt'))})
        write_json(out / 'summary.json', summary)
        log(f'COMPLETE train_seconds={optimization_seconds:.2f}; awaiting rollout selection')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--slider-scope', choices=('all', 'action'))
    parser.add_argument('--steps', type=int)
    parser.add_argument('--device')
    parser.add_argument('--out-dir')
    args = parser.parse_args()
    cfg = {**DEFAULTS, **(read_config(args.config) if args.config else {})}
    for key in ('slider_scope', 'steps', 'device', 'out_dir'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    train(cfg)


if __name__ == '__main__':
    main()
