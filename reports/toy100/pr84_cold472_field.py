"""Read-only common-batch G-field comparison at the stopped cold ring fit.

The saved accepted critic and already evaluated best finite critic require no
additional D fit. Eight cloned native G minibatches are shared with the
unrestricted population log-p/q comparator. Known ring centers grade vectors
only after the fields are formed; they never select a critic or update.
"""

import argparse
import gzip
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys

import torch
from torch.func import functional_call, jvp


def digest(data):
    return hashlib.sha256(data).hexdigest()


def artifact_bytes(folder, name):
    path = folder / name
    if path.exists():
        return path.read_bytes()
    with gzip.open(folder / (name + '.gz'), 'rb') as handle:
        return handle.read()


def sharp_width(critic, clean):
    with torch.no_grad():
        acc = 0.
        for dim in range(2):
            shift = torch.zeros_like(clean)
            shift[:, dim] = 1e-3
            acc = acc + ((critic(clean + shift) - critic(clean - shift)) / 2e-3).square()
        sharp = float(acc.mean().sqrt())
    if not math.isfinite(sharp) or sharp <= 1e-6:
        raise RuntimeError('frozen PR84 critic stencil is inactive at the saved state')
    return sharp, min(.15, .5 / sharp)


def common_g_batches(saved, generator, prior, mode_hold):
    """Native mode_hold G call order: prior, noisy fake, then fresh real."""
    stream = torch.Generator()
    stream.set_state(saved['rng']['data'])
    rows = []
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.set_rng_state(saved['rng']['torch'])
        for batch in range(8):
            latent, indices = prior.sample(mode_hold.BATCH, generator=stream)
            clean = generator(latent)
            noise = torch.randn_like(clean)
            real = mode_hold.sample_ring(mode_hold.ring_means(), mode_hold.BATCH,
                                         mode_hold.SIGMA, stream)
            fingerprint = digest(b''.join(value.detach().contiguous().cpu().numpy().tobytes()
                                          for value in (indices, noise, real)))
            rows.append(dict(batch=batch, indices=indices.detach(), noise=noise.detach(),
                             real=real.detach(), sha256=fingerprint))
    return rows


def vector_grade(vector, radial, toward_missing):
    inward = -(vector.double() * radial.double()).sum(-1)
    missing_work = (vector.double() * toward_missing.double()).sum(-1)
    return dict(vectors=vector.detach().tolist(), inward_work=inward.tolist(),
                inward_particles=int((inward > 0).sum()),
                mean_inward_work=float(inward.mean()),
                toward_nearest_missing_particles=int((missing_work > 0).sum()),
                mean_toward_nearest_missing_work=float(missing_work.mean()),
                rms=float(vector.double().square().sum(-1).mean().sqrt()))


def cosine(a, b):
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    divisor = float(a.norm() * b.norm())
    return float(a @ b / divisor) if divisor > 0 else None


def fields(saved, critic, row, width, gan, sigma, means, fit):
    generator, _, prior = fit.modules(saved)
    network = list(generator.parameters())
    params = network + list(prior.parameters())
    names = list(dict(generator.named_parameters()))
    clean = generator(prior.z).detach()
    distances = torch.cdist(clean, means)
    nearest = distances.argmin(1)
    radial = clean - means[nearest]
    represented = set(nearest[distances.min(1).values <= .21].tolist())
    missing = [index for index in range(len(means)) if index not in represented]
    if not missing:
        raise RuntimeError('saved ring state unexpectedly has no missing mode')
    missing_centers = means[missing]
    toward_missing = missing_centers[torch.cdist(clean, missing_centers).argmin(1)] - clean
    fake = generator(prior.z[row['indices']]) + sigma * row['noise']
    fake.retain_grad()
    loss = gan.g_loss(fit.smooth(critic, fake), fit.smooth(critic, row['real']))
    values = torch.autograd.grad(loss, [fake] + params)
    local = torch.zeros_like(clean).index_add(0, row['indices'], -values[0].detach())
    g_net = values[1:-1]
    g_prior = values[-1]

    def clean_function(network_values, z):
        return functional_call(generator, dict(zip(names, network_values)), (z,))

    zero_network = tuple(torch.zeros_like(p) for p in network)
    _, net_raw = jvp(clean_function, (tuple(network), prior.z),
                     (tuple(-g for g in g_net), torch.zeros_like(prior.z)))
    _, prior_raw = jvp(clean_function, (tuple(network), prior.z),
                       (zero_network, -g_prior))
    joint_raw = net_raw + prior_raw
    proposal, _, _ = fit.g_proposal(saved, critic,
                                   dict(indices=row['indices'], noise=row['noise'],
                                        real=row['real'], sigma=sigma),
                                   gan, saved['noise_policy']['_step_calls'])
    reported_raw = torch.tensor(proposal['raw_joint_parameter_descent']['vectors'])
    if not torch.allclose(joint_raw, reported_raw, atol=1e-4, rtol=1e-4):
        raise RuntimeError('independent joint raw JVP disagrees with frozen proposal observer')
    accepted = torch.tensor(proposal['accepted_joint']['vectors'])
    # `toward_missing` points from support to the target: the new vector to
    # target is target - (support + accepted), hence toward_missing - accepted.
    missing_distance_change = ((toward_missing - accepted).norm(dim=1)
                               - toward_missing.norm(dim=1))
    next_clean = clean + accepted
    next_distances = torch.cdist(next_clean, means)
    next_nearest = next_distances.argmin(1)
    assigned_distance_change = ((next_clean - means[nearest]).norm(dim=1)
                                - (clean - means[nearest]).norm(dim=1))
    return dict(loss=float(loss.detach()),
                batch_sha256=row['sha256'],
                local_output=vector_grade(local, radial, toward_missing),
                network_raw=vector_grade(net_raw.detach(), radial, toward_missing),
                prior_raw=vector_grade(prior_raw.detach(), radial, toward_missing),
                joint_raw=vector_grade(joint_raw.detach(), radial, toward_missing),
                accepted_adam=vector_grade(accepted, radial, toward_missing),
                accepted_nearest_missing_distance_change=missing_distance_change.tolist(),
                accepted_assigned_mode_distance_change=assigned_distance_change.tolist(),
                clean_assignment_before=torch.bincount(nearest, minlength=len(means)).tolist(),
                clean_assignment_after=torch.bincount(next_nearest, minlength=len(means)).tolist(),
                clean_particles_changing_nearest_mode=int((next_nearest != nearest).sum()),
                clean_hq_count_before=int((distances.min(1).values <= .21).sum()),
                clean_hq_count_after=int((next_distances.min(1).values <= .21).sum()),
                local_vs_network_cosine=cosine(local, net_raw),
                local_vs_joint_cosine=cosine(local, joint_raw),
                raw_network_gradient_norm=sum(float(g.double().square().sum())
                                              for g in g_net) ** .5,
                raw_prior_gradient_norm=float(g_prior.double().norm()),
                rho=proposal['rho'], factor=proposal['factor'],
                posthoc_fixed_grade_after=proposal['grade_after'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    declaration = json.loads(artifact_bytes(args.original, 'declaration.json'))
    for name, expected in declaration['source'].items():
        if digest((root / name).read_bytes()) != expected:
            raise RuntimeError(f'original source mismatch: {name}')
    sys.path.insert(0, str(root))
    from benchmarks.locked_shared import mode_hold
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
    from reports.toy100 import pr84_critic_relaxation as fit
    from reports.toy100 import pr84_population_field as population
    from reports.toy100.pr84_prediction_state_filter import state_hash

    torch.set_num_threads(1)
    failure = json.loads(artifact_bytes(args.capture, 'failure.json'))
    raw = artifact_bytes(args.capture, 'failed-fit.pt')
    if digest(raw) != failure['state_file_sha256']:
        raise RuntimeError('failed-fit capture bytes changed')
    payload = torch.load(BytesIO(raw), weights_only=True)
    if payload['host_update'] != 472 or payload['host_update'] != failure['host_update']:
        raise RuntimeError('wrong stopped fit')
    saved = payload['post_accepted_d']
    if state_hash(saved) != failure['accepted_d_sha256']:
        raise RuntimeError('accepted-D state changed')
    if saved['noise']['input_sigma'] != 0 or saved['noise']['output_sigma'] != .029 or saved['rng']['output'] is not None:
        raise RuntimeError('saved G-noise law differs from declared native mode_hold replay')
    config = json.loads(artifact_bytes(args.original, 'config.json'))
    recipe, _, _ = declared_recipe(config)
    gan, regularizer = recipe.make_loss(), recipe.make_gradient_penalty()
    outer_rng = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        generator, d_star, prior = fit.modules(saved)
        clean = generator(prior.z).detach()
        d_best = fit.modules(saved)[1]
        d_best.load_state_dict(payload['best'])
        means = mode_hold.ring_means()
        clean_distances = torch.cdist(clean, means)
        clean_nearest = clean_distances.argmin(1)
        represented_modes = sorted(set(clean_nearest[clean_distances.min(1).values <= .21].tolist()))
        missing_modes = [index for index in range(len(means)) if index not in represented_modes]
        sharp_star, width_star = sharp_width(d_star, clean)
        sharp_best, width = sharp_width(d_best, clean)
        # A guarded continuation would freeze the stencil after restoring
        # this saved best finite D. Use its width for all three comparisons.
        if width <= 0:
            raise RuntimeError('best-point G stencil unexpectedly inactive')
        rows = common_g_batches(saved, generator, prior, mode_hold)
        q_support = clean.detach().clone()
        ideal = lambda x: population.ratio_score(x, means, mode_hold.SIGMA,
                                                  q_support, saved['noise']['output_sigma'])
        variants = dict(accepted_d=d_star, best_finite_d=d_best,
                        unregularized_population_ratio=ideal)
        bank = payload['bank']
        pre_d = fit.modules(payload['pre_step'])[1]
        first_d_bank = {name: bank[name][:mode_hold.BATCH]
                        for name in ('real', 'fake')}
        first_d_loss = fit.d_loss(pre_d, first_d_bank, gan, regularizer, 472)[0]
        first_d_gradient = fit.gradients(first_d_loss, pre_d)
        recorded_adam_gradient = [saved['optimizer_d']['state'][index]['exp_avg']
                                  for group in saved['optimizer_d']['param_groups']
                                  for index in group['params']]
        if not all(torch.equal(left, right) for left, right in
                   zip(first_d_gradient, recorded_adam_gradient)):
            raise RuntimeError('first captured D bank is not the native host D batch')
        bank_loss = {}
        for name, critic in variants.items():
            total, logistic, penalty = fit.d_loss(critic, bank, gan, regularizer, 472)
            bank_loss[name] = dict(total=float(total.detach()),
                                   logistic=float(logistic.detach()),
                                   b_cap_penalty=float(penalty.detach()))
        original_width = fit.WIDTH
        try:
            fit.WIDTH = width
            arms = {name: [fields(saved, critic, row, width, gan,
                                  saved['noise']['output_sigma'], means, fit)
                           for row in rows] for name, critic in variants.items()}
        finally:
            fit.WIDTH = original_width
        summaries = {}
        for name, batch_rows in arms.items():
            summaries[name] = {
                key: dict(inward_particles=sum(x[key]['inward_particles'] for x in batch_rows),
                          total_particles=8 * len(clean),
                          mean_inward_work=sum(x[key]['mean_inward_work'] for x in batch_rows) / 8,
                          toward_nearest_missing_particles=sum(
                              x[key]['toward_nearest_missing_particles'] for x in batch_rows),
                          mean_toward_nearest_missing_work=sum(
                              x[key]['mean_toward_nearest_missing_work'] for x in batch_rows) / 8)
                for key in ('local_output', 'network_raw', 'prior_raw', 'joint_raw', 'accepted_adam')}
            summaries[name]['posthoc_grade_after'] = [x['posthoc_fixed_grade_after']
                                                       for x in batch_rows]
            changes = [value for row in batch_rows
                       for value in row['accepted_nearest_missing_distance_change']]
            summaries[name]['accepted_nearest_missing_distance'] = dict(
                decreases=sum(value < 0 for value in changes), total=len(changes),
                mean_change=sum(changes) / len(changes))
            assigned_changes = [value for row in batch_rows
                                for value in row['accepted_assigned_mode_distance_change']]
            summaries[name]['accepted_assigned_mode_distance'] = dict(
                decreases=sum(value < 0 for value in assigned_changes),
                total=len(assigned_changes),
                mean_change=sum(assigned_changes) / len(assigned_changes))
            summaries[name]['clean_assignment_changes'] = sum(
                row['clean_assignment_before'] != row['clean_assignment_after']
                for row in batch_rows)
            summaries[name]['clean_particles_changing_nearest_mode'] = sum(
                row['clean_particles_changing_nearest_mode'] for row in batch_rows)
            summaries[name]['clean_hq_count_after'] = [row['clean_hq_count_after']
                                                       for row in batch_rows]
    if not torch.equal(outer_rng, torch.get_rng_state()):
        raise RuntimeError('diagnostic changed the external global RNG')
    result = dict(scope='one stopped ring state, eight paired G batches, no D fit or outer training',
                  host_update=472, capture_raw_sha256=digest(raw),
                  first_d_gradient_bitwise_equal=True,
                  source=dict(original=declaration['source'],
                              population_field_sha256=digest((root / 'reports/toy100/pr84_population_field.py').read_bytes()),
                              fixed_eval_sha256=digest((root / 'reports/toy100/coverage_fixed_eval.py').read_bytes()),
                              diagnostic_sha256=digest(Path(__file__).read_bytes())),
                  runtime=dict(torch=torch.__version__, threads=torch.get_num_threads(),
                               cpu_capability=torch.backends.cpu.get_cpu_capability()),
                  sigma=dict(real=mode_hold.SIGMA, fake=saved['noise']['output_sigma'],
                             input=saved['noise']['input_sigma']),
                  population_scope='unrestricted unregularized log p/q; q support detached; not the penalized finite-MLP optimum',
                  stencil=dict(shared_width=width, d_best_sharpness=sharp_best,
                               d_star_own_width=width_star, d_star_sharpness=sharp_star),
                  clean_support=clean.tolist(), nearest_mode=clean_nearest.tolist(),
                  represented_clean_modes=represented_modes, missing_clean_modes=missing_modes,
                  batch_sha256=[row['sha256'] for row in rows],
                  bank_loss=bank_loss, summaries=summaries, arms=arms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='DONE', host_update=472, width=width,
                          bank_loss=bank_loss, summaries=summaries), allow_nan=False))


if __name__ == '__main__':
    main()
