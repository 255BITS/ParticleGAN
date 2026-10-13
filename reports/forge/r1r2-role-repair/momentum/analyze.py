"""Saved endpoint moments and fixed-panel gradients, without models, draws or training."""
from pathlib import Path
import hashlib
import json
import random
import torch

ROOT = Path(__file__).resolve().parents[4]
TASKS = ('gaussian1d_smoke', 'ring16_acquisition', 'five_word_joint_smoke')
PARENT = 'r1r2--169e9539b2dcfaab6b51e17f02907f28ea5b894f77155f7e688ab07a789de2b9'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def forward(x, weights, indices=(0, 2, 4)):
    for index in indices:
        x = x @ weights[f'net.{index}.weight'].double().T + weights[f'net.{index}.bias'].double()
        if index != indices[-1]:
            x = torch.nn.functional.leaky_relu(x, .2)
    return x

def critic_gradient(points, weights):
    points = points.detach().double().requires_grad_(True)
    inputs = points
    if 'freqs' in weights:
        phases = points.unsqueeze(-1) * weights['freqs'].double()
        inputs = torch.cat((points, phases.sin().flatten(1), phases.cos().flatten(1)), 1)
    gradient = torch.autograd.grad(forward(inputs, weights).sum(), points)[0]
    norms = gradient.norm(dim=1)
    return dict(panel_size=len(points), input_gradient_norm_mean=float(norms.mean()),
                input_gradient_norm_max=float(norms.max()), zero_gradient_rows=int((norms == 0).sum()))

def moments(state, word):
    results = []
    roles = ('G', 'E', 'prior') if word else ('G', 'prior')
    keys = dict(G='generator', E='encoder', D='critic', prior='prior') if word else dict(G='G', D='D', prior='prior')
    for optimizer_index, optimizer in enumerate(state['optimizers']):
        for role, group in zip(roles if optimizer_index == 0 else ('D',), optimizer['param_groups']):
            parameters = [value for name, value in state['models'][keys[role]].items()
                          if isinstance(value, torch.Tensor) and state['requires_grad'][keys[role]].get(name, False)]
            assert len(parameters) == len(group['params'])
            m_sq, v_sum, delta_sq, weight_sq, zeros, count, steps = 0., 0., 0., 0., 0, 0, set()
            for parameter, index in zip(parameters, group['params']):
                item = optimizer['state'][index]
                m, v = item['exp_avg'].double(), item['exp_avg_sq'].double()
                assert parameter.shape == m.shape == v.shape
                step = int(item['step'])
                steps.add(step)
                beta1, beta2 = group['betas']
                direction = m / (1 - beta1 ** step) / ((v / (1 - beta2 ** step)).sqrt() + group['eps'])
                m_sq += float(m.square().sum())
                v_sum += float(v.sum())
                delta_sq += float((group['lr'] * direction).square().sum())
                weight_sq += float(parameter.double().square().sum())
                zeros += int((m == 0).sum())
                count += m.numel()
            results.append(dict(role=role, betas=list(group['betas']), lr=group['lr'], steps=sorted(steps),
                parameter_count=count, zero_first_moment_fraction=zeros/count,
                first_moment_l2=m_sq ** .5, second_moment_rms=(v_sum/count) ** .5,
                saved_moment_displacement_l2=delta_sq ** .5,
                saved_moment_displacement_over_endpoint_weight=(delta_sq/max(weight_sq, 1e-300)) ** .5,
                first_moment_kind='instantaneous last training gradient' if group['betas'][0] == 0 else 'exponentially averaged training gradients; instantaneous gradient unavailable'))
    return results

def word_panel(models):
    chars, words = 'abcdefghijklmnopqrstuvwxyz_ ', ('apple', 'grape', 'lemon', 'melon', 'berry')
    indices = torch.tensor([[chars.index(char) for char in word + '_'] for word in words])
    bank = torch.nn.functional.one_hot(indices, len(chars)).permute(0, 2, 1).double()
    prior = models['prior']['z'].double()
    generated = forward(prior, models['generator']).reshape(5, len(chars), 6).softmax(1)
    encoded = forward(bank.flatten(1), models['encoder'], (1, 3, 5))
    reconstruction = forward(encoded, models['generator']).reshape(5, len(chars), 6).softmax(1)
    names = lambda p: [''.join(chars[index] for index in row) for row in p.argmax(1).tolist()]
    joins_real = torch.cat((bank.flatten(1), encoded), 1)
    joins_fake = torch.cat((generated.flatten(1), prior), 1)
    distances = torch.pdist(prior)
    return dict(cohort='fixed five canonical inputs and saved prior means; no sampling or grading',
        prior_pair_distance_min=float(distances.min()), prior_pair_distance_max=float(distances.max()),
        encoded_pair_distance_min=float(torch.pdist(encoded).min()),
        generated_by_saved_row=names(generated), reconstructed_by_canonical_input=names(reconstruction),
        generated_min_token_max_probability=float(generated.max(1).values.min()),
        real_joint_critic=critic_gradient(joins_real, models['critic']),
        fake_joint_critic=critic_gradient(joins_fake, models['critic']))

def main():
    before_python, before_torch = random.getstate(), torch.get_rng_state().clone()
    rows, saved_streams = [], {}
    for label, publication in [('parent', 'reports/forge/r1r2-tier1-search/publication/results.json'),
                               ('momentum', 'reports/forge/r1r2-five-track-repair/publication/results.json')]:
        for result in read(ROOT/publication)['certified_results']:
            keep = result['candidate_id'] == PARENT if label == 'parent' else result['candidate_id'] == 'r1r2-five-track-momentum-v1'
            if not keep or result['task_id'] not in TASKS:
                continue
            archive = Path(result['checkpoint']['artifact_root']).parent
            path = archive/'provenance/provenance-state.pt'
            assert sha(path) == result['checkpoint']['sha256']
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert result['source_digest'] == '25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507'
            saved_streams[label, result['task_id']] = saved['streams']
            word = result['task_id'] == 'five_word_joint_smoke'
            state = saved['fixture']['api_state'] if word else saved['trainer']
            raw = read(archive/'raw-result.json')
            observations = raw['evidence']['observations']
            row = dict(recipe=label, task_id=result['task_id'], attempt_id=result['attempt_id'],
                candidate_revision=result['candidate_revision'], source_digest=result['source_digest'],
                checkpoint_sha256=sha(path), request_sha256=sha(archive/'request.json'),
                raw_result_sha256=sha(archive/'raw-result.json'), gate_status=result['gate_status'],
                endpoint_metrics=result['metrics'], optimizer_moments=moments(state, word),
                instantaneous_training_gradient_history='UNAVAILABLE: no optimizer-diagnostics trace or stored parameter gradients; beta1=.5 first moments are history averages')
            if word:
                row['fixed_panel'] = word_panel(state['models'])
                best = max(range(len(observations)), key=lambda index: (observations[index]['modes'], observations[index]['quality_fraction']))
                row['selected_word_observations'] = [{k: observations[index][k] for k in ('step','modes','mass_tv','quality_fraction','reconstruction_exact')} for index in sorted({0, best, len(observations)-1})]
            else:
                sample_path = archive/('observed-samples.pt' if result['task_id']=='ring16_acquisition' else 'evaluator/observed-samples.pt')
                samples = torch.load(sample_path, map_location='cpu', weights_only=True)[-1]['samples']
                row['sample_sha256'] = sha(sample_path)
                row['saved_output_critic_gradient'] = critic_gradient(samples, state['models']['D'])
                if result['task_id'] == 'gaussian1d_smoke':
                    row['gaussian_best_observed_cdf_ks'] = min(o['cdf_ks'] for o in observations)
                    row['gaussian_best_cdf_observation'] = min(observations,key=lambda o:o['cdf_ks'])
            rows.append(row)
    paired_cursors = {}
    for task in TASKS:
        left, right = (saved_streams[label, task] for label in ('parent', 'momentum'))
        selected = {key for key, binding in left['manifest']['bindings'].items()
                    if binding['family'] in ('prior', 'data', 'noise')}
        paired_cursors[task] = {key: torch.equal(left['states'][key], right['states'][key])
                                for key in sorted(selected)}
        assert all(paired_cursors[task].values())
    assert before_python == random.getstate() and torch.equal(before_torch, torch.get_rng_state())
    print(json.dumps(dict(schema_version=1, analysis='six saved endpoint role/moment and deterministic function probes',
        training_updates_added=0, sampling_draws_added=0, python_rng_unchanged=True, torch_cpu_rng_unchanged=True,
        paired_final_training_stream_cursor_equality=paired_cursors,
        limits_of_stream_proof='Training stream cursors match; instantaneous sampled row IDs and gradient support are not separately stored.',
        inspected_source_hashes={path: sha(ROOT/path) for path in ('particlegan/recipes.py', 'particlegan/recipe_schedules.py', 'experiments/forge/behavior_adapters.py', 'benchmarks/toy_audit/api_images.py', 'lib/toy_models.py')},
        limitations='Endpoint moments describe different trained states, not isolated role causality. Reconstructed moment displacement is a float64 formula estimate, not independently captured pre/post displacement. Fixed means/word panels are diagnostic cohorts and do not substitute task sampling or gates.', rows=rows),indent=2,sort_keys=True))

if __name__ == '__main__':
    main()
