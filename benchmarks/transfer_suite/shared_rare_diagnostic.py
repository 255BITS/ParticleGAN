"""Read-only per-particle replay of an archived rare-component failure.

The wrappers observe evaluation samples after the original scorer runs. They do
not alter inputs, scores, gradients, random streams, or the training update.
"""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from particlegan import ParticlePrior

from . import shared_pointnorm_search as runner, suite, vector_tasks
from .compare_defaults import plan, write
from .replay_shared_architecture import REPLAY_KEYS, numerical


def inspect_samples(fake, sample_ids, spec):
    means = torch.tensor(spec['means'], dtype=fake.dtype)
    covs = torch.tensor(spec['covariances'], dtype=fake.dtype)
    labels = torch.cdist(fake, means).argmin(1)
    first_position = {}
    for position, particle in enumerate(sample_ids.tolist()):
        first_position.setdefault(particle, position)
    unique_positions = torch.tensor(list(first_position.values()), dtype=torch.long)
    unique_points = fake[unique_positions]
    unique_labels = labels[unique_positions]
    unique_ids = sample_ids[unique_positions]
    rows = []
    for k in range(len(means)):
        points = fake[labels == k]
        particles = unique_points[unique_labels == k]
        ids = unique_ids[unique_labels == k]
        inverse = torch.linalg.inv(torch.linalg.cholesky(covs[k]))
        def geometry(p):
            if len(p) < 2:
                return dict(covariance=None, normalized_eigenvalues=[0., 0.])
            centered = p-p.mean(0)
            empirical = centered.T @ centered / len(p)
            eigen = torch.linalg.eigvalsh(inverse @ empirical @ inverse.T)
            return dict(covariance=empirical.tolist(), normalized_eigenvalues=eigen.tolist())
        raw_geometry = geometry(points)
        unique_geometry = geometry(particles)
        rows.append(dict(component=k, samples=len(points), unique_particles=len(particles),
                         sampled=raw_geometry, unique=unique_geometry,
                         particle_ids=ids.tolist(), particle_points=particles.tolist()))
    return dict(component=rows, sampled_particle_multiplicity=dict(Counter(sample_ids.tolist())))


def run(reference, output):
    expected = json.loads(gzip.decompress(reference.read_bytes()))
    card = expected['discriminator_variant']['overrides']['research_discriminator']
    job = next(job for job in plan() if job['spec']['name'] == expected['spec']['name'])
    assert job['spec']['name'] == 'vector_unequal_mass'
    assert expected['original_spec'] == job['spec']
    assert expected['recipe'] == json.loads(json.dumps(runner.recipe().to_dict()))
    assert runner.variant(card) == expected['discriminator_variant']
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    protocol = suite.snapshot(output)
    protocol.update(reference=str(reference.resolve()), reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                    purpose='Read-only support diagnosis; no selection points or training feedback')
    write(output/'protocol.json', protocol)
    sample_cache = {}
    diagnostics = []
    original_sample = ParticlePrior.sample
    original_score = vector_tasks.score_samples

    def sample(self, batch_size, generator=None, **kwargs):
        answer = original_sample(self, batch_size, generator, **kwargs)
        if batch_size == vector_tasks.EVAL_SAMPLES:
            sample_cache['indices'] = answer[1].detach().clone()
        return answer

    def score(fake, spec, completed_steps):
        answer = original_score(fake, spec, completed_steps)
        diagnostic = inspect_samples(fake.detach(), sample_cache['indices'], spec)
        diagnostic.update(step=completed_steps, stream='live' if len(diagnostics) % 2 == 0 else 'ema',
                          original_metrics=deepcopy(answer))
        diagnostics.append(diagnostic)
        return answer

    with patch.object(ParticlePrior, 'sample', sample), patch.object(vector_tasks, 'score_samples', score):
        actual = runner.episode(job, deepcopy(card))
    suite.verify_source(protocol)
    # Archived JSON stores recipe tuples as lists, so compare JSON-normalized data.
    actual = json.loads(json.dumps(actual, allow_nan=False))
    checks = {key: numerical(actual[key]) == numerical(expected[key]) for key in REPLAY_KEYS}
    write(output/'checks.json', dict(checks=checks, reference_sha256=protocol['reference_sha256']))
    write(output/'diagnostics.json', dict(rows=diagnostics))
    assert all(checks.values()), 'Replay differs from archived evidence; inspect checks.json'
    print(json.dumps(dict(checks=checks, verdict=actual['verdict'],
                          late_live=[dict(step=d['step'], component=[dict(component=c['component'],
                              unique_particles=c['unique_particles'], normalized_eigenvalues=c['unique']['normalized_eigenvalues'])
                              for c in d['component']]) for d in diagnostics if d['stream']=='live' and d['step'] >= 1000])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.reference, args.output)
