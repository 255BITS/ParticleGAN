"""Saved critic parameter-gradient diagnostic; no models, RNG draws or updates.

Sixteen declared target centers are a separate diagnostic fixture. Saved 4096
fake outputs are evaluation samples, never claimed contemporaneous training pairs.
Functional analytic input Jacobians remain differentiable in saved D parameters.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from particlegan.recipes import Recipe

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / 'reports/forge/r1r2-score-law-repair'
HELPER = ROOT / 'reports/forge/r1r2-objective-memory-repair/research/score-law/probe.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                      for p, g in zip(parameters, gradients, strict=True)])


def objectives(real, fake, saved, law, critic):
    sr, jr = critic(real, saved)
    sf, jf = critic(fake, saved)
    if law == 'relativistic':
        classification = F.softplus(sf[:, None] - sr[None, :]).mean()
    else:
        assert law == 'non_saturating'
        classification = F.softplus(-sr).mean() + F.softplus(sf).mean()
    return classification, .5 * jr.square().sum(1).mean(), .5 * jf.square().sum(1).mean()


def balance(row, critic):
    attempt = Path(row['checkpoint']['artifact_root']).parent
    state_path, panels_path = attempt / 'provenance/provenance-state.pt', attempt / 'observed-samples.pt'
    assert sha(state_path) == row['checkpoint']['sha256']
    request = json.loads((attempt/'request.json').read_text())['request']
    assert all(sha(ROOT/path)==digest for path,digest in request['source']['files'].items())
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    raw = json.loads((attempt / 'raw-result.json').read_text())
    assert sha(panels_path) == raw['evidence']['saved_observer_outputs']['sha256']
    fake = torch.load(panels_path, map_location='cpu', weights_only=True)[-1]['samples'].double()
    saved = {name: value.double().clone().requires_grad_(name.startswith('net.'))
             for name, value in state['trainer']['models']['D'].items()}
    names = [name for name in sorted(saved) if name.startswith('net.')]
    parameters = [saved[name] for name in names]
    theta = torch.arange(16, dtype=torch.float64) * (2 * torch.pi / 16)
    real = 3 * torch.stack((theta.cos(), theta.sin()), dim=1)
    sr, jr = critic(real, saved)
    sf, jf = critic(fake, saved)
    recipe = state['trainer']['recipe']
    law = recipe.get('loss', Recipe.__dataclass_fields__['loss'].default)
    classification, r1, r2 = objectives(real, fake, saved, law, critic)
    cls, first, second = [vector(loss, parameters) for loss in (classification, r1, r2)]
    errors, offset = [], 0
    for name, parameter in zip(names,parameters,strict=True):
        for index in dict.fromkeys((0,parameter.numel()-1)):
            perturbed=[]
            for sign in (-1,1):
                fixture={k:v.detach().clone() for k,v in saved.items()}
                fixture[name].flatten()[index]+=sign*1e-7
                perturbed.append([float(v) for v in objectives(real,fake,fixture,law,critic)])
            for number,gradient in enumerate((cls,first,second)):
                numerical=(perturbed[1][number]-perturbed[0][number])/2e-7
                errors.append(abs(numerical-float(gradient[offset+index])))
        offset+=parameter.numel()
    assert max(errors)<=5e-7, max(errors)
    gamma = recipe['reg_coeff']
    penalty = gamma * (first + second)
    norm = lambda v: float(torch.linalg.vector_norm(v))
    cosine = lambda a, b: float(torch.dot(a, b) / (a.norm() * b.norm()))
    return dict(role=row['role'], candidate_id=row['candidate_id'], candidate_revision=row['candidate_revision'],
        attempt_id=row['attempt_id'], loss=law, gamma=gamma,
        scientific_source_files_verified=len(request['source']['files']),
        parameter_gradient_check=dict(central_difference_coordinates=len(errors)//3,
            objective_derivative_checks=len(errors), epsilon=1e-7,
            maximum_absolute_error=max(errors),absolute_error_bound=5e-7),
        artifacts_sha256=dict(checkpoint=sha(state_path), panels=sha(panels_path), raw_result=sha(attempt/'raw-result.json')),
        fixture_counts=dict(real_centers=len(real), saved_fake_outputs=len(fake)),
        losses=dict(classification=float(classification.detach()), raw_r1=float(r1.detach()), raw_r2=float(r2.detach()),
                    gamma_weighted_penalty=float((gamma * (r1 + r2)).detach())),
        parameter_gradients=dict(classification_norm=norm(cls), raw_r1_norm=norm(first), raw_r2_norm=norm(second),
            gamma_weighted_penalty_norm=norm(penalty), weighted_penalty_over_classification=norm(penalty)/norm(cls),
            classification_penalty_cosine=cosine(cls, penalty), r1_r2_cosine=cosine(first, second),
            total_norm=norm(cls+penalty), parameter_count=len(cls)),
        penalty_input_gradients=dict(real_mean_squared=float(jr.detach().square().sum(1).mean()),
                                    fake_mean_squared=float(jf.detach().square().sum(1).mean())))


def main():
    torch.set_num_threads(1)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    spec = importlib.util.spec_from_file_location('saved_analytic_critic', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    publication = json.loads((OUT/'publication/results.json').read_text())
    assert publication['complete'] and publication['source_commit'] == '517257ff3879516ae2da64009c375ffffe9b001a'
    rows = [row for row in publication['certified_results']
            if row['task_id']=='ring16_acquisition' and row['role'] in ('ams_m110','non_saturating')]
    assert len(rows)==2
    records = [balance(row, helper.critic) for row in rows]
    after = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert before[0]==after[0] and before[1][0]==after[1][0] and np.array_equal(before[1][1],after[1][1])
    assert before[1][2:]==after[1][2:] and torch.equal(before[2],after[2])
    print(json.dumps(dict(schema_version=1,evidence_type='separate_saved_critic_gradient_fixture',qualification_input=False,
        publication_sha256=sha(OUT/'publication/results.json'),helper_sha256=sha(HELPER),probe_sha256=sha(Path(__file__)),
        source_digest=publication['source_digest'],records=records,
        updates_added=0,sampling_draws_added=0,python_numpy_torch_cpu_rng_unchanged=True,
        limitations='Target centers omit target Gaussian noise; saved fake panels are evaluation outputs. '
            'Raw parameter gradients are not native AMSGrad updates or historical forces. '
            'Endpoint balance cannot identify a useful gamma or explain earlier trajectory/shape. '
            'Pair expectation differs from a realized same-size training pairing; no grade or task law changes.'),
        indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':
    main()
