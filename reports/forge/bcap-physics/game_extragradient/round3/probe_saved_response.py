"""Deterministic finite response at archived Gaussian states; no training.

This simultaneous full-table cubature probe is explicitly distinct from the
alternating sampled-row training map. It does not estimate the antisymmetric
Jacobian or grant a quality gate. Archive identity and immutable input hashes
are retained. Local changes are discarded after each restored-state probe.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtri
import torch

ROOT = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location('archive_probe', ROOT / 'reports/forge/bcap-tier2-search/probe_failure_states.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    memory = json.loads((ROOT / 'reports/forge/bcap-tier2-search/failure-state-analysis.json').read_text())
    rows = []
    for artifact in memory['artifact_proofs']:
        if artifact['task'] != 'gaussian1d_stability' or not artifact['path'].endswith('state.pt'):
            continue
        path = Path(artifact['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256']
        state = torch.load(path, map_location='cpu', weights_only=True)
        before = archive.state_digest(state)
        g, d = archive.restore(state['trainer']['models'])
        table = torch.nn.Parameter(state['trainer']['models']['prior']['z'].double().clone())
        sigma = float(state['trainer']['models']['prior']['sigma'])
        offsets = torch.cat((torch.eye(2), -torch.eye(2))).double() * np.sqrt(2) * sigma
        count = 4 * len(table)
        mean = 3. if state['trainer']['completed_steps'] == 6000 else 2.
        real = torch.tensor(mean + .5 * ndtri((np.arange(count) + .5) / count))[:, None]
        parameters = [*d.parameters(), *g.parameters(), table]
        split = len(list(d.parameters()))
        def field():
            fake = g((table[:, None] + offsets).flatten(0, 1))
            ld = archive.GANLoss('non_saturating').d_loss(d(real), d(fake.detach()))
            ld = ld + archive.GradientPenalty(arm='b_cap', coeff=1., kappa=1.)(d, real, fake.detach())
            dg = torch.autograd.grad(ld, list(d.parameters()))
            lg = archive.GANLoss('non_saturating').g_loss(d(fake))
            gg = torch.autograd.grad(lg, [*g.parameters(), table])
            return [archive.direction(v, .018) for v in dg] + [
                archive.direction(v, .03 if i == len(gg)-1 else .012, prior=i == len(gg)-1)
                for i, v in enumerate(gg)]
        first = field()
        with torch.no_grad():
            for p, step in zip(parameters, first):
                p.add_(step)
        second = field()
        response = {}
        for role, a, b in [('D', first[:split], second[:split]),
                           ('G', first[split:-1], second[split:-1]),
                           ('prior', first[-1:], second[-1:])]:
            a, b = archive.flatten(a), archive.flatten(b)
            response[role] = dict(cosine=archive.cosine(a,b), predictor_norm=float(a.norm()),
                                  corrected_norm=float(b.norm()), relative_response=float((b-a).norm()/a.norm()))
        assert archive.state_digest(state) == before
        rows.append(dict(input=artifact, completed_steps=state['trainer']['completed_steps'], response=response))
    result = dict(schema_version=1, scope=__doc__, original_candidate_id=memory['candidate_id'],
                  original_source_digest=memory['source_digest'], protocol_seed=0,
                  optimizer_updates_added=0, random_draws_added=0,
                  seconds=time.monotonic()-started, rows=rows)
    (Path(__file__).parent / 'saved-response.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
