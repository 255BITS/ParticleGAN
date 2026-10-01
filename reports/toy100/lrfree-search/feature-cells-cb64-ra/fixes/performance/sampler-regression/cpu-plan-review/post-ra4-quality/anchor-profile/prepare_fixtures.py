"""Copy two fixed saved clean-table cases for a root-only GPU microprofile."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from fixture_utils import load_package, network, import_file, sha

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
OWNER = ROOT / 'integration/review/training-regression/post-ra4-quality'
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-root', type=Path, default=ROOT / 'pkg-CB64-RA6')
    args = parser.parse_args()
    target = HERE / 'inputs.pt'
    assert not target.exists()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    module, birth = load_package(args.package_root, 'fixture_package')
    contracts = import_file(OWNER / 'birth_contract_cases.py', 'fixed_birth_contracts')
    states = [ROOT / f'validation-ra4/learned/training/toy/CB64-RA4/checkpoint-{step:04d}.pt' for step in (1250, 2000)]
    paths = [*sorted((args.package_root / 'particlegan').rglob('*.py')), *states,
        OWNER / 'birth_contract_cases.py', Path(__file__), HERE / 'fixture_utils.py']
    before = {str(p): sha(p) for p in paths}
    rng = torch.get_rng_state().clone()
    cases = []
    for path in states:
        state = torch.load(path, map_location='cpu', weights_only=False)['trainer']
        weights = state['models']
        G, D, ema_G = (network(weights[name]) for name in ('G', 'D', 'ema_G'))
        trainer = SimpleNamespace(G=G, D=D)
        controller = SimpleNamespace(_heads=[D[4]], sample_shape=(2,))
        feature = birth.learned_latent_features(controller, trainer, G)
        z, ez = weights['prior']['z'], weights['ema_prior']['z']
        with torch.no_grad():
            q = feature(z).double()
            real = D[:4](state['birth_death']['reservoir']).double()
            snapshot = module.FeatureCellSnapshot.fit(real,
                generator=torch.Generator().set_state(state['cpu_rng']), cells=64, rank=8, chunk=256)
            flags, pvalues, _ = snapshot.support(q)
            stream = torch.Generator().set_state(state['cpu_rng'])
            law, child, parent, supported, phases = contracts.copy_phases(snapshot,
                dict(q=q, flags=flags, fake_features=q), stream, pvalues, 51)
        cases.append(dict(step=state['completed_steps'], snapshot=deepcopy(vars(snapshot)),
            models={name: deepcopy(weights[name]) for name in ('G', 'D', 'ema_G')},
            q=q, flags=flags, pvalues=pvalues, comparison=law, latents=z, ema_latents=ez,
            previous_children=child, previous_copy_parents=parent, supported_counts=supported,
            max_moves=51, scope='saved clean-table mechanical case, not historical emitted or copied action replay'))
    assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    assert before == {str(p): sha(p) for p in paths}
    torch.save(dict(cases=cases, sources=before, new_seeds=0), target)
    receipt = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), source_sha256=before,
        inputs_path=str(target), inputs_sha256=sha(target), cases=[c['step'] for c in cases],
        global_rng_unchanged=True, cuda_initialized=False, training_steps=0, new_seeds=0)
    (HERE / 'FIXTURE-RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', cases=receipt['cases'], inputs_sha256=receipt['inputs_sha256'])))


if __name__ == '__main__':
    main()
