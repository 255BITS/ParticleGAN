"""Whole-table native prior geometry from certified saved states, without draws.

These final 7k-update table probes use float64 and the matched initialized
reference geometry. They are diagnostics, not historical minibatch forces.
"""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from particlegan import CoulombPriorRegularizer
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.knowledge import _attempts
from experiments.forge.state import state_digest

HERE = Path(__file__).resolve().parent
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/coulomb')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(row):
    descriptor = row['evidence']['provenance_checkpoint']
    path = Path(descriptor['artifact_root']) / descriptor['path']
    require(file_hash(path) == descriptor['sha256'], 'Checkpoint bytes changed')
    state = torch.load(path, map_location='cpu', weights_only=False)
    require(state_digest(state) == descriptor['state_sha256'], 'Checkpoint state changed')
    return state['trainer']['models']['prior'], descriptor['sha256']


def main():
    progress = read_json(ARCHIVE / 'progress.json')
    state = read_json(ARCHIVE / 'queue/queue/state.json')
    require(progress['phase'] == 'complete' and not any(e['status'] in ('queued', 'running', 'paused')
        for e in state['submissions'].values()) and not any(j['status'] == 'running' for j in state['jobs'].values()), 'Finish round first')
    spec = importlib.util.spec_from_file_location('saved_coulomb_workflow', HERE.parent / 'workflow.py')
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    originals, _ = _attempts(ROOT)
    attempts = {a['attempt_id']: a for a in originals}
    identities = [progress['control_request']] + progress['search_requests']
    groups = {}
    for identity in identities:
        entry = state['submissions'][identity]
        require(file_hash(ROOT / 'particlegan/coulomb_prior.py') == entry['request']['source']['files']['particlegan/coulomb_prior.py'], 'Frozen public loss source differs')
        weight = entry['request']['candidate']['recipe_overrides']['prior_reg']
        groups[weight] = {r['task_id']: r for r in workflow.validated_submission_rows(state, entry, attempts)
                         if r['task_id'] in ('grid100', 'rotated100', 'staggered100')}
    baseline, cells = groups[0.], []
    for weight, rows in sorted(groups.items()):
        if weight == 0.: continue
        for task, row in sorted(rows.items()):
            require(row['receipt_binding']['valid_receipt'] and baseline[task]['receipt_binding']['valid_receipt'], 'Invalid original certificate')
            active, active_sha = load(row)
            control, control_sha = load(baseline[task])
            center, radius = active['_coulomb_initial_center'].double(), active['_coulomb_initial_radius'].double()
            require(row['initialization'] == baseline[task]['initialization'], 'Matched initialization differs')
            for label, source, sha in (('control', control, control_sha), (f'prior_reg={weight}', active, active_sha)):
                rows64 = source['z'].double()
                q = (rows64 - center) / radius.clamp_min(1e-6)
                mean = rows64.mean(0)
                covariance = (rows64 - mean).T @ (rows64 - mean) / (len(rows64) - 1)
                energy = CoulombPriorRegularizer(center=center, radius=radius)(rows64)
                confinement = .35 / 2 * q.square().sum(1).mean()
                cells.append(dict(task_id=task, candidate_weight=weight, role=label,
                    checkpoint_sha256=sha, rows=len(rows64), initial_radius=float(radius),
                    initial_center=center.tolist(), rms_radius_ratio=float((rows64 - mean).square().sum(1).mean().sqrt() / radius.clamp_min(1e-6)),
                    center_displacement_over_initial_radius=float((mean - center).norm() / radius.clamp_min(1e-6)),
                    per_dimension_std=rows64.std(0).tolist(), covariance_eigenvalues=torch.linalg.eigvalsh(covariance).tolist(),
                    unweighted_full_table_energy=float(energy), confining_energy=float(confinement),
                    repulsive_energy=float(energy - confinement), sampling_draws_added=0,
                    target='Diagnostic; no declared qualification gate'))
    receipt = dict(schema_version=1, status='PASS', scope='saved_native_full_table_geometry_diagnostic',
        source_commit=progress['source_commit'], source_digest=progress['source_digest'],
        cells=cells, qualification_input=False, dtype='torch.float64', optimizer_updates_added=0,
        sampling_draws_added=0, model_forwards_added=0,
        law='Full saved 20,000-row table, deterministic evenly spaced cyclic-offset rule capped at k=8, softening=.1, confinement=.35. Numeric offsets depend on supplied row count. Same task-initial reference applied to candidate and its matched control. GANTrainer uses its full table for <=1024 particles and unique G-phase sampled rows above 1024; all native tasks here have 20,000 particles. Their training node IDs and offsets vary with supplied unique rows. These float64 whole-table probes are not historical minibatch energies or actual normalized update forces.',
        script_sha256=file_hash(Path(__file__)), public_loss_sha256=file_hash(ROOT / 'particlegan/coulomb_prior.py'))
    atomic_json(HERE / 'prior-geometry.json', receipt)
    print(f'PASS: {len(cells)} saved whole-table native geometry records; zero updates/draws')


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
