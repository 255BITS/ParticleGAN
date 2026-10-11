"""Reproduce bounded software compatibility checks, never scientific gates."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


SOURCES = {
    'develop': '0337bbbc2300fd351e77e9a222b92401137fd421',
    'proximal': '59d46d57bbc78acdeb486fba8c796256ab8015b5',
    'coulomb': '5719cf00b41043602cffa2ca550273ce9b7c3895',
    'sigreg': '40acfafa8eacfe7a0a8168eefd7797f28fbb4635',
    'moment_tether': '67badefcab620f32a8982dd9cd144f04ddac2c5d',
    'initial_mmd': 'dd677a69bf10887830a3af65582ecdfac934b22f',
}
EXISTING = [
    'test_prior_proximal', 'test_coulomb_prior', 'test_sigreg', 'test_moment_tether',
    'test_initial_mmd', 'test_recipe_owned_priors', 'test_recipe_defaults', 'test_training',
    'test_forge_priors', 'test_forge_configuration_search', 'test_forge_search_space',
    'test_forge_boundaries',
]
HOSTS = [
    'test_toy_api_vectors', 'test_toy_api_images', 'test_forge_behavior_adapters',
    'test_forge_word_joint', 'test_forge_word_smoke_hold', 'test_forge_word_provenance_cuda',
    'test_forge_gaussian_smoke', 'test_forge_gaussian_media', 'test_forge_initialization',
    'test_forge_techniques', 'test_forge_technique_bindings', 'test_forge_trainer_families',
    'test_recipe_prior_task_migration',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    archive = args.archive.resolve()
    archive.mkdir(parents=True, exist_ok=False)
    (archive / 'logs').mkdir()
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'PARTICLEGAN_PRIOR_REVIEW_FIXTURES': str(archive / 'fixtures')}

    def run(label, command):
        path = archive / 'logs' / (label + '.log')
        print('Running', label, 'log:', path, flush=True)
        with path.open('w') as out:
            subprocess.run(command, cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)

    for role, commit in SOURCES.items():
        target = archive / 'software-sources' / role
        target.mkdir(parents=True)
        contents = subprocess.check_output(
            ['git', 'archive', commit, 'particlegan', 'experiments', 'configs/forge'], cwd=root)
        with tarfile.open(fileobj=io.BytesIO(contents)) as tree:
            tree.extractall(target, filter='data')
        run('probe-' + role, [sys.executable, str(root / 'reports/forge/prior-integration-review/probe.py'),
            '--source', str(target), '--role', role, '--output', str(archive / 'fixtures' / (role + '.pt'))])
    (archive / 'software-sources.json').write_text(json.dumps(SOURCES, indent=2) + '\n')
    for label, names in [('combined-existing', EXISTING), ('combined-hosts', HOSTS),
                         ('combined-integration', ['test_prior_feature_integration'])]:
        run(label, [sys.executable, '-m', 'pytest', '-q', *['tests/' + name + '.py' for name in names],
                    '--junitxml=' + str(archive / 'logs' / (label + '.xml'))])
    print('All software checks passed; no scientific gates were executed.', flush=True)


if __name__ == '__main__':
    main()
