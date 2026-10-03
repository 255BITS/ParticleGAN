"""Compose separately verified stability, performance and sampling contributions."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'pkg-CB64-RA2/particlegan'
SOURCE = ROOT / 'stability/pkg/particlegan'


def nodes(text, owner=None):
    tree = ast.parse(text)
    return tree.body if owner is None else next(n.body for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == owner)


def fragment(text, name, owner=None):
    node = next(n for n in nodes(text, owner) if isinstance(n,
        (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
    return ''.join(text.splitlines(keepends=True)[start-1:node.end_lineno])


def replace(text, name, replacement, owner=None):
    old = fragment(text, name, owner)
    assert text.count(old) == 1
    return text.replace(old, replacement, 1)


def main():
    feature = (TARGET / 'feature_cells.py').read_text()
    original = feature
    stable = (SOURCE / 'feature_cells.py').read_text()
    protected = {name: fragment(feature, name, owner) for name, owner in
        [('BoundedLatentGeometry', None), ('conditional_count_pvalues', None),
         ('fit', 'FeatureCellSnapshot'), ('_pool', 'FeatureCellSnapshot')]}
    helpers = '\n\n'.join(fragment(stable, n) for n in
        ('population_policy', 'SmallPopulationReferenceBirthDeath', 'make_feature_cell_birth_death'))
    feature = feature.replace('import weakref\n', 'import weakref\nfrom fractions import Fraction\n')
    feature = feature.replace('\n\nclass BoundedLatentGeometry:', '\n\n'+helpers+'\n\nclass BoundedLatentGeometry:', 1)
    mass_helpers = '\n\n'.join(fragment(stable, n, 'FeatureCellSnapshot') for n in
        ('_mass_targets', '_mass_topology', '_group_counts'))
    old_selection = fragment(feature, 'select_parents', 'FeatureCellSnapshot')
    feature = feature.replace(old_selection, mass_helpers+'\n\n'+fragment(stable,
        'select_parents', 'FeatureCellSnapshot'), 1)
    feature = replace(feature, 'ordinary_transport', fragment(stable,
        'ordinary_transport', 'FeatureCellSnapshot'), 'FeatureCellSnapshot')

    birth_death = fragment(stable, 'FeatureCellBirthDeath')
    for name in ('_jitter', 'perturb_latent', 'load_state_dict'):
        birth_death = replace(birth_death, name,
            fragment(original, name, 'FeatureCellBirthDeath'), 'FeatureCellBirthDeath')
    birth_death = birth_death.replace('real_anchors=1,parent_rank=None,jitter_std=JITTER_STD,jitter_cap=JITTER_CAP,',
        'real_anchors=1,parent_rank=None,\n'
        '                             latent_kernel="bounded_local_dv12",latent_neighbors=PARENT_RESERVOIR,\n'
        '                             latent_rank=recipe.birth_death_metric_rank,')
    birth_death = birth_death.replace('        self.snapshot = None\n',
        '        self.latent_geometry = BoundedLatentGeometry(rank=recipe.birth_death_metric_rank,\n'
        '                                                   neighbors=PARENT_RESERVOIR, chunk=recipe.birth_death_chunk)\n'
        '        self._sampling_prior, self._sampling_controller = trainer.prior, trainer.controller\n'
        '        self.snapshot = None\n', 1)
    feature = replace(feature, 'FeatureCellBirthDeath', birth_death)
    for name, owner in [('BoundedLatentGeometry', None), ('conditional_count_pvalues', None),
                        ('fit', 'FeatureCellSnapshot'), ('_pool', 'FeatureCellSnapshot')]:
        assert fragment(feature, name, owner) == protected[name], name
    assert 'jitter_std=JITTER_STD' not in fragment(feature, '__init__', 'FeatureCellBirthDeath')

    training = (TARGET / 'training.py').read_text()
    training_before = training
    training = training.replace('from .feature_cells import FeatureCellBirthDeath\n'
        '                self.birth_death = FeatureCellBirthDeath(self, seed + 6)',
        'from .feature_cells import make_feature_cell_birth_death\n'
        '                self.birth_death = make_feature_cell_birth_death(self, seed + 6)')
    training = replace(training, '_generate', fragment((SOURCE / 'training.py').read_text(),
        '_generate', 'GANTrainer'), 'GANTrainer')
    for name, text in [('feature_cells.py', feature), ('training.py', training)]:
        compile(ast.parse(text), name, 'exec')
        (TARGET / name).write_text(text)
    patches = []
    for name, before, after in [('feature_cells.py', original, feature),
                                 ('training.py', training_before, training)]:
        patches.extend(difflib.unified_diff(before.splitlines(keepends=True),
            after.splitlines(keepends=True), fromfile='a/particlegan/'+name,
            tofile='b/particlegan/'+name))
    (ROOT / 'integration/STABILITY-COMPOSED.diff').write_text(''.join(patches))
    receipt = dict(status='COMPOSED', performance_and_geometry_functions_preserved=True,
        final_source_sha256={name: hashlib.sha256((TARGET/name).read_bytes()).hexdigest()
            for name in ('feature_cells.py', 'training.py')},
        active_schema=3, selected_operator_dispatch=True,
        quality_execution_started=False)
    (ROOT / 'integration/COMPOSITION.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
