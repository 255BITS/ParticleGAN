"""Pin software fixtures as original v1 declarations before admission controls."""
from experiments.forge.contracts import atomic_json, file_hash, read_json


def pin_legacy(root):
    paths = sorted((root / 'configs/forge/ideas').glob('*.json'))
    atomic_json(root / 'configs/forge/legacy-ideas-v1.json', {
        'schema_version': 1, 'declarations': {str(path.relative_to(root)): file_hash(path)
                                           for path in paths if read_json(path)['schema_version'] == 1}})
