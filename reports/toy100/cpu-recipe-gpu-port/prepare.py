"""Prepare immutable baseline sources and explicit CPU-RNG diagnostic copies."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(archive, target, *, identical_overlap=False, json_only=False):
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError('unsafe archive member')
            if json_only and path.suffix != '.json':
                continue
            dest = target / path
            data = stream.extractfile(member).read()
            if dest.exists():
                if identical_overlap and dest.read_bytes() != data:
                    raise ValueError('inconsistent original sources: ' + member.name)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)


def prepare(root):
    bundle = Path(__file__).resolve().parent
    toy100 = bundle.parent
    archives = {
        'cuda': toy100 / 'gpu-known-winner-control/archives/simpler22_reference.tar.gz',
        'cpu_older': toy100 / 'simpler22/candidate19/source.tar.gz',
        'cpu_native': toy100 / 'simpler22/toy100/grid100/source.tar.gz',
    }
    expected = json.loads((bundle / 'archive-hashes.json').read_text())
    for name, archive in archives.items():
        if digest(archive) != expected[name]:
            raise ValueError('archive checksum mismatch: ' + name)
    root.mkdir(parents=True, exist_ok=False)
    for name in ('cpu', 'cuda', 'cuda_cpu_random'):
        repo = root / 'repos' / name
        repo.mkdir(parents=True)
        if name == 'cpu':
            extract(archives['cpu_older'], repo)
            extract(archives['cpu_native'], repo, identical_overlap=True)
            extract(archives['cuda'], repo, json_only=True)
        else:
            extract(archives['cuda'], repo)
        if name == 'cuda_cpu_random':
            for path in repo.rglob('*.py'):
                source = path.read_text()
                lines = source.splitlines(keepends=True)
                edits = []
                for node in ast.walk(ast.parse(source)):
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and isinstance(node.func.value, ast.Name)
                            and node.func.value.id == 'torch' and node.func.attr == 'Generator'):
                        start = sum(map(len, lines[:node.lineno-1])) + node.col_offset
                        end = sum(map(len, lines[:node.end_lineno-1])) + node.end_col_offset
                        edits.append((start, end, 'torch.Generator(device="cpu")'))
                for start, end, replacement in sorted(edits, reverse=True):
                    source = source[:start] + replacement + source[end:]
                source = source.replace(
                    '(torch.cuda.default_generators[torch.cuda.current_device()] if torch.get_default_device().type == "cuda" else torch.default_generator)',
                    'torch.default_generator')
                if str(path.relative_to(repo)) == 'particlegan/training.py':
                    assert source.count('if device != self.device:') == 1
                    source = source.replace('if device != self.device:',
                        'if device.type != "cpu":  # Diagnostic random-draw bridge; all model math stays CUDA.')
                path.write_text(source)
    (root / 'repos/cuda_cpu_init').symlink_to('cuda', target_is_directory=True)
    manifest = {name: {str(p.relative_to(root / 'repos' / name)): digest(p)
                      for p in (root / 'repos' / name).rglob('*') if p.is_file()}
                for name in ('cpu', 'cuda', 'cuda_cpu_random')}
    (root / 'prepared-sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    print(prepare(parser.parse_args().root.resolve()))
