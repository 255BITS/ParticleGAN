"""Restore exact reviewed port sources from compact, hash-pinned archives."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='append', help='Port directory; omit to restore every bundle')
    parser.add_argument('--with-provenance', action='store_true')
    args = parser.parse_args()
    folder = ROOT / 'port-bundles'
    index = json.loads((folder / 'index.json').read_text())
    names = args.candidate or list(index['bundles'])
    entries = [index['bundles'][name] for name in names]
    if args.with_provenance:
        entries.append(index['provenance'])
    restored = unchanged = 0
    for entry in entries:
        archive_path = folder / entry['file']
        assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == entry['sha256']
        with zipfile.ZipFile(archive_path) as archive:
            assert len(archive.namelist()) == len(set(archive.namelist())) == entry['members']
            for name in archive.namelist():
                relative = PurePosixPath(name)
                assert not relative.is_absolute() and '..' not in relative.parts
                target = ROOT / 'port-source' / name
                assert not target.is_symlink()
                data = archive.read(name)
                if target.exists():
                    assert target.read_bytes() == data, f'Refusing to overwrite changed source: {target}'
                    unchanged += 1
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    assert target.resolve().is_relative_to((ROOT / 'port-source').resolve())
                    target.write_bytes(data)
                    restored += 1
    print(json.dumps(dict(restored=restored, identical_existing=unchanged, bundles=len(entries))))


if __name__ == '__main__':
    main()
