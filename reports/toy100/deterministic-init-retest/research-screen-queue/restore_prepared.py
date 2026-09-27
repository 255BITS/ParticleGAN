#!/usr/bin/env python3
"""Restore exact local source preparations from retained archives; no execution."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import zipfile

HERE=Path(__file__).resolve().parent
def digest(data):return hashlib.sha256(data).hexdigest()

def main():
    index=json.loads((HERE/'prepared-bundles/index.json').read_text())
    for row in index['rows']:
        archive=HERE/row['archive'];assert digest(archive.read_bytes())==row['archive_sha256']
        root=HERE/row['directory_relative']
        assert not PurePosixPath(row['directory_relative']).is_absolute() and '..' not in PurePosixPath(row['directory_relative']).parts
        with zipfile.ZipFile(archive) as bundle:
            assert set(bundle.namelist())==set(row['files'])
            for name,want in row['files'].items():
                path=PurePosixPath(name);assert not path.is_absolute() and '..' not in path.parts
                data=bundle.read(name);assert digest(data)==want
                target=root/name
                if target.exists():assert digest(target.read_bytes())==want,'Refuse to overwrite changed source: '+str(target)
                else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        assert digest((root/'manifest.json').read_bytes())==row['manifest_sha256']
    print(json.dumps(dict(restored_or_verified=len(index['rows']),training=False)))

if __name__=='__main__':main()
