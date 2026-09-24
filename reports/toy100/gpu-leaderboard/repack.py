"""Create deterministic replay source archives from the isolated GPU copies."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
manifest = {}
for row in json.loads((root / 'candidates.json').read_text()):
    repo = Path(row['repo'])
    files = sorted(p for p in repo.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    buffer = io.BytesIO()
    hashes = {}
    with tarfile.open(fileobj=buffer, mode='w') as stream:
        for path in files:
            data = path.read_bytes()
            name = str(path.relative_to(repo))
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            stream.addfile(info, io.BytesIO(data))
            hashes[name] = hashlib.sha256(data).hexdigest()
    archive = root / 'archives' / (row['name'] + '.tar.gz')
    archive.parent.mkdir(exist_ok=True)
    archive.write_bytes(gzip.compress(buffer.getvalue(), mtime=0))
    manifest[row['name']] = dict(archive=archive.name, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                                files=len(hashes), source_hashes=hashes)
(root / 'source-archives.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Archived', len(manifest), 'candidates')
