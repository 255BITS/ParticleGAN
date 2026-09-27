#!/usr/bin/env python3
"""Seal compact research preparations; preserve expanded sources for launch."""
from pathlib import Path
import hashlib
import json
import zipfile

HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    prepared=json.loads((HERE/'prepared-index.json').read_text())
    destination=HERE/'prepared-bundles';destination.mkdir(exist_ok=True)
    rows=[]
    for row in prepared['rows']:
        root=Path(row['directory']);manifest=root/'manifest.json'
        assert sha(manifest)==row['manifest_sha256']
        files=json.loads(manifest.read_text())['files']|{'manifest.json':sha(manifest)}
        for name,want in files.items():assert sha(root/name)==want
        archive=destination/(root.name+'.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
            for name,want in sorted(files.items()):
                item=zipfile.ZipInfo(name,(2026,9,27,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED
                bundle.writestr(item,(root/name).read_bytes())
        row['directory_relative']=str(root.relative_to(HERE))
        row['required_review_relative']=str(Path(row['required_review']).relative_to(HERE))
        rows.append(dict(queue_row=row['queue_row'],candidate=row['candidate'],directory_relative=row['directory_relative'],archive=str(archive.relative_to(HERE)),archive_sha256=sha(archive),manifest_sha256=row['manifest_sha256'],files=files))
    (HERE/'prepared-index.json').write_text(json.dumps(prepared,indent=2)+'\n')
    value=dict(schema=1,scope='Exact source preparations only; CPU checks and quality remain separate. Expanded prepared/ is local and reproducibly restorable.',rows=rows)
    (destination/'index.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(bundles=len(rows),zip_bytes=sum((HERE/r['archive']).stat().st_size for r in rows))))

if __name__=='__main__':main()
